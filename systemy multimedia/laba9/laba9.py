
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


VIDEO_DIR = 'clips'
PLOT_DIR = 'plots'
os.makedirs(PLOT_DIR, exist_ok=True)

VIDEO_FILES = [f for f in os.listdir(VIDEO_DIR) if f.endswith(('.mp4', '.avi'))]
NUM_FRAMES = 7
KEYFRAME_INTERVALS = [1, 2, 4]
CHROMA_MODES = ['4:4:4', '4:2:2', '4:4:0', '4:2:0', '4:1:1', '4:1:0']
DIFF_DIVISORS = [2]
ROI = (50, 150, 50, 150)


def rgb_to_ycbcr(img): return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
def ycbcr_to_rgb(img): return cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)

def chroma_subsampling(cb, cr, mode):
    if mode == '4:4:4': return cb, cr
    if mode == '4:2:2': return cb[:, ::2], cr[:, ::2]
    if mode == '4:4:0': return cb[::2, :], cr[::2, :]
    if mode == '4:2:0': return cb[::2, ::2], cr[::2, ::2]
    if mode == '4:1:1': return cb[:, ::4], cr[:, ::4]
    if mode == '4:1:0': return cb[::4, ::4], cr[::4, ::4]
    return cb, cr

def chroma_upsampling(cb_sub, mode, shape): return cv2.resize(cb_sub, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)

def rle_encode(channel):
    flat = channel.flatten()
    encoded, prev, count = [], flat[0], 1
    for pixel in flat[1:]:
        if pixel == prev: count += 1
        else: encoded.append((prev, count)); prev, count = pixel, 1
    encoded.append((prev, count))
    return encoded

def rle_decode(encoded):
    flat = []
    for val, count in encoded:
        flat.extend([val] * count)
    return np.array(flat, dtype=np.uint8)

def plotDiffrence(ReferenceFrame, DecompressedFrame, ROI, pdf_pages):
    w1, w2, k1, k2 = ROI
    ref_crop = ReferenceFrame[w1:w2, k1:k2].astype(float)
    dec_crop = DecompressedFrame[w1:w2, k1:k2].astype(float)
    diff = ref_crop - dec_crop

    fig, axs = plt.subplots(1, 3, sharey=True)
    fig.set_size_inches(16, 5)

    axs[0].imshow(ref_crop.astype(np.uint8))
    axs[0].set_title("Reference (RGB)")

    axs[1].imshow(diff, cmap='seismic', vmin=-50, vmax=50)
    axs[1].set_title(f"Difference\n(min: {np.min(diff):.1f}, max: {np.max(diff):.1f})")

    axs[2].imshow(dec_crop.astype(np.uint8))
    axs[2].set_title("Decompressed (RGB)")

    for ax in axs:
        ax.axis('off')

    plt.tight_layout()
    pdf_pages.savefig(fig)
    plt.close()


def compute_metrics(original_frames, reconstructed_frames):
    mae, mse, psnr = [], [], []
    for orig, recon in zip(original_frames, reconstructed_frames):
        diff = orig.astype(np.float32) - recon.astype(np.float32)
        err = np.mean(np.abs(diff)); sq_err = np.mean(np.square(diff))
        mae.append(err); mse.append(sq_err)
        psnr.append(99.0 if sq_err == 0 else 20 * np.log10(255.0 / np.sqrt(sq_err)))
    return {'MAE': mae, 'MSE': mse, 'PSNR': psnr}

def plot_metrics(metrics, pdf_pages, title="Compression Metrics Over Time"):
    plt.figure(figsize=(12, 6))
    for metric_name, values in metrics.items():
        plt.plot(values, label=metric_name)

    plt.xlabel("Frame")
    plt.ylabel("Metric Value")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    pdf_pages.savefig()
    plt.close()


def process_video(video_path, chroma_mode, diff_divisor, keyframe_interval, use_rle=False):
    cap = cv2.VideoCapture(video_path)
    frames, keyframes, diffs = [], {}, {}
    frame_idx = 0
    while frame_idx < NUM_FRAMES:
        ret, frame = cap.read()
        if not ret: break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_ycbcr = rgb_to_ycbcr(frame_rgb)
        y, cb, cr = cv2.split(frame_ycbcr)
        cb_sub, cr_sub = chroma_subsampling(cb, cr, chroma_mode)

        if frame_idx % keyframe_interval == 0:
            keyframes[frame_idx] = (y, cb_sub, cr_sub)
            encoded = (y, cb_sub, cr_sub)
        else:
            key_y, key_cb, key_cr = keyframes[frame_idx - frame_idx % keyframe_interval]
            dy = ((y.astype(np.int16) - key_y.astype(np.int16)) // diff_divisor).astype(np.int8)
            dcb = ((cb_sub.astype(np.int16) - key_cb.astype(np.int16)) // diff_divisor).astype(np.int8)
            dcr = ((cr_sub.astype(np.int16) - key_cr.astype(np.int16)) // diff_divisor).astype(np.int8)
            if use_rle:
                dy, dcb, dcr = rle_encode(dy), rle_encode(dcb), rle_encode(dcr)
            diffs[frame_idx] = (dy, dcb, dcr)
            encoded = (dy, dcb, dcr)

        frames.append((frame_rgb, encoded))
        frame_idx += 1
    cap.release()
    return frames, keyframes, diffs

def reconstruct_video(frames, keyframes, diffs, chroma_mode, diff_divisor, keyframe_interval, use_rle=False):
    reconstructed_frames = []
    for idx in range(len(frames)):
        if idx % keyframe_interval == 0:
            y, cb_sub, cr_sub = keyframes[idx]
        else:
            dy, dcb, dcr = diffs[idx]
            key_y, key_cb, key_cr = keyframes[idx - idx % keyframe_interval]
            if use_rle:
                dy = rle_decode(dy).reshape(key_y.shape)
                dcb = rle_decode(dcb).reshape(key_cb.shape)
                dcr = rle_decode(dcr).reshape(key_cr.shape)
            y = (key_y.astype(np.int16) + (dy.astype(np.int16) * diff_divisor)).astype(np.uint8)
            cb_sub = (key_cb.astype(np.int16) + (dcb.astype(np.int16) * diff_divisor)).astype(np.uint8)
            cr_sub = (key_cr.astype(np.int16) + (dcr.astype(np.int16) * diff_divisor)).astype(np.uint8)
        cb = chroma_upsampling(cb_sub, chroma_mode, y.shape)
        cr = chroma_upsampling(cr_sub, chroma_mode, y.shape)
        frame_rgb = ycbcr_to_rgb(cv2.merge((y, cb, cr)))
        reconstructed_frames.append(frame_rgb)
    return reconstructed_frames

def main():
    with PdfPages(os.path.join(PLOT_DIR, "all_plots.pdf")) as pdf:
        for keyframe_interval in KEYFRAME_INTERVALS:
            for video_file in VIDEO_FILES:
                video_path = os.path.join(VIDEO_DIR, video_file)
                for chroma_mode in CHROMA_MODES:
                    for diff_divisor in DIFF_DIVISORS:
                        base = os.path.splitext(video_file)[0] + f"_kf{keyframe_interval}_mode-{chroma_mode}_div-{diff_divisor}"

                        frames, keys, diffs = process_video(video_path, chroma_mode, diff_divisor, keyframe_interval, use_rle=False)
                        orig_frames = [f[0] for f in frames]
                        rec = reconstruct_video(frames, keys, diffs, chroma_mode, diff_divisor, keyframe_interval, use_rle=False)

                        if keyframe_interval + 1 < len(rec):
                            plotDiffrence(rec[keyframe_interval], rec[keyframe_interval + 1], ROI, pdf)

                        plot_metrics(compute_metrics(orig_frames, rec), pdf, title=base + " (no RLE)")

                        frames_rle, keys_rle, diffs_rle = process_video(video_path, chroma_mode, diff_divisor, keyframe_interval, use_rle=True)
                        orig_frames_rle = [f[0] for f in frames_rle]
                        rec_rle = reconstruct_video(frames_rle, keys_rle, diffs_rle, chroma_mode, diff_divisor, keyframe_interval, use_rle=True)

                        if keyframe_interval + 1 < len(rec_rle):
                            plotDiffrence(rec_rle[keyframe_interval], rec_rle[keyframe_interval + 1], ROI, pdf)

                        plot_metrics(compute_metrics(orig_frames_rle, rec_rle), pdf, title=base + " (RLE)")


if __name__ == '__main__':
    main()

