import numpy as np
import matplotlib.pyplot as plt
import cv2
import textwrap
from matplotlib.backends.backend_pdf import PdfPages
import scipy.fftpack

# Твои функции rle_encode, rle_decode, dct2, idct2, zigzag, CompressBlock, DecompressBlock, CompressLayer, DecompressLayer
# (берём их из твоего кода без изменений)

def rle_encode(data):
    flat = data.flatten()
    buffer = np.zeros(flat.size * 2 + len(data.shape) + 1, dtype=int)  
    buffer[0] = len(data.shape)  
    buffer[1:1 + len(data.shape)] = data.shape  
    idx = 1 + len(data.shape)
    i = 0
    while i < flat.size:
        run_val = flat[i]
        run_length = 1
        while i + run_length < flat.size and flat[i + run_length] == run_val:
            run_length += 1
        buffer[idx] = run_length
        buffer[idx + 1] = run_val
        idx += 2
        i += run_length
    return buffer[:idx] 

def rle_decode(encoded):
    num_dims = encoded[0]
    shape = tuple(encoded[1:1 + num_dims])
    rle_data = encoded[1 + num_dims:]
    decoded = []
    i = 0
    while i < len(rle_data):
        count = rle_data[i]
        value = rle_data[i + 1]
        decoded.extend([value] * count)
        i += 2
    return np.array(decoded).reshape(shape)

def dct2(a):
    return scipy.fftpack.dct(scipy.fftpack.dct(a.astype(float), axis=0, norm='ortho'), axis=1, norm='ortho')

def idct2(a):
    return scipy.fftpack.idct(scipy.fftpack.idct(a.astype(float), axis=0, norm='ortho'), axis=1, norm='ortho')

def zigzag(A):
    template = np.array([
        [0,  1,  5,  6,  14, 15, 27, 28],
        [2,  4,  7,  13, 16, 26, 29, 42],
        [3,  8,  12, 17, 25, 30, 41, 43],
        [9,  11, 18, 24, 31, 40, 44, 53],
        [10, 19, 23, 32, 39, 45, 52, 54],
        [20, 22, 33, 38, 46, 51, 55, 60],
        [21, 34, 37, 47, 50, 56, 59, 61],
        [35, 36, 48, 49, 57, 58, 62, 63],
    ])
    if len(A.shape) == 2:
        B = np.zeros((64,))
        for r in range(8):
            for c in range(8):
                B[template[r, c]] = A[r, c]
    else:
        B = np.zeros((8, 8))
        for r in range(8):
            for c in range(8):
                B[r, c] = A[template[r, c]]
    return B

def CompressBlock(block, Q):
    dct = dct2(block - 128)
    qd = np.round(dct / Q).astype(int)
    return zigzag(qd)

def DecompressBlock(vector, Q):
    qd = zigzag(vector)
    dct = qd * Q
    block = np.round(idct2(dct) + 128).clip(0, 255).astype(np.uint8)
    return block

def CompressLayer(L, Q):
    S = []
    for w in range(0, L.shape[0], 8):
        for k in range(0, L.shape[1], 8):
            block = L[w:w+8, k:k+8]
            if block.shape != (8,8):
                padded_block = np.zeros((8,8))
                padded_block[:block.shape[0], :block.shape[1]] = block
                block = padded_block
            S.append(CompressBlock(block, Q))
    return np.concatenate(S)

def DecompressLayer(S, Q, shape):
    h, w = shape
    L = np.zeros((h, w), dtype=np.uint8)
    blocks_per_row = w // 8
    for idx, i in enumerate(range(0, len(S), 64)):
        vector = S[i:i+64]
        col = (idx % blocks_per_row) * 8
        row = (idx // blocks_per_row) * 8
        block = DecompressBlock(vector, Q)
        L[row:row+block.shape[0], col:col+block.shape[1]] = block[:min(8, h - row), :min(8, w - col)]
    return L.clip(0, 255).astype(np.uint8)

QY = np.array([
    [16, 11, 10, 16, 24,  40,  51,  61],
    [12, 12, 14, 19, 26,  58,  60,  55],
    [14, 13, 16, 24, 40,  57,  69,  56],
    [14, 17, 22, 29, 51,  87,  80,  62],
    [18, 22, 37, 56, 68, 109, 103,  77],
    [24, 36, 55, 64, 81, 104, 113,  92],
    [49, 64, 78, 87,103, 121, 120, 101],
    [72, 92, 95, 98,112, 100, 103,  99],
])

QC = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
])

QOnes = np.ones((8,8))

def chroma_subsampling(Y, Cb, Cr, mode="4:4:4"):
    if mode == "4:4:4":
        return Y, Cb, Cr
    elif mode == "4:2:2":
        Cb_sub = cv2.resize(Cb, (Cb.shape[1]//2, Cb.shape[0]), interpolation=cv2.INTER_AREA)
        Cr_sub = cv2.resize(Cr, (Cr.shape[1]//2, Cr.shape[0]), interpolation=cv2.INTER_AREA)
        return Y, Cb_sub, Cr_sub
    else:
        raise ValueError("Only 4:4:4 and 4:2:2 supported")

def chroma_upsampling(Cb_sub, Cr_sub, shape, mode="4:4:4"):
    if mode == "4:4:4":
        return Cb_sub, Cr_sub
    elif mode == "4:2:2":
        Cb_up = cv2.resize(Cb_sub, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        Cr_up = cv2.resize(Cr_sub, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        return Cb_up, Cr_up
    else:
        raise ValueError("Only 4:4:4 and 4:2:2 supported")

def CompressJPEG(RGB, chroma_mode="4:4:4", use_quant_table=True):
    YCrCb = cv2.cvtColor(RGB, cv2.COLOR_RGB2YCrCb)
    Y, Cr, Cb = YCrCb[:,:,0], YCrCb[:,:,1], YCrCb[:,:,2]  

    Y = Y.astype(np.float32)
    Cb = Cb.astype(np.float32)
    Cr = Cr.astype(np.float32)

    Y, Cb_sub, Cr_sub = chroma_subsampling(Y, Cb, Cr, chroma_mode)

    if use_quant_table:
        Qy_used = QY
        Qc_used = QC
    else:
        Qy_used = QOnes
        Qc_used = QOnes

    JPEG = {}
    JPEG['shape'] = RGB.shape
    JPEG['chroma_mode'] = chroma_mode
    JPEG['use_quant_table'] = use_quant_table

    JPEG['Y'] = CompressLayer(Y, Qy_used)
    JPEG['Cb'] = CompressLayer(Cb_sub, Qc_used)
    JPEG['Cr'] = CompressLayer(Cr_sub, Qc_used)

    JPEG['Cb_shape'] = Cb_sub.shape
    JPEG['Cr_shape'] = Cr_sub.shape

    rle_Y = rle_encode(JPEG['Y'])
    rle_Cb = rle_encode(JPEG['Cb'])
    rle_Cr = rle_encode(JPEG['Cr'])

    JPEG['rle_sizes'] = {
        'Y_original': JPEG['Y'].size,
        'Y_rle': rle_Y.size,
        'Cb_original': JPEG['Cb'].size,
        'Cb_rle': rle_Cb.size,
        'Cr_original': JPEG['Cr'].size,
        'Cr_rle': rle_Cr.size,
    }

    return JPEG

def DecompressJPEG(JPEG):
    shape = JPEG['shape']
    chroma_mode = JPEG['chroma_mode']

    Y = DecompressLayer(JPEG['Y'], QY if JPEG['use_quant_table'] else QOnes, (shape[0], shape[1]))
    Cb_sub = DecompressLayer(JPEG['Cb'], QC if JPEG['use_quant_table'] else QOnes, JPEG['Cb_shape'])
    Cr_sub = DecompressLayer(JPEG['Cr'], QC if JPEG['use_quant_table'] else QOnes, JPEG['Cr_shape'])

    Cb, Cr = chroma_upsampling(Cb_sub, Cr_sub, (shape[0], shape[1]), chroma_mode)

    YCrCb = np.stack([Y, Cr, Cb], axis=2).astype(np.uint8)
    RGB = cv2.cvtColor(YCrCb, cv2.COLOR_YCrCb2RGB)
    return RGB

def extract_patches(img, patch_size=128, patches_count=1):
    h, w = img.shape[:2]
    patches = []
    coords = [
        (0, 0),
        (0, max(0, w - patch_size)),
        (max(0, h//2 - patch_size//2), max(0, w//2 - patch_size//2)),
        (max(0, h - patch_size), 0)
    ]
    for y, x in coords[:patches_count]:
        patch = img[y:y+patch_size, x:x+patch_size]
        if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
            patches.append((patch, (y,x)))
    return patches

def compare_on_images(image_paths, pdf):
    for img_path in image_paths:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        patches = extract_patches(img)
        print(f"Image: {img_path} size: {img.shape}, patches: {len(patches)}")

        for idx, (patch, (y,x)) in enumerate(patches):
            fig, axes = plt.subplots(2, 4, figsize=(16,8))
            fig.suptitle(f"Image: {img_path} Patch #{idx+1} at ({y},{x})")

            variants = [
                ("4:4:4", True),
                ("4:4:4", False),
                ("4:2:2", True),
                ("4:2:2", False)
            ]

            for i, (chroma_mode, use_quant) in enumerate(variants):
                JPEG = CompressJPEG(patch, chroma_mode=chroma_mode, use_quant_table=use_quant)
                decompressed = DecompressJPEG(JPEG)

                rle_sizes = JPEG['rle_sizes']
                compression_ratio_Y = rle_sizes['Y_original'] / rle_sizes['Y_rle']
                compression_ratio_Cb = rle_sizes['Cb_original'] / rle_sizes['Cb_rle']
                compression_ratio_Cr = rle_sizes['Cr_original'] / rle_sizes['Cr_rle']

                axes[0, i].imshow(patch)
                axes[0, i].set_title(f"Orig\n{chroma_mode}, Q={use_quant}")
                axes[0, i].axis('off')
                axes[1, i].imshow(decompressed)
                axes[1, i].set_title(f" RLE Compression ratios: Y={compression_ratio_Y:.2f}\n Cb={compression_ratio_Cb:.2f}\n Cr={compression_ratio_Cr:.2f}")
                axes[1, i].axis('off')

            pdf.savefig(fig)
            plt.close(fig)

image_paths = [
    'image1.png',
    'image2.png',
    'image3.png',
    'image4.png',
]

with PdfPages('comparison_results.pdf') as pdf:
    compare_on_images(image_paths, pdf)
