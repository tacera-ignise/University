import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import os

def compute_metrics(original, distorted):
    original = original.astype(np.float32)
    distorted = distorted.astype(np.float32)

    mse_val = np.mean((original - distorted) ** 2)
    nmse_val = mse_val / np.mean(original ** 2) if np.mean(original ** 2) != 0 else 0
    psnr_val = 10 * np.log10(255 ** 2 / mse_val) if mse_val != 0 else float('inf')
    denom = np.sum(original * distorted)
    if_val = 1 - np.sum((original - distorted) ** 2) / denom if denom != 0 else 0
    ssim_val = ssim(
        cv2.cvtColor(original.astype(np.uint8), cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(distorted.astype(np.uint8), cv2.COLOR_BGR2GRAY),
        data_range=255
    )
    return mse_val, nmse_val, psnr_val, if_val, ssim_val

images = ['1.jpg', '2.jpg', '3.jpg', '4.jpg']
results = {}

for idx, filename in enumerate(images):
    img = cv2.imread(filename)
    if img is None:
        raise FileNotFoundError(f"Image '{filename}' not found.")

    category = f"Image_{idx+1}" 
    results[category] = []

    if idx == 0:
        # JPEG degradation
        qualities = [95, 75, 50, 30, 15]
        for q in qualities:
            _, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            decimg = cv2.imdecode(encimg, 1)
            metrics = compute_metrics(img, decimg)
            results[category].append((f"JPEG_q{q}", *metrics))

    elif idx == 1:
        # Gaussian Blur
        ksizes = [3, 5, 9, 15, 21]
        for k in ksizes:
            blur = cv2.GaussianBlur(img, (k, k), 0)
            metrics = compute_metrics(img, blur)
            results[category].append((f"GaussBlur_k{k}", *metrics))

    elif idx == 2:
        # Gaussian Noise
        sigmas = [5, 15, 30, 50, 75]
        for s in sigmas:
            noise = np.random.normal(0, s, img.shape).astype(np.float32)
            noisy = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            metrics = compute_metrics(img, noisy)
            results[category].append((f"GaussNoise_s{s}", *metrics))

    elif idx == 3:
        # Median Blur
        sizes = [3, 5, 7, 9, 11]
        for k in sizes:
            blurred = cv2.medianBlur(img, k)
            metrics = compute_metrics(img, blurred)
            results[category].append((f"MedianBlur_k{k}", *metrics))

for name, data in results.items():
    df = pd.DataFrame(data, columns=['Degradation', 'MSE', 'NMSE', 'PSNR', 'IF', 'SSIM'])
    print(f"\nResults for {name}:")
    print(df.to_string(index=False))
