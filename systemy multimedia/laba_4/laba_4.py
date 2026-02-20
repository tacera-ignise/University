
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

palette_1bit = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 1.0, 1.0],
])

palette_2bit = np.array([
    [0.0, 0.0, 0.0],
    [0.25, 0.25, 0.25],
    [0.75, 0.75, 0.75],
    [1.0, 1.0, 1.0],
])

palette_4bit = np.array([
    [0.0, 0.0, 0.0],
    [0.1, 0.1, 0.1],
    [0.2, 0.2, 0.2],
    [0.3, 0.3, 0.3],
    [0.4, 0.4, 0.4],
    [0.5, 0.5, 0.5],
    [0.6, 0.6, 0.6],
    [0.7, 0.7, 0.7],
    [0.8, 0.8, 0.8],
    [0.9, 0.9, 0.9],
    [1.0, 1.0, 1.0],
])

palette_8bit = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 0.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [1.0, 1.0, 1.0],
])

palette_16bit = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 1.0, 1.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [0.0, 0.5, 0.0],
    [0.5, 0.5, 0.5],
    [0.0, 1.0, 0.0],
    [0.5, 0.0, 0.0],
    [0.0, 0.0, 0.5],
    [0.5, 0.5, 0.0],
    [0.5, 0.0, 0.5],
    [1.0, 0.0, 0.0],
    [0.75, 0.75, 0.75],
    [0.0, 0.5, 0.5],
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 0.0],
])

def colorFit(pixel, Pallet):
    distances = np.linalg.norm(Pallet - pixel, axis=1)
    return Pallet[np.argmin(distances)]

def kwant_colorFit(img, Pallet):
    out_img = img.copy()
    for w in range(img.shape[0]):
        for k in range(img.shape[1]):
            out_img[w, k] = colorFit(img[w, k], Pallet)
    return out_img

def random_dithering(img):
    random_matrix = np.random.rand(img.shape[0], img.shape[1])
    gray_img = np.mean(img, axis=2)
    gray_img_rgb = np.repeat(gray_img[:, :, np.newaxis], 3, axis=2)
    binary_img = (gray_img >= random_matrix).astype(np.float32)
    return np.repeat(binary_img[:, :, np.newaxis], 3, axis=2)

def ordered_dithering(img, matrix_size=2, palette=palette_1bit):
    if matrix_size == 2:
        dither_matrix = np.array([[0, 2], [3, 1]]) / 4.0
    elif matrix_size == 4:
        dither_matrix = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]) / 16.0
    else:
        raise ValueError("Only 2x2 and 4x4 matrices are supported")
    
    gray_img = np.mean(img, axis=2)
    gray_img_rgb = np.repeat(gray_img[:, :, np.newaxis], 3, axis=2)
    height, width = gray_img.shape
    dithered_img = np.zeros_like(gray_img)
    
    for y in range(height):
        for x in range(width):
            threshold = dither_matrix[y % matrix_size, x % matrix_size]
            dithered_img[y, x] = 1.0 if gray_img[y, x] >= threshold else 0.0
    
    dithered_img_rgb = np.repeat(dithered_img[:, :, np.newaxis], 3, axis=2)
    quantized_dithered_img = kwant_colorFit(dithered_img_rgb, palette)
    
    return quantized_dithered_img

def floyd_steinberg_dithering(img, palette=palette_1bit):
    img = img.copy()
    height, width = img.shape[0], img.shape[1]
    gray_img = np.mean(img, axis=2)
    gray_img_rgb = np.repeat(gray_img[:, :, np.newaxis], 3, axis=2)
    
    for y in range(height):
        for x in range(width):
            oldpixel = gray_img[y, x]
            newpixel = 1.0 if oldpixel >= 0.5 else 0.0
            gray_img[y, x] = newpixel
            quant_error = oldpixel - newpixel
            
            if x + 1 < width:
                gray_img[y, x + 1] += quant_error * 7 / 16
            if y + 1 < height:
                if x - 1 >= 0:
                    gray_img[y + 1, x - 1] += quant_error * 3 / 16
                gray_img[y + 1, x] += quant_error * 5 / 16
                if x + 1 < width:
                    gray_img[y + 1, x + 1] += quant_error * 1 / 16
    
    dithered_img_rgb = np.repeat(gray_img[:, :, np.newaxis], 3, axis=2)
    quantized_dithered_img = kwant_colorFit(dithered_img_rgb, palette)
    
    return quantized_dithered_img

def process_images_and_save_to_pdf(image_paths, pdf_filename):
    with PdfPages(pdf_filename) as pdf:
        for image_path in image_paths:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_normalized = image / 255.0

            quantized_image_1bit = kwant_colorFit(image_normalized, palette_1bit)
            quantized_image_2bit = kwant_colorFit(image_normalized, palette_2bit)
            quantized_image_4bit = kwant_colorFit(image_normalized, palette_4bit)
            quantized_image_8bit = kwant_colorFit(image_normalized, palette_8bit)
            quantized_image_16bit = kwant_colorFit(image_normalized, palette_16bit)
            random_dithered = random_dithering(image_normalized)
            ordered_dithered = ordered_dithering(image_normalized)
            floyd_steinberg_dithered = floyd_steinberg_dithering(image_normalized)

            quantized_image_1bit = (quantized_image_1bit * 255).astype(np.uint8)
            quantized_image_2bit = (quantized_image_2bit * 255).astype(np.uint8)
            quantized_image_4bit = (quantized_image_4bit * 255).astype(np.uint8)
            quantized_image_8bit = (quantized_image_8bit * 255).astype(np.uint8)
            quantized_image_16bit = (quantized_image_16bit * 255).astype(np.uint8)
            random_dithered = (random_dithered * 255).astype(np.uint8)
            ordered_dithered = (ordered_dithered * 255).astype(np.uint8)
            floyd_steinberg_dithered = (floyd_steinberg_dithered * 255).astype(np.uint8)
            
            plt.figure(figsize=(15, 11))

            plt.subplot(3, 4, 1)
            plt.imshow(image)
            plt.title("Original Image")
            plt.axis("off")

            plt.subplot(3, 4, 2)
            plt.imshow(quantized_image_1bit)
            plt.title("1bit Palette")
            plt.axis("off")

            plt.subplot(3, 4, 3)
            plt.imshow(quantized_image_2bit)
            plt.title("2bit Palette")
            plt.axis("off")

            plt.subplot(3, 4, 4)
            plt.imshow(quantized_image_4bit)
            plt.title("4bit Palette")
            plt.axis("off")

            plt.subplot(3, 4, 5)
            plt.imshow(quantized_image_8bit)
            plt.title("Palette8")
            plt.axis("off")

            plt.subplot(3, 4, 6)
            plt.imshow(quantized_image_16bit)
            plt.title("Palette16")
            plt.axis("off")

            plt.subplot(3, 4, 7)
            plt.imshow(random_dithered)
            plt.title("Random Dithering")
            plt.axis("off")

            plt.subplot(3, 4, 8)
            plt.imshow(ordered_dithered)
            plt.title("Ordered Dithering")
            plt.axis("off")

            plt.subplot(3, 4, 9)
            plt.imshow(floyd_steinberg_dithered)
            plt.title("Floyd-Steinberg Dithering")
            plt.axis("off")

            pdf.savefig()
            plt.close()
image_paths = ["GS_0001.tif","GS_0002.png", "GS_0003.png", "SMALL_0006.jpg","SMALL_0007.jpg","SMALL_0008.jpg", "SMALL_0009.jpg"]
pdf_filename = "quantized_images.pdf"

process_images_and_save_to_pdf(image_paths, pdf_filename)
