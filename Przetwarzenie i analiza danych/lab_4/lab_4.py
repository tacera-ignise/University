import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def Dyskret(f:int ,F: float):
    dt=float(1/F)
    t=np.arange(0,1,dt)
    t=list(t)
    s=[]
    for i in range(0,len(t),1):
        s.append(np.sin(2*np.pi*f*t[i]))
    return t,s  
t,s=Dyskret(10,1000)
plt.figure(figsize=(8,5))
plt.scatter(t,s,label="Próbki")
plt.plot(t,s,color="black",label="{f}Hz")
plt.xlabel('Wartość')
plt.ylabel('Funkcja')
plt.grid(True)
plt.show()
# 4. Istnieje twierdzenie które mówi że częstotliwość próbkowania (fs) 
# powinna być co najmniej w dwa razy większa częstotliwości sygnalu.
# 5. Aliasing


# Kwantyzacja
image = Image.open("Aliasing.png")
image_array = np.array(image)
dimensions = image_array.shape
print("Wymiary macierzy (obrazka):", dimensions)
num_pixel_values = image_array.shape[-1]
print("Liczba wartości na piksel:", num_pixel_values)

# Metoda 1: Wyznaczenie jasności piksela
gray_brightness = ((np.max(image_array, axis=2) + np.min(image_array, axis=2)) // 2).astype(np.uint8)
# Metoda 2: Uśrednienie wartości piksela
gray_average = np.sum(image_array, axis=2) // 3
# Metoda 3: Wyznaczenie luminancji piksela
coefficients = [0.21, 0.72, 0.07]
gray_luminance = np.dot(image_array, coefficients).astype(np.uint8)

# Wygenerowanie histogramu dla każdego obrazu
hist_brightness, bins_brightness = np.histogram(gray_brightness.flatten(), bins=256, range=(0, 256))
hist_average, bins_average = np.histogram(gray_average.flatten(), bins=256, range=(0, 256))
hist_luminance, bins_luminance = np.histogram(gray_luminance.flatten(), bins=256, range=(0, 256))

# Redukcja liczby kolorów na histogramie do 16
hist_reduced, bins_reduced = np.histogram(gray_brightness.flatten(), bins=16, range=(0, 256))
print("Zakresy nowych kolorów:")
for i in range(len(bins_reduced) - 1):
    print(f"{bins_reduced[i]} - {bins_reduced[i+1]}")
new_gray_image = np.digitize(gray_brightness, bins_reduced[:-1], right=True)

fig, axes = plt.subplots(3, 4, figsize=(15, 10))
axes[0, 0].imshow(image)
axes[0, 0].set_title("Obraz oryginalny")
axes[0, 0].axis('off')
axes[0, 1].plot(hist_brightness)
axes[0, 1].set_title("Histogram (Jasność)")
axes[0, 2].imshow(gray_brightness, cmap='gray')
axes[0, 2].set_title("Obraz szary (Jasność)")
axes[0, 2].axis('off')
axes[0, 3].imshow(new_gray_image, cmap='gray')
axes[0, 3].set_title("Obraz zredukowany (Jasność)")
axes[0, 3].axis('off')

axes[1, 0].imshow(image)
axes[1, 0].axis('off')
axes[1, 1].plot(hist_average)
axes[1, 1].set_title("Histogram (Uśrednienie)")
axes[1, 2].imshow(gray_average, cmap='gray')
axes[1, 2].set_title("Obraz szary (Uśrednienie)")
axes[1, 2].axis('off')

axes[2, 0].imshow(image)
axes[2, 0].axis('off')
axes[2, 1].plot(hist_luminance)
axes[2, 1].set_title("Histogram (Luminancja)")
axes[2, 2].imshow(gray_luminance, cmap='gray')
axes[2, 2].set_title("Obraz szary (Luminancja)")
axes[2, 2].axis('off')

axes[2, 3].imshow(new_gray_image, cmap='gray')
axes[2, 3].set_title("Obraz zredukowany (Jasność)")
axes[2, 3].axis('off')
plt.tight_layout()
plt.show()


# Binaryzacja
image = Image.open("obraz.jpg")
gray_image = image.convert("L")
gray_array = np.array(gray_image)
histogram = np.histogram(gray_array.flatten(), bins=256, range=(0,256))[0]
plt.plot(histogram)
plt.title("Histogram obrazu szarego")
plt.xlabel("Wartość piksela")
plt.ylabel("Liczba pikseli")
plt.show()
def Prog_bin(hist):
    local_min = np.inf
    threshold = 0
    for i in range(1, len(hist) - 1):
        if hist[i-1] > hist[i] < hist[i+1]:
            if hist[i] < local_min:
                local_min = hist[i]
                threshold = i
    return threshold

threshold_value = Prog_bin(histogram)
print("Punkt progowania (próg binaryzacji):", threshold_value)
binary_array = np.where(gray_array < threshold_value, 0, 255).astype(np.uint8)

binary_image = Image.fromarray(binary_array)
plt.imshow(binary_image, cmap='gray')
plt.title("Obraz zbinaryzowany")
plt.axis('off')
plt.show()