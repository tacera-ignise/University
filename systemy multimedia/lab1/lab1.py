import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
import scipy.fftpack
import PyQt5

print(PyQt5.__file__)

 
# data - tablica zawierająca wartości próbek sygnału, każda z jej kolumn
#  zawiera osobny kanał wartości (domyślnie 2 to Lewy i Prawy).
# fs - częstotliwość próbkowania, czyli jak często sygnał rzeczywisty był próbkowany,
#  czyli jak wiele próbek znajduje się w jednej sekundzie trwania dźwięku.
data, fs = sf.read('sound1.wav', dtype='float32') 
print(data.dtype)
print(data.shape)
print(fs)

# Odpowiada za rozpoczęcie odtwarzania
sd.play(data,fs)
 # zapewnia, że program poczeka z wykonaniem kolejnych instrukcji
 # do momentu, aż dźwięk skończy się odtwarzać.

status = sd.wait()

 
#Do zapisu danych do pliku wave wykorzystujemy funkcję
#sf.write('sound_new.wav',new_data, fs)
sf.write('sound_L.wav',data[:,0], fs)
sf.write('sound_R.wav',data[:,1], fs)
sf.write('sound_mix.wav',data, fs)
x=np.arange(0,data.shape[0])/fs

plt.figure(figsize=(15,5))

plt.subplot(3,1,1)
plt.plot(x,data[:,0])
plt.xlabel("Czas")
plt.ylabel("Amplituda")
plt.title("Kanal1")

plt.subplot(3,1,3)
plt.plot(x,data[:,1])
plt.xlabel("Czas")
plt.ylabel("Amplituda")
plt.title("Kanal2")

plt.show()
#plt.figure(figsize=(10,5)) dla sowmestnych grafikow
 
