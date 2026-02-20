import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
import scipy.fftpack
import librosa

try:
    s, fs = sf.read('voice.wav', dtype='float32')
    if len(s.shape) > 1:
        # Przekształć sygnał stereo na mono, jeśli jest stereo
        s = np.mean(s, axis=1)
except RuntimeError as e:
    print(f"Błąd przy otwieraniu pliku: {e}")
    exit(1)

duration = len(s) / fs
bit_depth = s.dtype.itemsize * 8

print(f"Czas trwania: {duration} s")
print(f"Częstotliwość próbkowania: {fs} Hz")
print(f"Głębia bitowa: {bit_depth} bit")
print(f"Liczba kanałów: 1 (mono)")


time = np.linspace(0, duration, len(s))
plt.figure()
plt.plot(time * 1000, s)
plt.xlabel("Czas [ms]")
plt.ylabel("Amplituda")
plt.title("Sygnał")
plt.show()


if np.max(np.abs(s)) > 1:
    s = s / np.max(np.abs(s))


print(f"Maksymalna amplituda: {np.max(s)}")
print(f"Minimalna amplituda: {np.min(s)}")
print(f"Amplituda szumu na początku: {np.mean(s[:fs//10])}")
print(f"Amplituda szumu na końcu: {np.mean(s[-fs//10:])}")

def calculate_energy(frame):
    return np.sum(frame**2)

def calculate_zero_crossings(frame):
    zero_crossings = np.where(np.diff(np.sign(frame)))[0]
    return len(zero_crossings)

def analyze_frames(s, fs, frame_length_ms, overlap=0):
    frame_length = int(frame_length_ms * fs / 1000)  # długość okna w próbkach
    step = int(frame_length * (1 - overlap))  # krok z uwzględnieniem nakładania

    num_frames = (len(s) - frame_length) // step + 1
    energy = []
    zero_crossings = []

    for i in range(num_frames):
        start = i * step
        frame = s[start:start + frame_length]
        energy.append(calculate_energy(frame))
        zero_crossings.append(calculate_zero_crossings(frame))

    return np.array(energy), np.array(zero_crossings)


energy_10ms, zero_crossings_10ms = analyze_frames(s, fs, 10)


energy_10ms = energy_10ms / np.max(energy_10ms)
zero_crossings_10ms = zero_crossings_10ms / np.max(zero_crossings_10ms)

# Wizualizacja funkcji
time_frames = np.arange(len(energy_10ms)) * 10  # 10 ms na okno

plt.figure()
plt.plot(time_frames, energy_10ms, 'r', label='Energia')
plt.plot(time_frames, zero_crossings_10ms, 'b', label='Przejścia przez zero')
plt.xlabel("Czas [ms]")
plt.ylabel("Znormalizowana wartość")
plt.legend()
plt.title("Energia i przejścia przez zero dla okien 10 ms")
plt.show()

# Analiza wpływu długości okna
frame_lengths_ms = [5, 20, 50]
for fl in frame_lengths_ms:
    energy, zero_crossings = analyze_frames(s, fs, fl)
    energy = energy / np.max(energy)
    zero_crossings = zero_crossings / np.max(zero_crossings)
    time_frames = np.arange(len(energy)) * fl

    plt.figure()
    plt.plot(time_frames, energy, 'r', label='Energia')
    plt.plot(time_frames, zero_crossings, 'b', label='Przejścia przez zero')
    plt.xlabel("Czas [ms]")
    plt.ylabel("Znormalizowana wartość")
    plt.legend()
    plt.title(f"Energia i przejścia przez zero dla okien {fl} ms")
    plt.show()

# Analiza z nakładaniem okien 50%
energy_overlap, zero_crossings_overlap = analyze_frames(s, fs, 10, overlap=0.5)

energy_overlap = energy_overlap / np.max(energy_overlap)
zero_crossings_overlap = zero_crossings_overlap / np.max(zero_crossings_overlap)


time_frames_overlap = np.arange(len(energy_overlap)) * 5  # 5 ms na okno (z uwzględnieniem 50% nakładania)

plt.figure()
plt.plot(time_frames_overlap, energy_overlap, 'r', label='Energia')
plt.plot(time_frames_overlap, zero_crossings_overlap, 'b', label='Przejścia przez zero')
plt.xlabel("Czas [ms]")
plt.ylabel("Znormalizowana wartość")
plt.legend()
plt.title("Energia i przejścia przez zero dla okien 10 ms z nakładaniem 50%")
plt.show()

if len(s) >= 2048:
    vowel_start = int(1.0 * fs) 
    vowel_end = vowel_start + 2048
    vowel_segment = s[vowel_start:vowel_end]

    
    window = np.hamming(len(vowel_segment))
    windowed_segment = vowel_segment * window

  
    yf = scipy.fftpack.fft(windowed_segment)
    amplitude_spectrum = np.log(np.abs(yf))

 
    frequencies = np.fft.fftfreq(len(amplitude_spectrum), 1/fs)
    plt.figure()
    plt.plot(frequencies[:len(amplitude_spectrum)//2], amplitude_spectrum[:len(amplitude_spectrum)//2])
    plt.xlabel("Częstotliwość [Hz]")
    plt.ylabel("Amplituda [dB]")
    plt.title("Logarytmiczny spektrogram amplitud")
    plt.show()

# Określenie częstotliwości podstawowej F0
    peaks = np.where((amplitude_spectrum[1:-1] > amplitude_spectrum[:-2]) & 
                    (amplitude_spectrum[1:-1] > amplitude_spectrum[2:]))[0] + 1
    if peaks.size > 0:
        F0 = np.abs(frequencies[peaks[0]])
        print(f"Częstotliwość podstawowa (F0): {F0} Hz")

# Rozpoznawanie samogłosek
# W tym przykładzie zakłada się, że wyodrębniono fragment samogłoski, jak wskazano wcześniej
    vowel_lpc = librosa.lpc(windowed_segment, order=20)

# Wizualizacja LPC spektrogramu
    a = np.zeros(len(vowel_segment))
    a[:len(vowel_lpc)] = vowel_lpc
    lpc_spectrum = np.log(np.abs(np.fft.fft(a)))
    lpc_spectrum *= -1  # Odbicie spektrogramu

    plt.figure()
    plt.plot(frequencies[:len(amplitude_spectrum)//2], amplitude_spectrum[:len(amplitude_spectrum)//2], label='Spektrogram amplitud')
    plt.plot(frequencies[:len(lpc_spectrum)//2], lpc_spectrum[:len(lpc_spectrum)//2], 'r', label='LPC spektrogram')
    plt.xlabel("Częstotliwość [Hz]")
    plt.ylabel("Amplituda [dB]")
    plt.legend()
    plt.title("Analiza LPC")
    plt.show()
else:
        print("Sygnał jest zbyt krótki do analizy częstotliwościowej z oknem o długości 2048 próbek")
