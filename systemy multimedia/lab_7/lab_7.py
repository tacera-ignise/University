from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.pyplot as plt

# Функции для A-Law, Mu-Law, DPCM
def A_law_compress(signal, A=87.6):
    signal = np.clip(signal, -1.0, 1.0)
    abs_signal = np.abs(signal)
    encoded = np.where(
        abs_signal < 1 / A,
        A * abs_signal / (1 + np.log(A)),
        (1 + np.log(A * abs_signal)) / (1 + np.log(A))
    )
    return np.sign(signal) * encoded

def A_law_decompress(encoded, A=87.6):
    abs_encoded = np.abs(encoded)
    signal = np.where(
        abs_encoded < 1 / (1 + np.log(A)),
        abs_encoded * (1 + np.log(A)) / A,
        np.exp(abs_encoded * (1 + np.log(A)) - 1) / A
    )
    return np.sign(encoded) * signal

def mu_law_compress(signal, mu=255):
    signal = np.clip(signal, -1.0, 1.0)
    return np.sign(signal) * np.log1p(mu * np.abs(signal)) / np.log1p(mu)

def mu_law_decompress(encoded, mu=255):
    return np.sign(encoded) * (1 / mu) * ((1 + mu) ** np.abs(encoded) - 1)

def kwant(x, bit):
    return np.round(x * (2 ** (bit - 1) - 1)) / (2 ** (bit - 1) - 1)

def DPCM_compress(x, bit):
    y = np.zeros(x.shape)
    e = 0
    for i in range(x.shape[0]):
        y[i] = kwant(x[i] - e, bit)
        e += y[i]
    return y

def DPCM_decompress(y):
    x_rec = np.zeros(y.shape)
    e = 0
    for i in range(y.shape[0]):
        x_rec[i] = y[i] + e
        e = x_rec[i]
    return x_rec

def geom_mean_pred(X):
    if len(X) == 0:
        return 0
    X = np.where(X == 0, 1e-10, X) 
    gm = np.prod(np.abs(X))**(1/len(X)) 
    return np.sign(np.mean(X)) * gm 

def DPCM_compress_pred(x, bit, predictor, n):
    y = np.zeros(x.shape)
    xp = np.zeros(x.shape)
    for i in range(x.shape[0]):
        idx = np.arange(i - n, i)
        idx = idx[idx >= 0]
        pred = predictor(xp[idx])
        y[i] = kwant(x[i] - pred, bit)
        xp[i] = y[i] + pred
    return y

def DPCM_decompress_pred(y, predictor, n):
    x_rec = np.zeros(y.shape)
    for i in range(len(y)):
        idx = np.arange(i - n, i)
        idx = idx[idx >= 0]
        pred = predictor(x_rec[idx])
        x_rec[i] = y[i] + pred
    return x_rec

def generate_plots_for_pdf(x1, x2, bit_depths=[8, 7, 6, 5, 4, 3, 2]):
    with PdfPages('Wynik.pdf') as pdf:
        for x in [x1, x2]:

            for bit in bit_depths:
                plt.figure(figsize=(10, 10))
                y = 0.9 * np.sin(np.pi * x * 4)
                y_a_quant = kwant(A_law_compress(y), bit)
                y_a_dec = A_law_decompress(y_a_quant)

                y_mu_quant = kwant(mu_law_compress(y), bit)
                y_mu_dec = mu_law_decompress(y_mu_quant)

                y_dpcm = DPCM_compress(y, bit)
                y_dpcm_dec = DPCM_decompress(y_dpcm)

                y_dpcm_geom = DPCM_compress_pred(y, bit, geom_mean_pred, 3)
                y_dpcm_geom_dec = DPCM_decompress_pred(y_dpcm_geom, geom_mean_pred, 3)

                plt.subplot(4, 1, 1)
                plt.plot(x, y_a_dec)
                plt.title(f"A-Law Compression (Bit Depth: {bit})")
                plt.ylim(-1, 1)

                plt.subplot(4, 1, 2)
                plt.plot(x, y_mu_dec)
                plt.title(f"Mu-Law Compression (Bit Depth: {bit})")
                plt.ylim(-1, 1)

                plt.subplot(4, 1, 3)
                plt.plot(x, y_dpcm_dec)
                plt.title(f"DPCM without Prediction (Bit Depth: {bit})")
                plt.ylim(-1, 1)

                plt.subplot(4, 1, 4)
                plt.plot(x, y_dpcm_geom_dec)
                plt.title(f"DPCM with Prediction (Bit Depth: {bit})")
                plt.ylim(-1, 1)

                plt.tight_layout()
                pdf.savefig()  
                plt.close()

x1 = np.linspace(-1, 1, 1000)
x2 = np.linspace(-0.5, -0.25, 1000)

generate_plots_for_pdf(x1, x2)