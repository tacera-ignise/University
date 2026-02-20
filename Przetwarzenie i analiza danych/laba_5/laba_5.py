import numpy as np
import pandas as pd
import scipy.sparse as sp

def freq(x, prob=True): 
    x_str=x.astype(str)
    unique_values, counts = np.unique(x_str, return_counts=True)
    if prob:
        total_count = np.sum(counts)
        probabilities = counts / total_count
        return unique_values, probabilities
    else:
        return unique_values, counts.astype(float)

def freq2(x, y, prob=True):
   
    combined_values = list(zip(x, y))
    unique_values, unique_counts = np.unique(combined_values, axis=0, return_counts=True)
    xi = unique_values[:, 0]
    yi = unique_values[:, 1]
    if prob:
        total_count = np.sum(unique_counts)
        ni = unique_counts / total_count
        return xi, yi, ni
    else:
        return xi, yi, unique_counts
def entropy(x):
    unique_values, probabilities = freq(x)
    h = -np.sum(probabilities * np.log2(probabilities))
    return h

def info_gain(x, y):
   
    h_x = entropy(x)
    h_y = entropy(y)
    h_xy = entropy(np.column_stack((x, y)))
    ig = h_x + h_y - h_xy
    return ig

dat = pd.read_csv("zoo.csv")
xi, ni = freq(dat['milk'])
print("Unikalne wartości xi:", xi)
print("Prawdopodobieństwa ni:", ni)

xi, yi, ni = freq2(dat['milk'], dat['animal'])
print("\nUnikalne wartości xi:", xi)
print("Unikalne wartości yi:", yi)
print("Rozkład czestosci ni:", ni)

h = entropy(dat['milk'])
print("Entropia dla feature_x:", h)

ig = info_gain(dat['milk'], dat['animal'])
print("\nPrzyrost informacji między feature_x a feature_y:", ig)
columns = dat.columns
gain_values = {}
for column in columns:
    ig = info_gain(dat[column], dat['type'])
    gain_values[column] = ig
print("\nPrzyrost informacji dla poszczególnych atrybutów:")
for column, ig in gain_values.items():
    print(f"{column}: {ig}")