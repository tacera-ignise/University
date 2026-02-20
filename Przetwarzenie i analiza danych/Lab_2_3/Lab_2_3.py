import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
#pd.Series - lista
#dates = pd.date_range("20200301",periods=5,name='data')
#print(dates)
#df = pd.DataFrame(np.random.normal(5, 3), index=dates, columns=list("ABC"))
#print(df)
#df=pd.DataFrame(np.random.rand(),index=list(range(20)),columns=list("ABC"))
#df.index.name="index"
#print(df.head(3))
#print(df.tail(3))
#print(df.index.name)
#print(df.columns)
#print(df.to_numpy())
#print(df.loc[np.random.randint(0,19,5)])
#print(df.values[:,0])
#df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": ["a", "b", "c","d", "z"]},index=np.arange(1,6))
#print(df.mean())
#left = pd.DataFrame({"key": ["foo", "foo"], "lval": [1, 2]})
#right = pd.DataFrame({"key": ["foo", "foo"], "rval": [4, 5]})
#print(pd.merge(left,right))
#df = pd.DataFrame(
# {
#       "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
#       "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
#       "C": np.random.randn(8),
#       "D": np.random.randn(8),
#print(df)
#print(df.groupby("B")[["C", "D"]].sum())

def Zadania():
    df = pd.DataFrame({"X": [1, 2, 3, 4, 5], "Y": ["a","b","a","b","b"]})
    print("\n\nZadanie - 1\n",df.groupby("Y").mean())
    print('\n\nZadanie - 2\n',df.value_counts())

    autos= pd.read_csv('autos.csv')
    #print("\n\n----------------------------",autos_pd)21,205
    print("\n\nZadanie - 3\n",autos)
    
    avg_mpg_pd = autos.groupby('make')['highway-mpg'].mean()
    print("\n\nZadanie - 4\n",avg_mpg_pd)

    fuel_type_counts = autos.groupby('make')['fuel-type'].value_counts()

    print("\n\nZadanie - 5 \n Liczności dla każdego typu paliwa, zgrupowane według producenta:",fuel_type_counts)


    x = autos['length']
    y = autos['city-mpg']

    coefficients_linear = np.polyfit(x, y, 1)

    coefficients_quadratic = np.polyfit(x, y, 2)

    predicted_linear = np.polyval(coefficients_linear, x)
    
    predicted_quadratic = np.polyval(coefficients_quadratic, x)
    print("\n\nZadanie - 6\n",predicted_linear,predicted_quadratic)

    cor_cof, p_value = pearsonr(x, y)
    print("\n\nZadanie - 7\n",cor_cof, p_value)

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, label='Próbki')
    plt.plot(x, predicted_linear, color='red', label='Wielomian liniowy')
    plt.plot(x, predicted_quadratic, color='yellow', label='Wielomian 2 stopnia')
    plt.xlabel('Length')
    plt.ylabel('City-mpg')
    plt.title('Zadanie - 8')
    plt.legend()
    plt.grid(True)
    plt.show()

    kde = gaussian_kde(x)
    x_values = np.linspace(x.min(), x.max(), 800)
    y_values = kde.evaluate(x_values)

    plt.figure(figsize=(8, 5))
    plt.scatter(x, np.zeros_like(x), alpha=0.5, label='Próbki')
    plt.plot(x_values, y_values, color='black', label='Funkcja gęstości')
    plt.xlabel('Length')
    plt.ylabel('Gęstość')
    plt.title('Zadanie - 9')
    plt.legend()
    plt.grid(True)
    plt.show()


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.scatter(x, np.zeros_like(x), alpha=0.5, label='Próbki')
    ax1.plot(x_values, y_values, color='red', label='Funkcja gęstości')
    ax1.set_xlabel('Length')
    ax1.set_ylabel('Gęstość')
    ax1.set_title('Zadanie - 10 "length"')
    ax1.legend()
    ax1.grid(True)

    x_width = autos['width']
    kde_width = gaussian_kde(x_width)
    x_values_width = np.linspace(x_width.min(), x_width.max(), 800)
    y_values_width = kde_width.evaluate(x_values_width)

    ax2.scatter(x_width, np.zeros_like(x_width), alpha=0.5, label='Próbki')
    ax2.plot(x_values_width, y_values_width, color='blue', label='Funkcja gęstości')
    ax2.set_xlabel('Width')
    ax2.set_ylabel('Gęstość')
    ax2.set_title('Zadanie - 10 "width"')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


    
    x = autos['width']
    y = autos['length']
    kde = gaussian_kde([x, y])

    x_grid, y_grid = np.meshgrid(np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100))
    z = kde([x_grid.flatten(), y_grid.flatten()]).reshape(x_grid.shape)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_grid, y_grid, z, cmap='viridis')
    ax.set_xlabel('Width')
    ax.set_ylabel('Length')
    ax.set_zlabel('Gęstość')
    ax.set_title('Dwuwymiarowy estymator funkcji gęstości')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.contour(x_grid, y_grid, z, cmap='viridis')
    plt.scatter(x, y, alpha=0.5)
    plt.xlabel('Width')
    plt.ylabel('Length')
    plt.title('Dwuwymiarowy estymator funkcji gęstości z próbkami')
    plt.colorbar(label='Gęstość')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.contour(x_grid, y_grid, z, cmap='viridis')
    plt.scatter(x, y, alpha=0.5)
    plt.xlabel('Width')
    plt.ylabel('Length')
    plt.title('Dwuwymiarowy estymator funkcji gęstości z próbkami')
    plt.colorbar(label='Gęstość')
    plt.savefig('dwuwymiarowy_estymator_gestosci.png')
    plt.savefig('dwuwymiarowy_estymator_gestosci.pdf')
    plt.show()

Zadania()



