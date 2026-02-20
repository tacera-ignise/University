import numpy as np
a=np.array([1,2,3,4,5,6,7])
b=np.array([[1,2,3,4,5],[6,7,8,9,10]])
b=np.transpose(b)
c=np.arange(0,100,1)
d=np.linspace(0,2,10)
f=np.arange(0,100,5)
s=np.random.normal(0,2,20).round(2)
v=np.random.randint(1,1000,100)
a=np.zeros([3,2])
print(a)
a=np.ones([3,2])
a=np.random.randint(0,100,(2,2,3),dtype=np.int32)
print(a)
a_array=np.random.rand(5,5)*10
#print(a_array)
b_array=a_array.astype(np.int32)
#print(b_array)
a_array=np.round(a_array,0)
#print(a_array)

def zad_():
    b=np.array([[1,2,3,4,5],[6,7,8,9,10]],dtype=np.int32)
    print(np.ndim(b))
    print(np.size(b))
    print(b[:,0])

    c=np.random.randint(0,100,(20,7))
    print(c)
    print(c[:,0:4])
#zad_()

def zad__():
    a=np.random.randint(0,11,(3,3))
    b=np.random.randint(0,11,(3,3))
    dod=np.add(a,b)
    od=np.subtract(a,b)
    mnoz=np.multiply(a,b)
    dziel=a/b
    stiep=np.power(a,b)
    print(dod,od,mnoz,dziel,stiep)

    print("Wartości macierzy a większe lub równe 4:")
    print(a >= 4)
    print("\nWartości macierzy a większe lub równe 1 i mniejsze lub równe 4:")
    print((a >= 1) & (a <= 4))
    sum_diagonal_b = np.trace(b)
    print("\nSuma głównej przekątnej macierzy b:", sum_diagonal_b)
    print(b)
    sum_b=np.sum(b)
    print("\nSuma macierzy b: ",sum_b)
    min_b=np.min(b)
    print("\nMin macierzy b: ",min_b)
    max_b=np.max(b)
    print("\nMax macierzy b: ",max_b)
    std_b=np.std(b)
    print("\nOdchylenie standardowe macierzy b: ",std_b)

    srednia_wierszy = np.mean(b, axis=1)
    srednia_kolumn = np.mean(b, axis=0)
    print("Średnia dla wierszy w macierzy b:", srednia_wierszy)
    print("Średnia dla kolumn macierzy b:", srednia_kolumn)
#zad__()

def zad___():
    a=np.arange(0,50,1)
    print(np.reshape(a,(10,5)))
    print(np.resize(a,(10,5)))
    print(np.ravel(a))
    # funkcja ravel tworzy tablicu jednomiarowu z wielomianowej tablicy
    c=np.arange(1,6,1)
    print("C: ",c)
    b=np.arange(1,5,1)
    print("B: ",b)
    c=c[:,np.newaxis]
    print("Wynik: ",c+b)
    # newaxis służy do dodania nowego wymiaru do tablicy
#zad___()

def zad____():
    a=np.random.randint(0,100,(5,5))
    wiersz_sort=np.sort(a)
    kol_sort=np.sort(a,0)
    print(a)
    print("\n Wiersz Sort",wiersz_sort)
    print('\n Kol Sort',kol_sort)

    b=np.array([(1,'MZ','mazowieckie'),(2,'ZP','zachodniopomorskie'),(3,'ML','malopolskie')])
    print("\n\n",b)
    sort_index=np.argsort(b[:,1])
    tab_sort=b[sort_index]
    print("\n",tab_sort)
    print("\n",b[[1]])
#zad____()

def Zad_Podsumowujace():
    a=np.random.randint(0,100,(10,5))
    print("Zadanie 1: ",a)
    print("Suma glównej przekątnej: ",np.trace(a))
    print("Diag: ",np.diag(a))

    a=np.random.normal(size=(5,5)).round(2)
    b=np.random.normal(size=(5,5)).round(2)
    print("\n\nZadanie - 2 wynik: ",a*b)

    a=np.random.randint(1,100,100)
    a=np.reshape(a,(20,5))
    b=np.random.randint(1,100,100)
    b=np.reshape(b,(20,5))
    wynik=a+b
    print("\n\nZadanie - 3: ",wynik)
    
    a= np.random.randint(1, 10, size=(4, 5))
    b= np.random.randint(1, 10, size=(5, 4))
    print("\n\nTab A:",a)
    print("\nTab B:",b)
    a=np.transpose(a)
    print("\n\nTranspose A:",a)
    wynik=a+b
    print("\n\nZadanie - 4: ",wynik)
    
    print("\n\nZadanie - 5: ", a[:,2:4:1]*b[:,2:4:1])

    a=np.random.normal(size=(4,5)).round(2)
    b=np.random.uniform(size=(4,5)).round(2)
    print("\n\nZadanie - 6: \nTab:A",a)
    print("\nTab:B",b)
    a_mid=np.mean(a)
    a_std=np.std(a)
    a_var=np.var(a)
    a_sum=np.sum(a)
    a_min=np.min(a)
    a_max=np.max(a)
    print("Tab A:\nWartość srednie: ",a_mid,"\nSTD: ",
          a_std,"\nWariacja: ",a_var,"\nSuma: ",a_sum,
          "\nMaksimum: ",a_max,"\nMin: ",a_min)
    b_mid=np.mean(b)
    b_std=np.std(b)
    b_var=np.var(b)
    b_sum=np.sum(b)
    b_min=np.min(b)
    b_max=np.max(b)
    print("\n\nTab B:\nWartość srednie: ",b_mid,"\nSTD: ",
          b_std,"\nWariacja: ",b_var,"\nSuma: ",b_sum,
          "\nMaksimum: ",b_max,"\nMin: ",b_min)

    a=np.random.randint(0,10,(5,5))
    b=np.random.randint(0,10,(5,5))
    print("\n\nZadanie - 7 \nTab A:\n",a,"\nTab B:\n",b)
    proste=a*b
    Dot=np.dot(a,b)
    print("\nWynik 1:",proste,"\nWynik 2:",Dot)
    print('''Operator * mnoży odpowiadające sobie elementy dwóch macierzy.Funkcja np.dot wykonuje operację mnożenia macierzowego.
Jeśli chcemy wykonać operację mnożenia macierzowego (zgodnie z definicją matematyczną) używamy funkcji np.dot.''')

    a=np.random.randint(0,100,(6,6))
    print("\n\nZadanie - 8: \nTab:\n",a)
    kroki = a.strides
    print("\nStrides: ",kroki)
    # Użycie funkcji as_strided do wyboru danych
    wyn = np.lib.stride_tricks.as_strided(a, shape=(3, 5), strides=(kroki[0], kroki[1]))
    print("\nWynik: \n",wyn)

    a=np.random.randint(0,100,(3,3))
    print("\nZadanie - 9:\nTab A\n",a)
    b=np.random.randint(0,100,(3,3))
    print("Tab B:\n",b)
    print("VSTACK:\n",np.vstack((a,b)))
    print("HSTACK:\n",np.hstack((a,b)))
    print(''' np.vstack łączy tablice wzdłuż pionowej osi, czyli dodaje kolejne wiersze.
 np.hstack łączy tablice wzdłuż poziomej osi, czyli dodaje kolejne kolumny.''')

    macierz = np.array([[0,1,2,3,4,5],
                    [6,7,8,9,10,11],
                    [12,13,14,15,16,17],
                    [18,19,20,21,22,23]])

    bloki_danych = np.lib.stride_tricks.as_strided(macierz, shape=(2,2,2,3),strides=(48,12,24,4))
    print("\n\nZadanie -10\nBlok: ",bloki_danych)
    maksymalne_wartosci = np.max(bloki_danych,axis=(2,3))

    print("Maksymalne wartości dla każdego bloku danych:")
    print(maksymalne_wartosci)

Zad_Podsumowujace()