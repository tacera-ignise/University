
from matplotlib import pyplot as plt
import numpy as np
from class1 import SimplePerceptron
import time
def fake_data(m=100):
    x=np.random.uniform(0,2*np.pi,m)
    z=np.random.uniform(-1,1,m)
    X=np.column_stack((np.ones(m),x,z))
    y=np.where(np.abs(np.sin(X[:,1]))>np.abs(X[:,2]),-1,1)
    return X,y,z

def Gauss(I,m,X,center,sigma):
    Z=np.zeros((I,m))
    for i in range(I):
        for j in range(m):
            dist=((X[i,1]-center[j,0])**2)+((X[i,2]-center[j,1])**2)
            Z[i,j]=np.exp(-dist/(2*sigma**2))
    return np.column_stack((np.ones(I),Z))

if __name__=="__main__":
    np.random.seed(0)
    I=1000
    X,y,z=fake_data(m=I)
    
    X[:,1]=2*(X[:,1]/(2*np.pi))-1
    
    m=100
    sigma=0.2
    center=np.random.uniform(-1,1,(m,2))
    Z=Gauss(I,m,X,center,sigma)
    
    clf = SimplePerceptron(learning_rate=0.1, k_max=10000)
    t1 = time.time()
    clf.fit(Z, y)
    t2 = time.time()
    
    print(f"WEIGHTS: {clf.w_}")
    print(f"STEPS: {clf.k_}")
    print(f"Time: {t2 - t1:.4f}s")
    print(f"ACC: {clf.score(Z, y)}")
    
    plt.scatter(X[:, 1], X[:, 2], c=y, cmap="bwr", marker=".")
    xx, yy = np.meshgrid(np.linspace(-1, 1, 200), np.linspace(-1, 1, 200))
    grid = np.column_stack((np.ones(xx.ravel().shape), xx.ravel(), yy.ravel()))
    Z_grid = Gauss(grid.shape[0], m, grid, center, sigma)
    Z_pred = clf.decision_function(Z_grid).reshape(xx.shape)
    plt.contour(xx, yy, Z_pred, levels=[0], colors="black")
    
    plt.title("Decision Boundary in Original Space")
    plt.xlabel("x1 (normalized)")
    plt.ylabel("x2 (normalized)")
    plt.show()
