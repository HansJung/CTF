import numpy as np
import pandas as pd

def fY(U0,U1,X):
    return (U0|U1)^X

np.random.seed(1)
N = 10000
U0 = np.random.binomial(1,0.2,N)
U1 = np.random.binomial(1,0.5,N)
X = np.random.binomial(1,0.4,N) * (1-(U0^U1))
Y = fY(U0,U1,X)

OBS = pd.DataFrame({'X':X,'Y':Y})

X0 = np.array([0]*N)
X1 = np.array([1]*N)
Y0 = fY(U0,U1,X0)
Y1 = fY(U0,U1,X1)

print('Y0:',np.mean(Y0))
print('Y1:',np.mean(Y1))
print('Y|x0:',np.mean(OBS[OBS['X']==0]['Y']))
print('Y|x1:',np.mean(OBS[OBS['X']==1]['Y']))

# Bound
l0 = np.mean(OBS[OBS['X']==0]['Y'])*(1-np.mean(X))
l1 = np.mean(OBS[OBS['X']==1]['Y'])*(np.mean(X))
h0 = l0 + np.mean(X)
h1 = l1 + (1-np.mean(X))

print(l0,np.mean(Y0),h0)
print(l1,np.mean(Y1),h1)

