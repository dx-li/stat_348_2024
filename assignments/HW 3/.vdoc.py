# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
import pandas as pd
import numpy as np
import scipy.linalg as la
```
#
df = pd.read_csv('solubility.csv')
X = df.loc[:, df.columns!='Solubility'].to_numpy()
X = np.hstack((np.ones((X.shape[0],1)),X))
u, s, vh = la.svd(X, full_matrices=False)
print(s)
print(pd.DataFrame(np.round(X@vh.T, 4)))
```
#
#
print(np.round(vh.T[:, -3], 2))
print('intercept', list(df.columns[:-1]))
```
#
#
#
#
Y = df['Solubility'].to_numpy()
Y = Y.T
def obj(beta):
    return np.sum((u.T@Y-s@vh.T@beta)**2)
import scipy.optimize as opt
beta0 = np.zeros(X.shape[1])
res = opt.minimize(obj, beta0)
print(res.x)
print(res.message)
```
#

for j in range(X.shape[1]):
    print((u.T@Y)[j]/s[j])
#
#
#
