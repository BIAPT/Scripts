import numpy as np
import pandas as pd

matrix = np.zeros((105,105))
nr = np.arange(0,5460)

a=0
for i in range (1,105):
    rng = np.arange(105 - i)
    fill=nr[a:a+len(rng)]
    matrix[rng, rng+i] = fill
    a=a+len(fill)

mat=pd.DataFrame(matrix)
#mat = mat.applymap(str)
#np.savetxt('foo.txt', mat, fmt='%s',delimiter=';')


exportdata = np.zeros((105,105))
exportdata=pd.DataFrame(exportdata)

eoe

for i in range(0,len(eoe)):
    a=eoe[i]
    p=np.where(mat == a)
    exportdata.iloc[p[1][0],p[0][0]]=1


for i in range(0,len(eoe[0])):
    a=eoe[0][i]
    p=np.where(mat == a)
    exportdata.iloc[p[1][0],p[0][0]]=1

np.where(exportdata==1)
np.savetxt('test_electrode.txt', exportdata, fmt='%s',delimiter=';')

