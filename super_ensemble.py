import glob
import pandas as pd
import numpy as np

csv1 = pd.read_csv('super_ensemble2.csv')
csv2 = pd.read_csv('super_ensemble3.csv')
csv3 = pd.read_csv('super_ensemble4.csv')

csv1 = np.asarray(csv1)
csv2 = np.asarray(csv2)
csv3 = np.asarray(csv3)


data = []
for i in range(0,2416):
    data.append([])
for i in range(0,len(data)):
    for j in range(0,1601):
        if j == 0:
            data[i].append(csv1[i][j])
        else:
            data[i].append(0.0)

print(len(data), '  ', len(data[0]))

for i in range(0,len(data)):
    for j in range(1,len(data[i])):
        norain = 0
        if csv1[i][j] < 0.1:
            norain += 1
        if csv2[i][j] < 0.1:
            norain += 1
        if csv3[i][j] < 0.1:
            norain += 1


        rainval = 0.0
        raindenom = 0.0

        if csv1[i][j] >= 0.1:
            rainval += csv1[i][j]
            raindenom += 1.0
        if csv2[i][j] >= 0.1:
            rainval += csv2[i][j]
            raindenom += 1.0
        if csv3[i][j] >= 0.1:
            rainval += csv3[i][j]
            raindenom += 1.0


        if raindenom != 0.0:
            rainval *= 1.0/raindenom
        if norain >= 2:
            data[i][j] = 0.0
            continue
        if norain < 2:
            data[i][j] = rainval
            continue

cols = ['id']
for i in range(1,1601):
    name = 'px_' + str(i)
    cols.append(name)
df = pd.DataFrame(data,columns=cols)
print(df.head(10))
df.to_csv('super_ensemble_blue.csv',index=False)
