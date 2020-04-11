import numpy as np
import PIL
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
def chk(x):
    cnt = 0
    for i in range(40):
        for j in range(40):
            if int(x[i][j]) == -9999:
                cnt += 1
    return cnt==1600

def impute(x):
    cnt = 0
    sum = 0
    for i in range(40):
        for j in range(40):
            if int(x[i][j]) != -9999:
                sum += x[i][j]
                cnt += 1
    avg = sum/cnt
    for i in range(40):
        for j in range(40):
            if int(x[i][j]) == -9999:
                x[i][j] = avg
    return x

train_dir = './train/'
train_files = [train_dir + x for x in os.listdir(train_dir)]
save_dir = './train_processed/'


# global min and max
min_b = 1e18
max_b = -1e18
gmi_long_max = -1e18
gmi_long_min = 1e18
gmi_lat_max = -1e18
gmi_lat_min = 1e18
dpr_long_max = -1e18
dpr_long_min = 1e18
dpr_lat_max = -1e18
dpr_lat_min = 1e18
iter = []
cnt = 0
for path in train_files:
    print("processing {}...".format(cnt))
    data = np.load(path)
    if chk(data[:,:,14]) == False:
        iter.append(path)
    for i in [0,1,2,3,4,5,6,7,8,10,11,12,13]:
        if i < 10:
            min_b = min(min_b,np.min(data[:,:,i]))
            max_b = max(max_b,np.max(data[:,:,i]))
        elif i == 10:
            gmi_long_min = min(gmi_long_min,np.min(data[:,:,i]))
            gmi_long_max = max(gmi_long_max,np.max(data[:,:,i]))
        elif i == 11:
            gmi_lat_min = min(gmi_lat_min,np.min(data[:,:,i]))
            gmi_lat_max = max(gmi_lat_max,np.max(data[:,:,i]))
        elif i == 12:
            dpr_long_min = min(dpr_long_min,np.min(data[:,:,i]))
            dpr_long_max = max(dpr_long_max,np.max(data[:,:,i]))
        elif i == 13:
            dpr_lat_min = min(dpr_lat_min,np.min(data[:,:,i]))
            dpr_lat_max = max(dpr_lat_max,np.max(data[:,:,i]))
    cnt += 1
print("brightness: min = {}, max = {}".format(min_b,max_b))
print("gmi longitude: min = {}, max = {}".format(gmi_long_min,gmi_long_max))
print("gmi latitude: min = {}, max = {}".format(gmi_lat_min,gmi_lat_max))
print("dpr longitude: min = {}, max = {}".format(dpr_long_min,dpr_long_max))
print("dpr latitude: min = {}, max = {}".format(dpr_lat_min,dpr_lat_max))

cnt = 0
for path in iter:
    print("File {}........".format(cnt))
    data = np.load(path)

    # discard if label is missing, if not impute with mean
    impute(data[:,:,14])

    # normalize first 9 channels
    for i in range(0,9):
        data[:,:,i] = (data[:,:,i]-min_b)/(max_b-min_b)

    # convert indicator type channel
    for i in range(0,40):
        for j in range(0,40):
            val = str(data[:,:,9][i][j])[0]
            #val = np.floor(val/100.0)
            if val == '0':
                data[:,:,9][i][j] = 0.5
            elif val == '1':
                data[:,:,9][i][j] = 1.0
            elif val == '2':
                data[:,:,9][i][j] = 1.5
            elif val == '3':
                data[:,:,9][i][j] = 2.0

    # normalize longitude and latitude
    data[:,:,10] = (data[:,:,10]-gmi_long_min)/(gmi_long_max-gmi_long_min)
    data[:,:,11] = (data[:,:,11]-gmi_lat_min)/(gmi_lat_max-gmi_lat_min)
    data[:,:,12] = (data[:,:,12]-dpr_long_min)/(dpr_long_max-dpr_long_min)
    data[:,:,13] = (data[:,:,13]-dpr_lat_min)/(dpr_lat_max-dpr_lat_min)

    cnt = cnt+1
    np.save(save_dir+path[8:]+"_processed",data)


print("================NOW TEST DATA==============")
test_dir = './test/'
test_files = [test_dir + x for x in os.listdir(test_dir)]
save_dir = './test_processed/'

# global min and max
min_b = 1e18
max_b = -1e18
gmi_long_max = -1e18
gmi_long_min = 1e18
gmi_lat_max = -1e18
gmi_lat_min = 1e18
dpr_long_max = -1e18
dpr_long_min = 1e18
dpr_lat_max = -1e18
dpr_lat_min = 1e18
cnt = 0
for path in test_files:
    print("processing {}...".format(cnt))
    data = np.load(path)
    for i in [0,1,2,3,4,5,6,7,8,10,11,12,13]:
        if i < 10:
            min_b = min(min_b,np.min(data[:,:,i]))
            max_b = max(max_b,np.max(data[:,:,i]))
        elif i == 10:
            gmi_long_min = min(gmi_long_min,np.min(data[:,:,i]))
            gmi_long_max = max(gmi_long_max,np.max(data[:,:,i]))
        elif i == 11:
            gmi_lat_min = min(gmi_lat_min,np.min(data[:,:,i]))
            gmi_lat_max = max(gmi_lat_max,np.max(data[:,:,i]))
        elif i == 12:
            dpr_long_min = min(dpr_long_min,np.min(data[:,:,i]))
            dpr_long_max = max(dpr_long_max,np.max(data[:,:,i]))
        elif i == 13:
            dpr_lat_min = min(dpr_lat_min,np.min(data[:,:,i]))
            dpr_lat_max = max(dpr_lat_max,np.max(data[:,:,i]))
    cnt += 1
print("brightness: min = {}, max = {}".format(min_b,max_b))
print("gmi longitude: min = {}, max = {}".format(gmi_long_min,gmi_long_max))
print("gmi latitude: min = {}, max = {}".format(gmi_lat_min,gmi_lat_max))
print("dpr longitude: min = {}, max = {}".format(dpr_long_min,dpr_long_max))
print("dpr latitude: min = {}, max = {}".format(dpr_lat_min,dpr_lat_max))


cnt = 0
for path in test_files:
    print("File {}........".format(cnt))
    data = np.load(path)

    # normalize first 9 channels
    for i in range(0,9):
        data[:,:,i] = (data[:,:,i]-min_b)/(max_b-min_b)

    # convert indicator type channel
    for i in range(0,40):
        for j in range(0,40):
            val = str(data[:,:,9][i][j])[0]
            #val = np.floor(val/100.0)
            if val == '0':
                data[:,:,9][i][j] = 0.5
            elif val == '1':
                data[:,:,9][i][j] = 1.0
            elif val == '2':
                data[:,:,9][i][j] = 1.5
            elif val == '3':
                data[:,:,9][i][j] = 2.0

    # normalize longitude and latitude
    data[:,:,10] = (data[:,:,10]-gmi_long_min)/(gmi_long_max-gmi_long_min)
    data[:,:,11] = (data[:,:,11]-gmi_lat_min)/(gmi_lat_max-gmi_lat_min)
    data[:,:,12] = (data[:,:,12]-dpr_long_min)/(dpr_long_max-dpr_long_min)
    data[:,:,13] = (data[:,:,13]-dpr_lat_min)/(dpr_lat_max-dpr_lat_min)

    cnt = cnt+1
    np.save(save_dir+path[8:]+"_processed",data)
