import numpy as np
import pandas as pd
import PIL
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor
import joblib
import time

train_dir = './train_processed/'
train_files = [train_dir + x for x in os.listdir(train_dir)]

labels = ["b1","b2","b3","b4","b5","b6","b7","b8","b9","Indicator","GMI Longitude","GMI Latitude","DPR Longitude","DPR Latitude","Precipitation"]
features = []
cnt = 0

lm = SGDRegressor(warm_start=True)
# process train data
for path in train_files[:10000]:
    print("Processing train datapoint {}........".format(cnt))
    if path == "./train_processed/.DS_Store":
        continue;
    data = np.load(path)
    # we end up with 1600 rows per data
    features = []
    for i in range(40):
        for j in range(40):
            single_feature = []
            for k in range(15):
                single_feature.append(data[:,:,k][i][j])
            single_feature = tuple(single_feature)
            features.append(single_feature)
    df = pd.DataFrame.from_records(features)
    X = df.iloc[:,:-1]
    Y = df.iloc[:,-1]
    lm.fit(X,Y)
    cnt = cnt+1

# making prediction
test_dir = './test_processed/'
test_files = [test_dir + x for x in os.listdir(test_dir)]
cnt = 0
prediction = []
for path in test_files:
    print("Processing test data {}".format(cnt))
    data = np.load(path)
    for i in range(40):
        for j in range(40):
            single_feature = []
            for k in range(14):
                single_feature.append(data[:,:,k][i][j])
            X = np.asarray(single_feature).reshape(-1,14)
            p = lm.predict(X)
            prediction.append(p[0])
    cnt = cnt+1


print("Clipping prediction results........")
prediction = np.maximum(prediction,0.0)
# prepare submission file
print("Processing submission file.......")
submission_labels = []
submission_features = []
for i in range(1,1601):
    submission_labels.append("px_"+str(i))
ss = pd.read_csv('sample_submission.csv')
splitted = np.split(prediction,len(test_files)) # split into arrays each of 1600 elements
for x in splitted:
    x = tuple(x)
    submission_features.append(x)
df_submission = pd.DataFrame.from_records(submission_features,columns=submission_labels)
df_submission.insert(0,"id",ss['id'],True)
print(df_submission.head(10))
print("Saving output CSV file........")
df_submission.to_csv('sgd_regression_10000.csv',index=False)
print("Done!!")
