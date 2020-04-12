import numpy as np
import pandas as pd
import PIL
from PIL import Image
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

train_dir = './train_processed/'
train_files = [train_dir + x for x in os.listdir(train_dir)]

labels = ["b1","b2","b3","b4","b5","b6","b7","b8","b9","Indicator","GMI Longitude","GMI Latitude","DPR Longitude","DPR Latitude","Precipitation"]
features = []
cnt = 0
# process train data
for path in train_files:
    print("Processing train datapoint {}........".format(cnt))
    if path == "./train_processed/.DS_Store":
        continue;
    data = np.load(path)
    # we end up with 1600 rows per data
    for i in range(40):
        for j in range(40):
            single_feature = []
            for k in range(15):
                single_feature.append(data[:,:,k][i][j])
            single_feature = tuple(single_feature)
            features.append(single_feature)
    cnt = cnt+1
# produced dataframe
print("Creating train set dataframe........")
df = pd.DataFrame.from_records(features,columns=labels)
X = df.iloc[:,:-1]
Y = df.iloc[:,-1]
# fit linear regression model
print("Fitting linear regression........")
lm = LinearRegression()
lm.fit(X,Y)
# process test data
test_dir = './test_processed/'
test_files = [test_dir + x for x in os.listdir(test_dir)]

test_labels = ["b1","b2","b3","b4","b5","b6","b7","b8","b9","Indicator","GMI Longitude","GMI Latitude","DPR Longitude","DPR Latitude"]
test_features = []
cnt = 0
for path in test_files:
    print("Processing test data {}........".format(cnt))
    data = np.load(path)
    # we end up with 1600 rows per data
    for i in range(40):
        for j in range(40):
            single_feature = []
            for k in range(14):
                single_feature.append(data[:,:,k][i][j])
            single_feature = tuple(single_feature)
            test_features.append(single_feature)
    cnt = cnt+1
# make prediction
print("Creating test dataframe and making predictions on it........")
df_test = pd.DataFrame.from_records(test_features,columns = test_labels)
prediction = lm.predict(df_test)
# clip prediction
print("Clipping prediction results........")
prediction = np.maximum(prediction,0.0)

# prepare submission file
print("Processing output csv file........")
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
df_submission.to_csv('multiple_linear_regression.csv',index=False)
print("Done!!")
