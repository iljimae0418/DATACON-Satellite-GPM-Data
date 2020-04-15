import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import time

train_dir = './train_processed/'
train_files = [train_dir + x for x in os.listdir(train_dir)]
cnt = 0
rf_reg = RandomForestRegressor(warm_start=True,n_estimators=1000,random_state=156,verbose=3,n_jobs=-1)
# batch training
for t in range(0,len(train_files),7):
    print("Processing train datapoint {}........".format(cnt))
    path = train_files[t]
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
    rf_reg.fit(X,Y)
    rf_reg.n_estimators += 1000
    cnt = cnt+1

# call dataframe
print("loading test dataframe........")
test_df = pd.read_csv('test_df.csv')
print(test_df.shape)
# make predictions
print("Making Prediction......")
start_time = time.time()
prediction = rf_reg.predict(test_df)
print("Took {} seconds".format(time.time()-start_time))
print("Clipping prediction results........")
prediction = np.maximum(prediction,0.0)
# prepare submission file
print("Processing submission file.......")
submission_labels = []
submission_features = []
for i in range(1,1601):
    submission_labels.append("px_"+str(i))
ss = pd.read_csv('sample_submission.csv')
splitted = np.split(prediction,2416) # split into arrays each of 1600 elements
for x in splitted:
    x = tuple(x)
    submission_features.append(x)
df_submission = pd.DataFrame.from_records(submission_features,columns=submission_labels)
df_submission.insert(0,"id",ss['id'],True)
print(df_submission.head(10))
print("Saving output CSV file........")
df_submission.to_csv('rf_reg_skip_2.csv',index=False)
print("Done!!")
