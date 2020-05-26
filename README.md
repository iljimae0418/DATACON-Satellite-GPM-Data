# DATACON Satellite GPM Data Contest 

## Introduction  
[Contest Link](https://dacon.io/competitions/official/235591/overview/) 


[Very rough preliminary data analysis](https://github.com/iljimae0418/DATACON-Satellite-GPM-Data/blob/master/DACON%20precipitation%20prediction%20contest%20preliminary%20data%20analysis%20%2B%20Simple%20Linear%20Regression.ipynb)

Brief Explanation of data: We are given 9 microwave images (of different wavelengths) of a particular region and these images are represented by images of size 40x40. We are also given information about the geographical region (encoded as numbers e.g. pixels encoded as 0 are ocean, pixels encoded as 1 are land etc) and the longitude/latitude information about each of the pixels. The output of the model should be a 40x40 pixels indicating the precipitation value for each pixel. 

Scoring metric: MAE/F1. MAE is calculated only for pixels where the actual value of the precipitation is greater than 0.1. F1 score is calcualted after converting all the pixels with precipitation >= 0.1 to be 1 and precipitation < 0.1 to be 0. 

Final Result: Top 10% on the private leaderboard (team name: Puzzle Collectors, ranked 17th out of 213 teams). Public leaderboard score of 1.53, private leaderboard score of 1.59. 

## Environments 
- Ran simple baseline models on local computer 
- Migrated to Paperspace Gradient (cloud GPU service) for running deep learning models. Utilized P4000,P5000 and P6000 GPUs. 

## Methods 
Baseline models: Applied linear regression (SGDRegressor) and Random Forest after seperating data into individual pixels. The python scripts are in folder LinearRegression and RegressionTrees. These baseline models recorded a MAE/F1 score of around 6.06 on the public leaderboard. 

UNet: The next step was to use UNet. UNet is a form of CNN that is commonly used for image segmentation problems. It has an encoder-decoder structure that has a contracting path (reduces dimensions via maxpooling) and an expanding path (using Conv2DTranspose) thus resembling a U-structure. The baseline UNet model in the folder UNET was able to score an MAE/F1 score of 2.3 on the leaderboard. A modified structure unet_tall.py structure when trained on 80% of the full training data (and validated on 20%) scored a MAE/F1 score of 1.63 on the public leaderboard. Upon average ensembling two UNet models of similar structure, we were able to pull the score up to 1.61. 

ResNet: We realised that pixels that are far away from the regions where there is rainfall should not affect the regions where there is rainfall. The UNet structure compresses the dimensions and tries to make spatial connections with regions of rainfall and regions that are far away. We thought this may harm performance so we decided to get rid of the contracting/expanding path and just proceeded with a simple residual network structure. Examples of resnet structures that was used can be seen in the folder Neural networks, the files being resnet_fold1.ipynb and resnet_fold2.ipynb. The ResNet structure was the most successful and a well trained model was able to score 1.55 on the leaderboard and upon ensembling ResNet models with UNet and Xception models, we were able to achieve a final score of 1.53 on the public leaderboard. 


## Experiments    
- Xception: We tried to replace the Conv2D in the resnet structure with SeparableConv2D, but it did not increase speed nor did it dramatically improve the performance of the network (still scored around 1.61 similar to the UNet). 

- K-fold cross validation: Instead of the simple train/validation split we also tried to use k-fold cross validation. Examples of this can be seen in the folders resnet_fold1.ipynb and resnet_fold2.ipynb. Due to the lack of time we only tried for k=2, where one model was trained on the first half of the data then validated on the second half and one model was trained on the second half of the data and validated on the first half. They were then ensembled. Although the models being trained for the two folds were the same, we tried to add some variation by using different data augmentation methods for each training fold. More variability/diversity means we can expect a more dramatic result upon ensembling.   

- Data Preprocessing: There were over 76,000 data for the train set and 2416 data for the test set. The train set had a bit of a data imbalance as there were many datapoints where most of the precipitation values were 0 (and the precipitation channels were very sparse in general). We thus only used datapoints that had the sum of all the precipitation values > 50. We also got rid of datapoints that contained some missing values (represented as -9999) in the train data. We also augmented the training set data by applying rotations (90,180,270 degrees) and by transposing the data. In general, training with more data (with augmentation) improved the result and if we had time we could have collected more data from the NASA website instead of simply augmenting. 

- Custom loss function: Instead of simply using the built in 'mae' loss function in keras, we decdied to write a custom loss function that was more relevant to the problem. Specifically we tried MAE/dice loss and MAE/binary cross entropy. We could not directly use F1 because it is not differentiable. 

- Making the training more difficult for better generalization: We tried adding Gaussian noise at the beginning in the input data to make training more difficult. Also, added shuffling of the batches. Later on we also experimented with adding dropouts to our residual NN models. 

- BatchNormalization: Instead of (manually) normalizing the inputs before feeding it into the network, we decided to BatchNormalize the inputs first and then feed the input into the neural network.

- Experimented with various batch sizes such as 128,512,1024,100,20,16 etc but eventually decided to run with a batch size of 1 as the network seemed to train much faster and the performance was not bad too. 

- Concatenation instead of add for residual (skip) connections 

- Instead of the simple average ensemble, we came up with our own method of ensembling. We ensembled an odd number of predictions (each from a different model) and if more than half of models agreed that there was no rain, then we set the prediction to zero, and if less than half of the models agreed that there was no rain, we take the average of the precipitations. This seemed to be much better than a simple averaging ensemble. 
