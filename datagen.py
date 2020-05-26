import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, directories, avgs, stds, batch_size=516, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.directories = directories
        self.avgs = avgs
        self.stds = stds
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.directories) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        directories_temp = [self.directories[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(directories_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.directories))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, directories_temp):

        X = []
        Y = []
        for file in directories_temp:
            flag90 = file[0:6]
            flag180 = file[0:7]
            flag270 = file[0:7]
            currfile = file
            action = 'nothing'
            if flag90 == 'rot90_':
                action = flag90
                currfile = file[6:]
            if flag270 == 'rot270_':
                action = flag270
                currfile = file[7:]
            if flag180 == 'rot180_':
                action = flag180
                currfile = file[7:]

            datapoint = np.load(currfile)


            for i in range(0,40):
                for j in range(0,40):
                    tempval = float(int(str(datapoint[i][j][9])[:1]))
                    datapoint[i][j][9] = (np.log(tempval+1.0) - self.avgs[9])/(self.stds[9]+0.0000001)
            for i in range(0,14):
                if i != 9:
                    datapoint[:,:,i] = (np.log(datapoint[:,:,i]+1.0)-self.avgs[i])/(self.stds[i]+0.0000001)

    #        noise = np.random.normal(0.0,0.05,22400).reshape(40,40,14)
    #        datapoint[:,:,:14] += noise

            if action == 'rot90_':
                for i in range(0,15):
                    datapoint[:,:,i] = np.rot90(datapoint[:,:,i])

            if action == 'rot180_':
                for i in range(0,15):
                    datapoint[:,:,i] = np.rot90(datapoint[:,:,i])
                    datapoint[:,:,i] = np.rot90(datapoint[:,:,i])

            if action == 'rot270_':
                for i in range(0,15):
                    datapoint[:,:,i] = np.rot90(datapoint[:,:,i])
                    datapoint[:,:,i] = np.rot90(datapoint[:,:,i])
                    datapoint[:,:,i] = np.rot90(datapoint[:,:,i])

            feature = datapoint[:,:,:10]
            target = datapoint[:,:,-1].reshape(40,40,1)
            X.append(feature)
            Y.append(target)
        X = np.asarray(X)
        Y = np.asarray(Y)

        return X,Y


train_gen = DataGenerator(trainfiles, avgs, stds, batch_size=500, shuffle=True)
val_gen = DataGenerator(valfiles, avgs, stds, batch_size=500, shuffle=True)
