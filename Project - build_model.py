import pandas as pd
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from utils import INPUT_SHAPE, batch_generator
from keras.layers import Conv2D, Dropout, Dense, Flatten
import argparse


# Reproducible results
np.random.seed(1)


def load_data(para):
    
    # Takes the  CSV file and converts into a single dataframe variable
    df = pd.read_csv('/Users/student/Documents/GitHub/How_to_simulate_a_self_driving_car/driving_log.csv', names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
    
    #
    # Input data - Store the images from our car simulator
    X = df[['center', 'left', 'right']].values
    
    #  Output data - Steering commands
    y = df['steering'].values
    
    
    # Now we can split the data into a training (80), testing(20), and validation set
    
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = para.test_size, random_state = 1)
    
    return X_train, X_valid, y_train, y_valid


def build_model(para):
    
    
    mdl = Sequential()
    mdl.add(keras.layers.Lambda(lambda x: x/127.5-1.0, input_shape = INPUT_SHAPE))
    
    #Feature engineering
    # Convolution: 5x5, filter: 24, 36 and 48, strides: 2x2, activation: ELU
    mdl.add(Conv2D(24, 5, 5, activation='elu', strides = (2, 2)))
    mdl.add(Conv2D(36, 5, 5, activation='elu', strides = (2, 2)))
    mdl.add(Conv2D(48, 5, 5, activation='elu', strides = (2, 2)))
    
    #Feature engineering
    # Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
    mdl.add(Conv2D(64, 3, 3, activation='elu'))
    mdl.add(Conv2D(64, 3, 3, activation='elu'))
    
    #  Dropout rate at 0.5
    mdl.add(Dropout(para.keep_prob))
    mdl.add(Flatten())
    
    # predicting the steering angles of the vehicle
    #  A Fully connected netowrk: (neurons: 100,50,10 and 1), activation: ELU
    mdl.add(Dense(100, activation='elu'))
    mdl.add(Dense(50, activation='elu'))
    mdl.add(Dense(10, activation='elu'))
    mdl.add(Dense(1))
    mdl.summary()
    
    return mdl

# converting a string to boolean value for comman line argument

def bol(string):
    
    string = string.lower()
    return string == 'true' or string == 'yes' or string == 'y' or string == '1'


def trainning(mdl, para, X_train, X_valid, y_train, y_valid):
    
    # Model saved after every epoch.
    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5', monitor='val_loss',verbose=0, best = para.best, mode = 'auto')
    
    # Gradient descent
    mdl.compile(loss ='mean_squared_error', optimizer = Adam(lr = para.learning_rate))
    
    # Fits the model batch-by-batch of the generated data.
    # Parallelisim - Train our model and reshape our data.
    mdl.fit_generator(batch_generator(para.data_dir, X_train, y_train, para.batch_size, True), para.samples_per_epoch, para.nb_epoch, max_q_size=1,
                      validation_data=batch_generator(para.data_dir, X_valid, y_valid, para.batch_size, False),
                      samples = len(X_valid),
                      cb = [checkpoint],
                      verbose=1)
def main():
    
    p = argparse.ArgumentParser(description='Behavioural Cloning')
    p.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='data')
    p.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    p.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    p.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=10)
    p.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=20000)
    p.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=40)
    p.add_argument('-o', help='save best models only', dest='save_best_only',    type=bol,   default='true')
    p.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
    para = p.parse_args()
    
    #print parameters
    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(para).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    #load data
    data = load_data(para)
    
    #build model
    mdl = build_model(para)
    
    #train model on data, it saves as model.h5
    trainning(mdl, para, *data)


if __name__ == '__main__':
    main()
