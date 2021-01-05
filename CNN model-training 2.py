import csv
import cv2
from os import path
import numpy as np
import matplotlib.pyplot as plt
from keras.cnn_cnn_models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from sklearn.utils import shuffle
from sklearn.cnn_cnn_model_selection import train_test_split


def load(Path):
 
    lines = []
    with open(path.join(Path,'driving_log.csv')) as f:
        content = csv.reader(f)
        for line in content:
            lines.append(line)

    return lines


def brightness(image):

    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = np.random.uniform(0.2,0.8)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)

    return image1


def network_cnn_model():

    cnn_model = Sequential()
    cnn_model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(160,320,3)))
    cnn_model.add(Cropping2D(cropping=((70,25),(0,0))))
    
    cnn_model.add(Convolution2D(32, (3, 3), activation='relu',input_shape=(1,28,28), data_format='channels_first', padding='SAME'))
    cnn_model.add(MaxPooling2D())
    cnn_model.add(Dropout(0.1))
    
    cnn_model.add(Convolution2D(64, (3, 3), activation='relu', input_shape=(1,28,28), data_format='channels_first', padding='SAME'))
    cnn_model.add(MaxPooling2D())
    cnn_model.add(Dropout(0.1))
    
    cnn_model.add(Convolution2D(128, (3, 3), activation='relu', input_shape=(1,28,28), data_format='channels_first', padding='SAME'))
    cnn_model.add(MaxPooling2D())
    
    cnn_model.add(Convolution2D(256, (3, 3), activation='relu', input_shape=(1,28,28), data_format='channels_first', padding='SAME'))
    cnn_model.add(MaxPooling2D())

    cnn_model.add(Flatten())
    cnn_model.add(Dense(120))
    cnn_model.add(Dense(20))
    cnn_model.add(Dense(1))
    
    return cnn_model

def balance_data(sample, visulization_flag ,N=60, K=1,  bins=100):


    angles = []
    for line in sample:
        angles.append(float(line[3]))
    
    # n = bins in histogram
    n, bins, patches = plt.hist(angles, bins=bins, color= 'red', linewidth=0.1)
    angles = np.array(angles)
    n = np.array(n)
    
    # Largest K bins
    idx = n.argsort()[-K:][::-1]    
    
    # index which will be removed
    index_delete = []                    
    for i in range(K):
        if n[idx[i]] > N:
            ind = np.where((bins[idx[i]]<=angles) & (angles<bins[idx[i]+1]))
            ind = np.ravel(ind)
            np.random.shuffle(ind)
            index_delete.extend(ind[:len(ind)-N])

    # collect data without the index_delete data (deleted column)
    balanced_sample = [v for i, v in enumerate(sample) if i not in index_delete]
    balanced_angles = np.delete(angles,index_delete)

    plt.subplot(1,2,2)
    plt.hist(balanced_angles, bins=bins, color= 'red', linewidth=0.1)
    plt.title('modified histogram', fontsize=20)
    plt.xlabel('steering angle', fontsize=20)
    plt.ylabel('counts', fontsize=20)

    if visulization_flag:
        plt.figure
        plt.subplot(1,2,1)
        n, bins, patches = plt.hist(angles, bins=bins, color='red', linewidth=0.1)
        plt.title('Unbalanced data', fontsize=20)
        plt.xlabel('steering angle', fontsize=20)
        plt.ylabel('counts', fontsize=20)
        plt.show()

        plt.figure
        aa = np.append(balanced_angles, -balanced_angles)
        bb = np.append(aa, aa)
        plt.hist(bb, bins=bins, color='orange', linewidth=0.1)
        plt.title('Balanced data', fontsize=20)
        plt.xlabel('steering angle', fontsize=20)
        plt.ylabel('counts', fontsize=20)
        plt.show()

    return balanced_sample



def augment(png, angles):

    augmented_png = []
    augmented_angles = []
    for image, angle in zip(png, angles):

        augmented_png.append(image)
        augmented_angles.append(angle)

        # flip images
        flipped_image = cv2.flip(image,1)
        flipped_angle = -1.0 * angle
        augmented_png.append(flipped_image)
        augmented_angles.append(flipped_angle)

        # change brigthness of image
        image_b1 = brightness(image)
        image_b2 = brightness(flipped_image)
        
        # append png
        augmented_angles.append(angle)
        augmented_angles.append(flipped_angle)
        
        augmented_png.append(image_b1)
        augmented_png.append(image_b2)
        


    return augmented_png, augmented_angles





def new_data(sample, train_flag, batch_size=32):
    
    num_sample = len(sample)
    
    # Value used to correct png from left to right
    correction = 0.2  
    
    # Infinite loop for new_data 
    while 1:  
        shuffle(sample)
        for offset in range(0, num_sample, batch_size):
            batch_sample = sample[offset:offset+batch_size]

            png = []
            angles = []

            for line in batch_sample:
                angle = float(line[3])
                c_imagePath = line[0].replace(" ", "")
                c_image = cv2.imread(c_imagePath)
                png.append(c_image)
                angles.append(angle)
                
                # Adding left and right png for training data
                if train_flag:  
                    left_path = line[1].replace(" ", "")
                    right_path = line[2].replace(" ", "")
                    left_im = cv2.imread(left_path)
                    right_im = cv2.imread(right_path)
                    png.append(left_im)
                    png.append(right_im)
                    angles.append(angle - correction)
                    angles.append(angle + correction)
                    

            # returns 3 new augmented png from one input image
            augmented_png, augmented_angles = augment(png, angles)

            X_train = np.array(augmented_png)
            y_train = np.array(augmented_angles)
            
            yield shuffle(X_train, y_train)



# load file
Path = '/Users/student/Documents/GitHub/behavioral-cloning'
print('loading...')
sample = load(Path)

# Histogram of steering angles - balanced data
sample = balance_data(sample, visulization_flag=True)

# split data into training and validation
train_sample, validation_sample = train_test_split(sample, test_size=0.3)

# cnn_cnn_model - compile and train
train_new_data = new_data(train_sample, train_flag=True, batch_size=32)
validation_new_data = new_data(validation_sample, train_flag=False, batch_size=32)

# define the cnn network cnn_model
cnn_model = network_cnn_model()
cnn_model.summary()

epch = 4
cnn_model.compile(loss='mse', optimizer='adam')

history = cnn_model.fit_new_data(train_new_data, steps_per_epoch=len(train_sample)*12,
                              epochs=epch, validation_data=validation_new_data)

cnn_model.save('/Users/student/Documents/GitHub/behavioral-cloning/cnn_model.h6')
