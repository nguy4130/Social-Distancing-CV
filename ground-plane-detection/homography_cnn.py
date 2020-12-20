# USAGE
# python transform_example.py --image images/example_01.png --coords "[(73, 239), (356, 117), (475, 265), (187, 443)]"
# python transform_example.py --image images/example_02.png --coords "[(101, 185), (393, 151), (479, 323), (187, 441)]"
# python transform_example.py --image images/example_03.png --coords "[(63, 242), (291, 110), (361, 252), (78, 386)]"

# import the necessary packages
from transform import four_point_transform, resizePatches
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from sklearn.model_selection import train_test_split
import keras.backend as K
import scipy.io
from keras.preprocessing.image import load_img

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization, InputLayer
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras import optimizers
import tensorflow as tf

def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

def buildModel():
    
    batch_size = 32    
    TARGET_PATCH_SIZE = (120, 160) 

    input_shape = (TARGET_PATCH_SIZE[0], TARGET_PATCH_SIZE[1], 2)
    n_variables = 8
    
    model = Sequential()
    model.add(Conv2D(batch_size, kernel_size=(3, 3), strides=(1, 1),
                      activation='relu',
                      input_shape = input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(batch_size, (3, 3), activation='relu')) #32 was 64
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(batch_size, (3, 3), activation='relu')) #32 was 64
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(n_variables)) #activation='softmax')? 
    # model.summary()
    
    model.compile(loss= euclidean_distance_loss,
                  # optimizer=keras.optimizers.adam(lr=0.01),
                  optimizer = 'rmsprop',
                  metrics=['accuracy', 'mean_squared_error']) #mean_squared_error - in valid shape 8 from nn vs 3X3 in test data 
    
    # RootMeanSquaredError
    return model

def build_model_1():
    kernel_size = 3
    pool_size = 2
    filters = 32
    dropout = 0.5
    
    batch_size = 32    
    TARGET_PATCH_SIZE = (120, 160) 

    input_shape = (TARGET_PATCH_SIZE[0], TARGET_PATCH_SIZE[1], 2)
    model = Sequential()
    model.add(InputLayer(input_shape))
    model.add(Conv2D(filters=filters,\
            kernel_size=kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=filters,\
            kernel_size=kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size))
    model.add(Conv2D(filters=filters,\
            kernel_size=kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=filters,\
            kernel_size=kernel_size, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size))
    model.add(Conv2D(filters=filters*2,\
            kernel_size=kernel_size, activation='relu', padding='same',))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=filters*2,\
            kernel_size=kernel_size, activation='relu', padding='same',))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size))
    model.add(Conv2D(filters=filters*2,\
            kernel_size=kernel_size, activation='relu', padding='same',))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=filters*2,\
            kernel_size=kernel_size, activation='relu', padding='same',))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(dropout))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(dropout))
    #for regression model
    model.add(Dense(8))
    
    #use optimizer Stochastic Gradient Methond with a Learning Rate of 0.005 and momentum of 0.9
    #sgd = optimizers.SGD(lr=0.005, momentum=0.9, decay=0.001355)
    sgd = optimizers.SGD(lr=0.005, momentum=0.9)
    
    #compile model
    model.compile(loss=euclidean_distance_loss,\
            # optimizer=sgd, 
            optimizer = 'rmsprop',
            metrics=['mean_squared_error', 'accuracy'])
    return model

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
    
# load and prepare an image
def load_image_pixels(filename, shape):
	# load the image to get its shape
	image = load_img(filename)
	width, height = image.size
	# load the image with the required size
	image = load_img(filename, target_size=shape)
	# convert to numpy array
	image = img_to_array(image)
	# scale pixel values to [0, 1]
	image = image.astype('float32')
	image /= 255.0
	# add a dimension so that we have one sample
	image = expand_dims(image, 0)
	return image, width, height
    
def loadData(folderPath, matFileName):

    mat = scipy.io.loadmat(folderPath + matFileName)
    # mat = list(mat['img_data'])
    if matFileName == 'BBox_Conv.mat':
        mat = list(mat['bbox_coords'])
    elif matFileName == 'BBox_Conv_2.mat':
        mat = list(mat['bbox_conv'])
    else:
        mat = list(mat['img_data'])
        
    print("actual len of coords:"+ str(len(mat)))
    training_data = []
    predictions = []
    fail = []
    for i in range(0, len(mat)):
        
        if matFileName == 'BBox_ConversionRate.mat' or matFileName == 'BBox_Conv.mat' or matFileName == 'BBox_Conv_2.mat':
            image = cv2.imread(folderPath+"Img_"+ str(i+1)+".png")
        else:
            image = cv2.imread(folderPath+"Img_"+ str(i)+".png")

        if image is not None and len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # PAD_SIZE = 80
        # image = np.pad(image, 80, pad_with)
        
        if np.array(mat[i])[0].shape[0] == 4 and np.array(mat[i])[0].shape[1] == 2 :

            pts = np.array(mat[i])[0]
            # pts = pts+ PAD_SIZE
            patch_orig, patch_warped, rect, dst, M, warped, warped_whole = four_point_transform(image, pts)
            if i in [28,29,30]:
                plt.imshow(image)
                plt.title("image"+ str(i))
                plt.show()
                plt.imshow(warped_whole)
                plt.title("warped_whole"+ str(i))
                plt.show()
                plt.imshow(patch_orig)
                plt.title('patch_orig'+ str(i))
                plt.show()
                plt.imshow(patch_warped)
                plt.title('patch_warped'+str(i))
                plt.show()
                plt.imshow(warped)
                plt.title('warped'+str(i))
                plt.show()
            if patch_orig.shape[0]!=0 and patch_warped.shape[0]!=0 and patch_orig.shape[1]!=0 and patch_warped.shape[1]!=0:
                # resize all patches to same size : TARGET_PATCH_SIZE (80, 80)
                patch_orig, patch_warped, rect, dst = resizePatches(patch_orig, patch_warped, rect, dst, image, warped_whole)
                # print("After rescaling:"+ str(patch_orig.shape)+","+ str(rect)+","+ str(dst))
                print(image.shape)
                targetDim = (160,120)
                image_resized = cv2.resize(image, targetDim)
                warped_whole_resized = cv2.resize(warped_whole, targetDim)
                training_image = np.dstack((image_resized, warped_whole_resized))
                # training_image = np.dstack((patch_orig, patch_warped))
                
                H_four_points = rect-dst
                training_data.append(training_image)
                predictions.append(H_four_points.flatten())
            else:
                print("fail: patch_orig.shape: "+ str(patch_orig.shape))
                print("fail: patch_warped.shape: "+ str(patch_warped.shape))
                fail.append(i+1)
    # print(fail)           
    return training_data, predictions
    
    
if __name__ == '__main__':
    
    training_data = []
    predictions = []
    
    # folderPath = "/Users/vcroopana/python_workspace/compVision/cv_proj_py/data/TinyImgs/"
    # matFileName = 'BBox_PPM.mat'
    # training_data, predictions = loadData(folderPath, matFileName)
    # print("size of preds:"+ str(len(predictions)))
    
    folderPath = "/Users/vcroopana/python_workspace/compVision/cv_proj_py/data/TinyImgs_2/"
    matFileName = 'BBox_ConversionRate.mat'
    training_data_2, predictions_2 = loadData(folderPath, matFileName)
    print("size of preds:"+ str(len(predictions_2)))

    training_data.extend(training_data_2)
    predictions.extend(predictions_2)
    print("Final size of preds:"+ str(len(predictions)))
    
    folderPath = "/Users/vcroopana/python_workspace/compVision/cv_proj_py/data/RedoImgs/"
    matFileName = 'BBox_Conv.mat'
    training_data_3, predictions_3 = loadData(folderPath, matFileName)
    print("size of preds:"+ str(len(predictions_3)))
     
    training_data.extend(training_data_3)
    predictions.extend(predictions_3)
    
    folderPath = "/Users/vcroopana/python_workspace/compVision/cv_proj_py/data/RedoImgs_2/"
    matFileName = 'BBox_Conv_2.mat'
    training_data_4, predictions_4 = loadData(folderPath, matFileName)
    print("size of preds:"+ str(len(predictions_4)))
     
    training_data.extend(training_data_4)
    predictions.extend(predictions_4)
    
    print("Final size of preds:"+ str(len(predictions)))
    
    model = buildModel()
    # model = build_model_1()
    training_data = np.asarray(training_data)
        
    print("shape of training data:"+ str(training_data.shape))
    
    predictions = np.asarray(predictions)
    print("shape of training labels:"+ str(predictions.shape))
    
    train_images, test_images, train_homo_vals, test_homo_vals = train_test_split(training_data, predictions, test_size=0.2, random_state=42)
    model.fit(train_images, train_homo_vals,
                  batch_size=32, # batch size was 64
                  epochs= 15,
                   verbose=1
                  # ,validation_data=(training_data, predictions))
                   ,validation_split = 0.1)
                  # ,callbacks=[history])
        
    score = model.evaluate(test_images, test_homo_vals, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

