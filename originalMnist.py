# creating the original MNIST Dataset

import matplotlib.pyplot as plt
import keras.datasets.mnist as mnist
import numpy as np
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers import Conv2D
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from keras.layers import Dropout
import keras.preprocessing.image as image
from keras.utils import np_utils

# defining the training and the test set

( x_train , y_train ) , ( x_test , y_test ) = mnist.load_data()

# plotting a subplot 

#for i in range( 9 ) : 
#    var = 331 + i
#    plt.subplot( var )
#    plt.imshow( x_train[i])
#plt.show()

# expanding the dimension of the training and the test set

x_train = np.expand_dims( x_train , axis = 3 )
x_test = np.expand_dims( x_test , axis = 3 )

# normalizing the values of x_train , x_test bw 0-1 

x_train = x_train/255 
x_test = x_test/255

# one hot encoding the output variables 

y_train = np_utils.to_categorical( y_train , 10 )
y_test = np_utils.to_categorical( y_test , 10 )

# creating the model 

classifier = Sequential()

classifier.add( Conv2D( filters = 32  , input_shape = ( 28, 28, 1) , kernel_size = ( 3 , 3 ) , activation = 'relu' ))
classifier.add( MaxPooling2D( pool_size = ( 2 , 2)))

classifier.add( Conv2D( filters = 64 , kernel_size = ( 3 , 3), activation = 'relu' ))
classifier.add(MaxPooling2D( pool_size = ( 2 , 2)))

Dropout( 0.2 )

classifier.add( Flatten() )

# adding the fully connected layers to the model

classifier.add( Dense( 128 , activation = 'relu' ))
Dropout( 0.2 )
classifier.add( Dense( 10 , activation = 'sigmoid' ))


# compiling the model 

classifier.compile( optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy' ] )

classifier.fit( x= x_train , y = y_train , batch_size = 32 , validation_data = ( x_test , y_test ) , epochs = 5 )

# evaluating the model 

classifier.evaluate( x_test , y_test )

# predicting th new images
# defining the new fuction to classify images

def predictDigit( filePath ):
    #doing image preprocessing 
    test_image = image.load_img( grayscale = True , path = filePath , target_size = ( 28 , 28 ))
    plt.imshow( test_image )
    test_image = image.img_to_array( test_image )
    test_image = np.expand_dims( test_image , axis = 0 )
    # predicting the digit 
    ans = classifier.predict(test_image )
    print( ' I think the digit is probbably ' , ans )

filePath = 'Screenshot1.jpg'
predictDigit( filePath )










