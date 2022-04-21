try:
    from keras.datasets import mnist
except:
    from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from matplotlib import pyplot

import os

os.environ["CUDA_VISIBLE_DEVICES"]= "-1"

from FIIL import PixelImportanceAnalyzer

def DigitRecognition_example():
    # loading dataset
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    X_train = X_train[:,:]/255.
    X_test = X_test[:,:]/255.

    X_train = X_train.reshape(X_train.shape[0],28,28,1)
    X_test = X_test.reshape(X_test.shape[0],28,28,1)

    model = load_model('examples/MNIST.h5')

    fiil = PixelImportanceAnalyzer(model)
    fiil.get(X_test[2], Y_test[2])

    raise ValueError ("Inside DigitRecognition_example")

    #shape of dataset
    print('X_train: ' + str(X_train.shape))
    print('Y_train: ' + str(Y_train.shape))
    print('X_test:  '  + str(X_test.shape))
    print('Y_test:  '  + str(Y_test.shape))



    # plotting
    for i in range(9):
        pyplot.subplot(550 + 1 + i)
        pyplot.imshow(X_train[i], cmap=pyplot.get_cmap('gray'))
        pyplot.show()
