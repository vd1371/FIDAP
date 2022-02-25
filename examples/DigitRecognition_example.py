try:
    from keras.datasets import mnist
except:
    from tensorflow.keras.datasets import mnist

from matplotlib import pyplot

def DigitRecognition_example():
    # loading dataset
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

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
