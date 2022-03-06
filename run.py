from examples import *
from examples.Classification import *
from examples.Regression import *

def run():

	# Classification examples
	RF_example()
	#KNN_example()
	#LogR_example()
	#SVC_example()
	#MLP_example()
	#DT_example()
	#ET_example()
	#RN_example()
	#PA_example()
	#GBC_example()
	#CBC_example()

	# Regression examples
	#DNN_example()
	#DTR_example()
	#ETR_example()
	#GBR_example()
	#LR_example()
	#NB_example()
	#PAR_example() #this example works but gives negative R2 score value.
	#SVR_example()
	#XGB_example()

	# Clustering examples
	#KMClst_example()
	#MSClst_example()

	# Image example
	#DigitRecognition_example()


if __name__ == "__main__":
	run()
