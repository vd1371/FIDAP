import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

def get_pixel_importance(img, **params):

	img = np.array([img])

	model = params.get("model")
	preds = model.predict(img)[0]
	max_prob, label = np.max(preds), np.argmax(preds)

	pixel_imps = np.zeros(shape = (28, 28))
	final_image = pixel_imps
	repetition_vector = np.zeros(shape = (28, 28))

	plt.ion()

	repetition_vector

		# final_image += pixel_imps

	pixel_imps /= repetition_vector

	plt.clf()
	plt.ioff()
	plt.imshow(pixel_imps, interpolation='nearest',cmap=plt.get_cmap('gray'))
	plt.show()

		# (1-(max_prob - new_prob)/max_prob)
		# (max_prob - new_prob)