import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

def get_pixel_importance2(img, **params):

	img = np.array([img])

	model = params.get("model")
	n_simulations = params.get("n_simulations")

	preds = model.predict(img)[0]
	max_prob, label = np.max(preds), np.argmax(preds)

	pixel_imps = np.zeros(shape = (28, 28))
	final_image = pixel_imps
	repetition_vector = np.zeros(shape = (28, 28))

	plt.ion()

	for i in range(n_simulations):
		new_img = img.copy()

		rand = np.random.random()
		mask = np.random.choice([0, 1], size = (28, 28), p = [rand, 1-rand])
		mask = np.random.choice([0, 1], size = 28, p = [rand, 1-rand])

		# new_img[0, x_start: x_end, y_start: y_end, 0] = \
		# 	np.zeros(shape = ((x_end-x_start), (y_end-y_start))) + 0.5
		
		new_img[0, mask, 0] = 0.5

		# repetition_vector[x_start: x_end, y_start: y_end] += 1

		new_prob = model.predict(new_img)[0][label]

		print (new_prob, max_prob)

		plt.clf()
		plt.imshow(new_img[0].reshape(28, 28),
					cmap=plt.get_cmap('gray'))

		plt.title(f"new{new_prob} - max {max_prob}")
		plt.pause(0.0001)
		plt.draw()

		# pixel_imps[x_start: x_end, y_start: y_end] += \
		# 		(max_prob - new_prob)

		# final_image += pixel_imps

	plt.clf()
	plt.ioff()
	plt.imshow(final_image, interpolation='nearest',cmap=plt.get_cmap('gray'))
	plt.show()

		# (1-(max_prob - new_prob)/max_prob)
		# (max_prob - new_prob)