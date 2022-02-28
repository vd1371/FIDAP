import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

def get_pixel_importance(img, **params):

	img = np.array([img])

	model = params.get("model")
	preds = model.predict(img)[0]
	max_prob, label = np.max(preds), np.argmax(preds)

	pixel_imps = np.zeros(shape = (28, 28))
	repetition_vector = np.zeros(shape = (28, 28))

	plt.ion()

	pixs = 14
	pad = 1
	padding_times = int((28-pixs)/pad) + 1
	for i in range(padding_times):
		for j in range(padding_times):

			x_init = min(i*pad, 28-pixs)
			x_end = min(i*pad+pixs, 28)
			y_init = min(j*pad, 28-pixs)
			y_end = min(j*pad+pixs, 28)

			print (i, j)

			new_img = img.copy()

			new_img[0, x_init: x_end, y_init: y_end, 0] = \
				np.zeros(shape = (pixs, pixs)) + 0.5
			repetition_vector[x_init: x_end, y_init: y_end] += 1

			new_prob = model.predict(new_img)[0][label]

			if i > 3 or True:
				print (new_prob, max_prob)
				plt.clf()
				plt.imshow(new_img[0].reshape(28, 28),
							cmap=plt.get_cmap('gray'))
				plt.title(f"new{new_prob} - max {max_prob}")
				plt.pause(0.00001)
				plt.draw()

			pixel_imps[x_init: x_end, y_init: y_end] += \
				(1-(max_prob - new_prob)/max_prob)

	pixel_imps = pixel_imps / np.max(repetition_vector)

	plt.ioff()
	fig, ax = plt.subplots(1, 2)
	ax[0].imshow(pixel_imps, cmap=plt.get_cmap('gray'))
	ax[1].imshow(img[0].reshape(28, 28), cmap=plt.get_cmap('gray'))
	plt.show()
 
	raise ValueError("Inside get_pixel_imporatnce")
