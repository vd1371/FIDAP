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

	pixs = 6
	square = 6
	pad = 2
	pad_x = 2
	pad_y = 2
	pad_x_temp = pad_x
	pad_y_temp = pad_y

	padding_times = int((28-pixs)/pad) + 1

	while pad_x_temp <= square:

		while pad_y_temp <= square:

			x_start = 0
			x_end = pad_x_temp
			y_start = 0
			y_end = pad_y_temp

			new_img = img.copy()

			new_img[0, x_start: x_end, y_start: y_end, 0] = \
				np.zeros(shape = (x_end, pad_y_temp)) + 0.5
			repetition_vector[x_start: x_end, y_start: y_end] += 1

			new_prob = model.predict(new_img)[0][label]

			print (new_prob, max_prob)
			plt.clf()
			plt.imshow(new_img[0].reshape(28, 28),
							cmap=plt.get_cmap('gray'))
			plt.title(f"new{new_prob} - max {max_prob}")
			plt.pause(0.1)
			plt.draw()

			pixel_imps[x_start: x_end, y_start: y_end] += \
				(max_prob - new_prob)
				# (1-(max_prob - new_prob)/max_prob)
				# (max_prob - new_prob)
			pad_y_temp += pad_y
		for j in range(1,padding_times):
			x_start = 0
			x_end = pad_x_temp
			y_start = min(j*pad_y, 28-square)
			y_end = min(j*pad_y+square, 28)
			
			new_img = img.copy()

			new_img[0, x_start: x_end, y_start: y_end, 0] = \
				np.zeros(shape = (x_end, square)) + 0.5
			repetition_vector[x_start: x_end, y_start: y_end] += 1

			new_prob = model.predict(new_img)[0][label]

			print (new_prob, max_prob)
			plt.clf()
			plt.imshow(new_img[0].reshape(28, 28),
							cmap=plt.get_cmap('gray'))
			plt.title(f"new{new_prob} - max {max_prob}")
			plt.pause(0.1)
			plt.draw()

			pixel_imps[x_start: x_end, y_start: y_end] += \
				(max_prob - new_prob)
				# (1-(max_prob - new_prob)/max_prob)
				# (max_prob - new_prob)
		pad_y_temp = pad_y
		while pad_y_temp <= square:

			x_start = 0
			x_end = pad_x_temp
			y_start = 28-square+pad_y_temp
			y_end = 28

			new_img = img.copy()

			new_img[0, x_start: x_end, y_start: y_end, 0] = \
				np.zeros(shape = (x_end, square-pad_y_temp)) + 0.5
			repetition_vector[x_start: x_end, y_start: y_end] += 1

			new_prob = model.predict(new_img)[0][label]

			print (new_prob, max_prob)
			plt.clf()
			plt.imshow(new_img[0].reshape(28, 28),
							cmap=plt.get_cmap('gray'))
			plt.title(f"new{new_prob} - max {max_prob}")
			plt.pause(0.1)
			plt.draw()

			pixel_imps[x_start: x_end, y_start: y_end] += \
				(max_prob - new_prob)
				# (1-(max_prob - new_prob)/max_prob)
				# (max_prob - new_prob)
			pad_y_temp += pad_y
		pad_x_temp += pad_x
		pad_y_temp = pad_y