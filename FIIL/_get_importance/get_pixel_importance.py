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

	for dif_pad_x in range(4,20,4):

		pad_x = dif_pad_x

		padding_times_x = (int(28/pad_x)+1) if 28%pad_x>0 else (int(28/pad_x))

		for dif_pad_y in range(4,20,4):

			pad_y = dif_pad_y

			padding_times_y = (int(28/pad_y)+1) if 28%pad_y>0 else (int(28/pad_y))


			for i in range(padding_times_x):

				for j in range(padding_times_y):

					x_start = i*pad_x
					x_end = min(x_start+pad_x, 28)
					y_start = j*pad_y
					y_end = min(y_start+pad_y, 28)

					new_img = img.copy()

					new_img[0, x_start: x_end, y_start: y_end, 0] = \
						np.zeros(shape = ((x_end-x_start), (y_end-y_start))) + 0.5

					repetition_vector[x_start: x_end, y_start: y_end] += 1

					new_prob = model.predict(new_img)[0][label]

					print (new_prob, max_prob)

					plt.clf()
					plt.imshow(new_img[0].reshape(28, 28),
								cmap=plt.get_cmap('gray'))

					plt.title(f"new{new_prob} - max {max_prob}")
					plt.pause(0.0001)
					plt.draw()

					pixel_imps[x_start: x_end, y_start: y_end] += \
							(max_prob - new_prob)

		final_image += pixel_imps

	plt.imshow(final_image, interpolation='nearest',cmap=plt.get_cmap('gray'))
	plt.show()

		# (1-(max_prob - new_prob)/max_prob)
		# (max_prob - new_prob)