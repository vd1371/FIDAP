import numpy as np
import matplotlib.pyplot as plt

def _plot_box_and_save(fiil_values_instances_dict,features_names):

	fiil_values_instances = list(fiil_values_instances_dict.values())
	
	Golden_ratio = (1 + np.sqrt(5))/2

	fig_length = 4 * len(fiil_values_instances)

	fig = plt.figure(figsize = (fig_length, fig_length/Golden_ratio))

	# creating the bar plot

	plot = plt.boxplot(fiil_values_instances, labels=features_names)


	for cap in plot['caps']:
		cap.set(color =[0,0,0.9],
			linewidth = 3)

	for whisker in plot['whiskers']:
		whisker.set(color =[0,0,0.7],
			linewidth = 3)

	for boxe in plot['boxes']:
		boxe.set(color =[0,0,0.5],
			linewidth = 3)

	for median in plot['medians']:
		median.set(color =[0,0,0.3],
			linewidth = 3)

	font_size_num = len(features_names)*4

	font_size_tit = len(features_names)*6

	plt.rcParams['font.size'] = font_size_num

	plt.rcParams['font.family'] = 'Times New Roman'

	plt.xlabel("features", fontsize=font_size_tit)

	plt.ylabel("features importance value", fontsize=font_size_tit)

	plt.title("feature importance analysis", fontsize=font_size_tit)

	plt.show()

	fig.savefig('Box_Output.svg',dpi=200)