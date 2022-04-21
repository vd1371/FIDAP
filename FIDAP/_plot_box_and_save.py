import numpy as np
import matplotlib.pyplot as plt

def _plot_box_and_save(**params):
	
	fiil_values_instances_dict = params.get("features_importance_instances")
	direc = params.get("direc")
	output_fig_format = params.get("output_fig_format")

	features_names = list(fiil_values_instances_dict.keys())
	plt.rcParams['font.family'] = 'Times New Roman'

	fiil_values_instances = list(fiil_values_instances_dict.values())
	
	Golden_ratio = (1 + np.sqrt(5))/2

	fig_length = 4 * len(fiil_values_instances)

	fig = plt.figure()
	plot = plt.boxplot(fiil_values_instances,
						labels=features_names,
						showfliers=False)
	
	plt.xlabel("Features")
	plt.xticks(rotation=90)
	plt.ylabel("FIIL Values")
	plt.tight_layout()


	fig.savefig(f'{direc}/Box_Output.{output_fig_format}', dpi=300)