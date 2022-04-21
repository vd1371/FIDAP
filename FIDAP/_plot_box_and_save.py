import numpy as np
import matplotlib.pyplot as plt

def _plot_box_and_save(**params):
	
	FIDAP_values_instances_dict = params.get("features_importance_instances")
	direc = params.get("direc")
	output_fig_format = params.get("output_fig_format")

	features_names = list(FIDAP_values_instances_dict.keys())
	plt.rcParams['font.family'] = 'Times New Roman'

	FIDAP_values_instances = list(FIDAP_values_instances_dict.values())
	
	Golden_ratio = (1 + np.sqrt(5))/2

	fig_length = 4 * len(FIDAP_values_instances)

	fig = plt.figure()
	plot = plt.boxplot(FIDAP_values_instances,
						labels=features_names,
						showfliers=False)
	
	plt.xlabel("Features")
	plt.xticks(rotation=90)
	plt.ylabel("FIDAP Values")
	plt.tight_layout()


	fig.savefig(f'{direc}/Box_Output.{output_fig_format}', dpi=300)