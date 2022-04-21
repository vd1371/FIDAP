import pandas as pd

def _summarize(**params):
	
	features_importance_instances = params.get("features_importance_instances")
	direc = params.get("direc")
	
	df = pd.DataFrame.from_dict(features_importance_instances)
	output = df.describe()

	output.to_csv(f'{direc}/FIDAP_report.csv')