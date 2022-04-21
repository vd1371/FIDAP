from FIDAP import FeatureImportanceAnalyzer

import pprint

def _analyze(*args, **kwargs):
	
	FIDAP = FeatureImportanceAnalyzer(*args, **kwargs)
	pprint.pprint(FIDAP.get())
	FIDAP.boxplot()