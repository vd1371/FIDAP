from FIIL import FeatureImportanceAnalyzer

import pprint

def _analyze(*args, **kwargs):
	
	fiil = FeatureImportanceAnalyzer(*args, **kwargs)
	pprint.pprint(fiil.get())
	fiil.boxplot()