

def _get_preds(X, model, pred_fn):

	if pred_fn == None:
		return model.predict(X)

	else:
		'''
		assuming model is an object and has a function with the name of 
		pred_fn
		Please also put a try and except to capture errors in case model doesn't
		have the "pred_fn" method	
		'''
