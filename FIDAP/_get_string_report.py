


def _get_string_report(feature_importance):

	max_len = 80
	len_features = 70

	report_str = "Feature"

	for j in range(max_len - len(report_str) - 7):
		report_str += ' '
	report_str += "FIDAP  \n"

	for j in range(max_len):
		report_str += "-"
	report_str += '\n'

	for k, v in feature_importance.items():

		len_k = min(len(k), len_features)

		report_str += k[:len_k]
		
		if len_k == len_features:
			report_str += "..."
		else:
			if v < 0:
				n_spaces = len_features + 2 - len(k)
			else:
				n_spaces = len_features + 3 - len(k)
			for j in range(n_spaces):
				report_str += " "

		report_str += f"{v:.4f}\n"

	for j in range(max_len):
		report_str += "-"
	report_str += '\n'

	return report_str