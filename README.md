# FIDAP (**F**eature **I**mportance by **DA**ta **P**ermutation)

## Brief Introduction:
**FIDAP** (**F**eature **I**mportance by **DA**ta **P**ermutation) is a model free feature importance analysis module that can be used for evaluating the importance of various features of an ML model. This method uses an appropriate metric (i.e., R2 for regression models, accuracy score for classification models, and silhouette score for clustering models) to evaluate the ML model under investigation while shuffling the data for each feature. If a feature is highly important, then shuffling the data pertaining to that feature would significantly impact the evaluation metric. Conversely, if a feature is of low importance, then shuffling the data related to that feature would not significantly affect the evaluation metric. The idea is inspired by the feature importance analysis in the Random Forest paper.

Figure 1 presents the flowchart of the FIDAP method.

![](https://github.com/vd1371/FIDAP/blob/main/resources/FlowChart.png)

_Figure 1: Flowchart of the FIDAP method_
#
# How to Use:
FIDAP can be directly cloned from the girhub repository
```
git clone https://github.com/vd1371/FIDAP
```
And then move "FIDAP" folder to your root directory.

Since it is still early to make this library a python package, we decided to suffice with the github repository.

#
# Examples:
Various ML models, as mentioned in Table 1, are used as examples to showcase the capability of the FIDAP method in feature importance analysis. However, due to the modularity and flexibility of the developed model, it can be expanded to be used for other types of ML models.

_Table 1: Some of the ML models that are used as examples_
| Classification | Regression | Clustering |
| ------ | ------ | ------ | ------ |
| Random Forest | Linear Regression | Mean Shift |
| Support Vector | Support Vector | K Means | 
| Multi-layer Perceptron | Deep Neural Net | | 
| Decision Tree | Decision Tree | | 
| Extra Trees | Extra Trees | | 
| Radius Neighbors | Naïve Bayes | | 
| Passive Aggressive | Passive Aggressive | | 
| Gradient Boosting | Gradient Boosting | | 
| Cat Boost | XG Boost | | 
| K Nearest Neighbors | | | 
| Logistic Regression | | | 
#

# Results and Insights:
Using the FIDAP method for feature importance analysis, one can sort the feature of their ML model in order of importance. For example, Figure 2 represents the result of feature importance analysis for a RF model on the Iris dataset ![](https://archive.ics.uci.edu/ml/datasets/iris)(https://archive.ics.uci.edu/ml/datasets/iris). As it can be seen, the most critical feature of this model is the “petal length.” In other words, not incorporating this feature in the associated ML model would result in a significant model error. Conversely, not including the “sepal width” feature, which is the least important feature in the ML model, would lead to a minor error.

![](https://github.com/vd1371/FIDAP/blob/main/resources/Box_Output.jpg)

_Figure 2: Feature importance analysis for an RF model on the Iris dataset_
#
Alongside developing the feature importance analysis model for regression, classification, and clustering models, a separate model for pixel importance analysis of images is developed. Although this module is at its embryonic stages, this model scores pixels according to their impact on the accuracy of the model. More specifically, if eliminating a pixel causes a considerable increase in model error, that pixel is highly important and vice versa. Figure 3 depicts the pixel importance analysis on digit 2 of the MNIST dataset. The whiter the dots in the right photo, the more critical they are. Black dots are of no importance, and eliminating them from the original image would have no impact on the result.

![](https://github.com/vd1371/FIDAP/blob/main/resources/Pixel-Importance.jpg)

_Figure 3: Pixel importance analysis for a handwritten digit recognition model on the MNIST dataset_
#


## Example
```python
from FIDAP import FeatureImportanceAnalyzer
>>> fidap = FeatureImportanceAnalyzer(Model, X_test, y_test,
...                                    n_feature_combination = 3,
...                                    n_simulations = 10)
>>> fidap.run()
>>> print (fidap)
```
```
Feature                                                                  FIDAP  
--------------------------------------------------------------------------------
F(0,)-sepal length (cm)                                                 -0.0200
F(1,)-sepal width (cm)                                                  -0.0089
F(2,)-petal length (cm)                                                  0.1067
F(3,)-petal width (cm)                                                   0.3000
F(0, 1)-sepal length (cm)-sepal width (cm)                              -0.0378
F(0, 2)-sepal length (cm)-petal length (cm)                              0.1711
F(0, 3)-sepal length (cm)-petal width (cm)                               0.3489
F(1, 2)-sepal width (cm)-petal length (cm)                               0.0911
F(1, 3)-sepal width (cm)-petal width (cm)                                0.3067
F(2, 3)-petal length (cm)-petal width (cm)                               0.5489
F(0, 1, 2)-sepal length (cm)-sepal width (cm)-petal length (cm)          0.1311
F(0, 1, 3)-sepal length (cm)-sepal width (cm)-petal width (cm)           0.4000
F(0, 2, 3)-sepal length (cm)-petal length (cm)-petal width (cm)          0.6156
F(1, 2, 3)-sepal width (cm)-petal length (cm)-petal width (cm)           0.5689
--------------------------------------------------------------------------------

```


---
## Documentation
```
Feature Importance by DAta Permutation

	The idea is based on the data permutation proposed by Dr. Breiman in
	his famous Random Forest paper.

	Parameters
	----------
	model: callable
		The prediction or clustering model

	X: pandas.Dataframe, list of lists, numpy 2D array
		The input variables as pandas dataframe

	Y: pandas.Dataframe, pandas.Seris, list, numpy array
		The output variable. It must be one dimensional. If one-hot encoding
		is used for the multiclass classification as the output variable,
		only one of the outputs must be passed.

	features: list, default = None
		length of passed features must be equal to number of input variables
		if None:
			the X.columns will be used as features if X is a pd.DataFrame
			"X{i}" will be used, e.g., X0, X1, X2...

	metric_fn: str or callable, default = None
		The metric_fn needs to be ascending
		if None:
			the default metric function will be used for the metric_fn
				accuracy for classification
				R2 for regression
				silhouette for clustering
		if callable:
			the passed function will be used. The 
	
		if str:
			sklearn.metrics.get_scorer will be called
			more info: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
	
	n_simulations: int, default = 10
		the number of random permutation for each feature
		min (n_simulations) = 10

	pred_fn: str, default = "predict"
		The prediction method of the passed model. The default value is 
		"predict"

	direct: str, default = "."
		the directory for saving the summary and the figures. Default:
		root directory

	verbose: {True, False}, default = False
		verbosity

	n_feature_combination: int, default = 1
		number of features to be permutated for analysis
		Default is 1 and max is len(features)

	output_fig_format: str, default = 'tif'
		the format of the figure to be saved, default is ".tif"

	Attributes
	----------
	features_importance: {features: average of feature importance vals}
		the average of the feature importance for features

	features_importance_instances: {feature: [instances of feature importances]}
		the instances of features importance for features

	Methods:
	----------
	get():
		returns feature_importance

	boxplot():
		saves the boxplot figure into the directory passed by user

	summary():
		saves the report into the directory passed by user

	run():
		boxplot() + summary()

	__str__():
		return the string format of features importances
```

---
## Requirements
 1. SciPy
 2. Pandas
 3. Numpy
 4. sklearn
 5. keras
 6. xgboost
 7. catboost
---
## If you are using this repository

#### Please cite the below reference(s)
L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.

---
## License
© Vahid Asghari, Amin Baratian 2022. Licensed under the General Public License v3.0 (GPLv3).

---


