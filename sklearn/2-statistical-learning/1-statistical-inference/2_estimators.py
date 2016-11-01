# Estimators
# Any object that learns from data (has a .fit() method) is an estimator. This includes classifiers, regression or clustering algorithms, and transformers
estimator.fit(data)

# Their paramters can be set during instantiation or later.
estimator = Estimator(param1=1, params2=2)
estimator.param3 = 3

# Some params can also be estimated during data fitting. These are indicated with a trailing underscore
estimator.estimated_param_