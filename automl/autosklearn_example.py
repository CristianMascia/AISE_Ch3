import numpy as np
import sklearn.datasets
import sklearn.metrics
import autosklearn.classification
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=60,
    per_run_time_limit=29,
    memory_limit=1024 * 10
)

wine_data = load_wine()


feature_names = wine_data.feature_names
X, y = wine_data.data, wine_data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)


automl.fit(X_train, y_train, dataset_name='wine')

print(automl.show_models())
print(automl.sprint_statistics())
predictions = automl.predict(X_test)
sklearn.metrics.accuracy_score(y_test, predictions)