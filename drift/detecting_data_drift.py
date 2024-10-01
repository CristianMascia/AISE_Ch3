from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from alibi_detect.cd import TabularDrift

wine_data = load_wine()
feature_names = wine_data.feature_names
X, y = wine_data.data, wine_data.target
X_ref, X_test, y_ref, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# features drift - original data
cd = TabularDrift(x_ref = X_ref, p_val = 0.05)
preds = cd.predict(X_test)
labels = ['No', 'Yes']
print('Drift: {}'.format(labels[preds['data']['is_drift']]))
# features drift - modified data
X_test_cal_error = 1.1 * X_test
preds = cd.predict(X_test_cal_error)
labels = ['No', 'Yes']
print('Drift: {}'.format(labels[preds['data']['is_drift']]))

# labels drift - original data
cd = TabularDrift(x_ref = y_ref, p_val = 0.05)
preds = cd.predict(y_test)
labels = ['No', 'Yes']
print('Drift: {}'.format(labels[preds['data']['is_drift']]))
# labels drift - modified data
y_test_cal_error = 1.1 * y_test
preds = cd.predict(y_test_cal_error)
labels = ['No', 'Yes']
print('Drift: {}'.format(labels[preds['data']['is_drift']]))