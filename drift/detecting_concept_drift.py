from alibi_detect.cd import MMDDriftOnline
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"


wine_data = load_wine()


feature_names = wine_data.feature_names
X, y = wine_data.data, wine_data.target
X_ref, X_test, y_ref, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

xx = []
yy = []
dd = []
for i in range(len(X_test)):
    ert = 20 
    window_size = 10
    cd = MMDDriftOnline(X_ref, ert, window_size, backend='pytorch', n_bootstraps=2500)
    
    xx.append(i)
    yy.append(X_test[i])
    if cd.predict(X_test[i])['data']['is_drift']:
        dd.append(i)
plt.plot(xx, yy)
for d in dd:
    plt.axvline(x=d, color='orange', linewidth=2)
plt.show()