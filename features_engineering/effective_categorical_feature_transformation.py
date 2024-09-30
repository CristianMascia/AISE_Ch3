from sklearn import preprocessing

data = [['Bleach'], ['Cereal'], ['Toiled Roll']]

onehot_enc = preprocessing.OneHotEncoder()
onehot_enc.fit(data)

for c in data:
    print("{}: {}".format(c, onehot_enc.transform([c]).toarray()))
