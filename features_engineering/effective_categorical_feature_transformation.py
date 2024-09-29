from sklearn import preprocessing

data = [['Bleach'], ['Cereal'], ['Toiled Roll']]

onehot_enc = preprocessing.OneHotEncoder()
onehot_enc.fit(data)
print(onehot_enc.transform(data).toarray())
