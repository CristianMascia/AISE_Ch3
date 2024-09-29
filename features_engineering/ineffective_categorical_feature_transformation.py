from sklearn import preprocessing

data = [['Bleach'], ['Cereal'], ['Toiled Roll']]

ordinal_enc = preprocessing.OrdinalEncoder()
ordinal_enc.fit(data)
print(ordinal_enc.transform(data))
