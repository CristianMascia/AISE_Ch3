from sklearn import preprocessing

data = [['Bleach'], ['Cereal'], ['Toiled Roll']]

ordinal_enc = preprocessing.OrdinalEncoder()
ordinal_enc.fit(data)

for c in data:
    print("{}: {}".format(c, ordinal_enc.transform([c])[0]))
