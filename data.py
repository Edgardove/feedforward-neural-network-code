import numpy, pandas
from sklearn import model_selection, preprocessing

#--------------------dataset 64 features handwritten digits--------------------
one_hot_encoding=numpy.array([
    [1,0,0,0,0,0,0,0,0,0],
    [0,1,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,0],
    [0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,1,0,0,0,0],
    [0,0,0,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0,1,0,0],
    [0,0,0,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,0,0,1]
], dtype=numpy.dtype('Float64'))

data_train=pandas.read_csv("data/optdigits.tra.csv") # classes 0 - 9
data_train["class"]=one_hot_encoding[data_train['class'].values].tolist()
X_train=numpy.array(data_train.drop(["class"], 1), dtype=numpy.dtype('Float64'))
X_train=preprocessing.scale(X_train)
# X_train=preprocessing.normalize(X_train, axis=0)
y_train=numpy.array(data_train["class"])

# print(X_train.min(), X_train.max())

data_test=pandas.read_csv("data/optdigits.tes.csv") # classes 0 - 9
data_test["class"]=one_hot_encoding[data_test['class'].values].tolist()
X_test=numpy.array(data_test.drop(["class"], 1), dtype=numpy.dtype('Float64'))
X_test=preprocessing.scale(X_test)
# X_test=preprocessing.normalize(X_test, axis=0)
y_test=numpy.array(data_test["class"])
#------------------------------------end------------------------------------