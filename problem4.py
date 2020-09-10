import numpy as np
import matplotlib as plt
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import random
from sklearn.neighbors import KNeighborsClassifier

data = np.loadtxt("Datos misteriosos.txt")

# print(len(data[0]))
# print(data[0][0])
# print(data[661][0])
# print(data.shape)
cantidadDeDatos = data.shape[0]
cantidadDeParametros = data.shape[1]-1

#Etiquetas
y = data[:,0]
x = []

print(y)
# print(data[0][1::])
for i in range(0, cantidadDeDatos):
  x.append(data[i][1:])

x = np.array(x)


clf = KNeighborsClassifier()


for i in range(2,11):
  print("=================== Medición con K = ",i)
  kf = KFold(n_splits = i, shuffle=True)


  accp = 0

  for train_index, test_index in kf.split(x):
    x_train = x[train_index, :]
    y_train = y[train_index]
    clf.fit(x_train, y_train)

    x_test = x[test_index, :]
    y_test = y[test_index]

    y_pred = clf.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("acc = ", (cm[0,0]+cm[1,1])/len(y_test))
    accp += (cm[0,0]+cm[1,1])/len(y_test)

  print("Average acurracy is ", accp/i)