import numpy as np
import matplotlib as plt
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import random


def getRandom(data, clases, porcentaje):
  x = []
  y = []
  for i in range(0, int(len(data)*porcentaje)):
    r = random.randint(0, len(data)-1)
    x.append(data[r])
    y.append(clases[r])
  
  x = np.array(x)
  y = np.array(y)

  return x, y



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
print(type(x))


print("==============================LINEAL 100%")

kf = KFold(n_splits=5, shuffle=True)

clf = svm.SVC(kernel='linear')

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



print("Average accuracy is ", accp/5)

for i in range (10, 100, 10):
  porcentaje = i/100
  
  x_porcentaje, y_porcentaje = getRandom(x,y,porcentaje)


  print("==============================LINEAL ", i,"%" )

  accp = 0
  getRandom(x, y, porcentaje)
  for train_index, test_index in kf.split(x_porcentaje):
    x_train = x_porcentaje[train_index, :]
    y_train = y_porcentaje[train_index]
    clf.fit(x_train, y_train)

    x_test = x_porcentaje[test_index, :]
    y_test = y_porcentaje[test_index]

    y_pred = clf.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("acc = ", (cm[0,0]+cm[1,1])/len(y_test))
    accp += (cm[0,0]+cm[1,1])/len(y_test)



  print("Average accuracy is ", accp/5)

