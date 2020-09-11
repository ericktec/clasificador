import matplotlib as plt
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

wine = datasets.load_wine()

x = wine.data
y = wine.target
features = wine.feature_names
n_features = len(features)
print(features, n_features)


plt.scatter(x[:,1], x[:,2], c = y, cmap = plt.cm.Set1, edgecolor = 'k')
plt.xlabel(features[1])
plt.ylabel(features[2])
plt.show()

bestAverage = 0
bestPrediction = ""


print("==============================LINEAL")

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

  print("acc = ", (cm[0,0]+cm[1,1]+cm[2,2])/len(y_test))
  accp += (cm[0,0]+cm[1,1]+cm[2,2])/len(y_test)


bestAverage = accp/5
bestPrediction = "Linear"
print("Average accuracy is ", accp/5)


print("==============================RBF")


rbf = svm.SVC(kernel='rbf')

accp = 0



for train_index, test_index in kf.split(x):
  x_train = x[train_index, :]
  y_train = y[train_index]
  rbf.fit(x_train, y_train)

  x_test = x[test_index, :]
  y_test = y[test_index]

  y_pred = rbf.predict(x_test)
  cm = confusion_matrix(y_test, y_pred)
  print(cm)

  print("acc = ", (cm[0,0]+cm[1,1]+cm[2,2])/len(y_test))
  accp += (cm[0,0]+cm[1,1]+cm[2,2])/len(y_test)


if(accp/5 > bestAverage):
  bestAverage = accp/5
  bestPrediction = "RBF"


print("Average accuracy is ", accp/5)


print("==============================NEIGH")

neigh = KNeighborsClassifier(n_neighbors=3)

accp = 0

for train_index, test_index in kf.split(x):
  x_train = x[train_index, :]
  y_train = y[train_index]
  neigh.fit(x_train, y_train)

  x_test = x[test_index, :]
  y_test = y[test_index]

  y_pred = neigh.predict(x_test)
  cm = confusion_matrix(y_test, y_pred)
  print(cm)

  print("acc = ", (cm[0,0]+cm[1,1]+cm[2,2])/len(y_test))
  accp += (cm[0,0]+cm[1,1]+cm[2,2])/len(y_test)


if(accp/5 > bestAverage):
  bestAverage = accp/5
  bestPrediction = "NEIGH"

print("Average accuracy is ", accp/5)



print("==============================DTC")
dtc = DecisionTreeClassifier(random_state=0)

accp = 0

for train_index, test_index in kf.split(x):
  x_train = x[train_index, :]
  y_train = y[train_index]
  dtc.fit(x_train, y_train)

  x_test = x[test_index, :]
  y_test = y[test_index]

  y_pred = dtc.predict(x_test)
  cm = confusion_matrix(y_test, y_pred)
  print(cm)

  print("acc = ", (cm[0,0]+cm[1,1]+cm[2,2])/len(y_test))
  accp += (cm[0,0]+cm[1,1]+cm[2,2])/len(y_test)

if(accp/5 > bestAverage):
  bestAverage = accp/5
  bestPrediction = "DTC"


print("Average accuracy is ", accp/5)




print("The best classifier was ", bestPrediction)





