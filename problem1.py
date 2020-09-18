import matplotlib as plt
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix

iris = datasets.load_iris()


x = iris.data
y = iris.target
features = iris.feature_names
n_features = len(features)

plt.scatter(x[:,1], x[:,2], c = y, cmap = plt.cm.Set1, edgecolor = 'k')
plt.xlabel(features[1])
plt.ylabel(features[2])
plt.show()

kf = KFold(n_splits=10, shuffle=True)

clf = svm.SVC(kernel='linear')

accp = 0

recall1 = 0
recall2 = 0
recall3 = 0

recallAverage1 = 0
recallAverage2 = 0
recallAverage3 = 0

precision1 = 0
precision2 = 0
precision3 = 0

precisionAverage1 = 0
precisionAverage2 = 0
precisionAverage3 = 0





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

  recall1 = cm[0,0]/(cm[0,0] + cm[0,1] + cm[0,2])
  print("Recall1", recall1)
  recall2 = cm[1,1]/(cm[1,0] + cm[1,1] + cm[1,2])
  print("Recall2", recall2)
  recall3 = cm[2,2]/(cm[2,0] + cm[2,1] + cm[2,2])
  print("Recall3", recall3)

  recallAverage1 += recall1

  recallAverage2 += recall2

  recallAverage3 += recall3

  precision1 = cm[0,0]/(cm[0,0] + cm[1,0] + cm[2,0])
  print("Precision 1 ", precision1)
  precision2 = cm[1,1]/(cm[0,1] + cm[1,1] + cm[2,1])
  print("Precision 2 ", precision2)
  precision3 = cm[2,2]/(cm[0,2] + cm[1,2] + cm[2,2])
  print("Precision 3 ", precision3)

  precisionAverage1 += precision1
  precisionAverage2 += precision2
  precisionAverage3 += precision3



print("Average accuracy is ", accp/10)
print("Average recall class 1", recallAverage1/10)
print("Average recall class 2", recallAverage2/10)
print("Average recall class 3", recallAverage3/10)
print("Average precision class 1", precisionAverage1/10)
print("Average precision class 2", precisionAverage2/10)
print("Average precision class 3", precisionAverage3/10)


