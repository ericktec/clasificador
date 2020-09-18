# ------------------------------------------------------------------------------------------------------------------
import numpy as np

from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

data = np.loadtxt("Datos misteriosos.txt")

cantidadDeDatos = data.shape[0]
cantidadDeParametros = data.shape[1]-1

# Etiquetas
y = data[:, 0]
y = y-1
x = []

print(y)
# print(data[0][1::])
for i in range(0, cantidadDeDatos):
    x.append(data[i][1:])

x = np.array(x)
print(type(x))

n_features = cantidadDeParametros


clf = Sequential()
clf.add(Dense(8, input_dim=n_features, activation='relu'))
clf.add(Dense(8, activation='relu'))
clf.add(Dense(1, activation='sigmoid'))

# Compile model
clf.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Evaluate model
kf = KFold(n_splits=5, shuffle=True)

acc = 0
recall = np.array([0., 0.])
for train_index, test_index in kf.split(x):

    # Training phase
    x_train = x[train_index, :]
    y_train = y[train_index]

    clf = Sequential()
    clf.add(Dense(8, input_dim=n_features, activation='relu'))
    clf.add(Dense(8, activation='relu'))
    clf.add(Dense(1, activation='sigmoid'))
    # clf.compile(loss='categorical_crossentropy', optimizer='adam') # For 2-class problems, use 
    clf.compile(loss='binary_crossentropy', optimizer='adam')
    clf.fit(x_train, y_train, epochs=150, batch_size=5, verbose=0)

    # Test phase
    x_test = x[test_index, :]
    y_test = y[test_index]
    # For 2-class problems, use (clf.predict(x_test) > 0.5).astype("int32")
    y_pred = (clf.predict(x_test) > 0.5).astype("int32")

    cm = confusion_matrix(y_test, y_pred)

    acc += (cm[0, 0]+cm[1, 1])/len(y_test)

    recall[0] += cm[0, 0]/(cm[0, 0] + cm[0, 1])
    recall[1] += cm[1, 1]/(cm[1, 0] + cm[1, 1])

print("Red neuronal")

acc = acc/5
print('ACC = ', acc)

recall = recall/5
print('RECALL = ', recall)


kf = KFold(n_splits=5, shuffle=True)

clf = svm.SVC(kernel='linear')

acc = 0
recall = np.array([0., 0.])

for train_index, test_index in kf.split(x):
    x_train = x[train_index, :]
    y_train = y[train_index]
    clf.fit(x_train, y_train)

    x_test = x[test_index, :]
    y_test = y[test_index]

    y_pred = clf.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)

    acc += (cm[0, 0]+cm[1, 1])/len(y_test)

    recall[0] += cm[0, 0]/(cm[0, 0] + cm[0, 1])
    recall[1] += cm[1, 1]/(cm[1, 0] + cm[1, 1])


print("SVG")

acc = acc/5
print('ACC = ', acc)

recall = recall/5
print('RECALL = ', recall)
