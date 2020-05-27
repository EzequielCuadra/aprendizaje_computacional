from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from time import time

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

np.set_printoptions(formatter={'float': lambda x: '%.3f' % x})

# ===============================================================================================
# Prepare data
# ===============================================================================================

# Load dataset
inputData = np.loadtxt(open('DATASET.csv'), delimiter=",", skiprows=1, dtype='float')
# Atributes except 'Class' column
X = inputData[:, :-1]
# Class labels
y = inputData[:, -1]

# Standarize data
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

kf = KFold(n_splits=2)

# Split in train and test set
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# ===============================================================================================
# KNN
# ===============================================================================================

# Moment at we start building the model
time_ini_knn = time()
# Apply 5NN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
# Moment at we end building the model
time_fin_knn = time()
# Time spent in build the model
t_knn = time_fin_knn - time_ini_knn
print('==== Time for building kNN ==== ')
print(t_knn)
# Predictions
y_pred_knn = knn.predict(X_test)

# Time to classify:
# Moment at we start
time_ini_knn = time()
# classify
scores_knn = cross_val_score(knn, X, y, cv=2, scoring='accuracy')
# Moment at we end
time_fin_knn = time()
# Total time spent in classify
t_knn = time_fin_knn - time_ini_knn
print('==== Classify Time ====')
print(t_knn)

# Print Results
trueEstimates = np.count_nonzero((y_pred_knn - y_test) == 0)
totalEstimates = len(y_pred_knn)
print('==== WRONG PREDICTED ==== ')
print(totalEstimates - trueEstimates)
print('==== CORRECT PREDICTED ==== ')
print(trueEstimates)
print('==== CONFUSION MATRIX KNN ==== ')
print(confusion_matrix(y_test, y_pred_knn))
print('==== PRECISION ====')
print(precision_score(y_test, y_pred_knn) * 100, '%')
print('==== ACCURACY ==== ')
print(accuracy_score(y_test, y_pred_knn) * 100, '%')
print('==== RECALL ==== ')
print(recall_score(y_test, y_pred_knn, average='binary') * 100, '%')

# ===============================================================================================
# SVM
# ===============================================================================================

# Moment at we start building the model
time_ini_svm = time()
svm = SVC(gamma='auto')
svm.fit(X_train, y_train)
# Moment at we end building the model
time_fin_svm = time()
# Time spent in build the model
t_svm = time_fin_svm - time_ini_svm
print('==== Time for building SVM ==== ')
print(t_svm)

# Prediction
y_pred_svm = svm.predict(X_test)

# Time to classify:
# Moment at we start
time_ini_svm = time()
# classify
scores_svm = cross_val_score(svm, X, y, cv=2, scoring='accuracy')
# Moment at we end
time_fin_svm = time()
# Total time spent in classify
t_svm = time_fin_svm - time_ini_svm
print('==== Classify Time ====')
print(t_svm)

# Print Results
trueEstimates = np.count_nonzero((y_pred_svm - y_test) == 0)
totalEstimates = len(y_pred_svm)
print('==== WRONG PREDICTED ==== ')
print(totalEstimates - trueEstimates)
print('==== CORRECT PREDICTED ==== ')
print(trueEstimates)
print('==== CONFUSION MATRIX SVM ==== ')
print(confusion_matrix(y_test, y_pred_svm))
print('==== PRECISION ====')
print(precision_score(y_test, y_pred_svm) * 100, '%')
print('==== ACCURACY ==== ')
print(accuracy_score(y_test, y_pred_svm) * 100, '%')
print('==== RECALL ==== ')
print(recall_score(y_test, y_pred_svm, average='binary') * 100, '%')

# ===============================================================================================
# NEURAL NETWORK
# ===============================================================================================

# Moment at we start building the model
time_ini_rn = time()
mlp = MLPClassifier(solver='adam', activation='relu', hidden_layer_sizes=(6, 4), max_iter=1000)
mlp.fit(X_train, y_train)
# Moment at we end building the model
time_fin_rn = time()
# Time spent in build the model
t_rn = time_fin_rn - time_ini_rn
print('==== Tiempo de construccion de la RED NEURONAL ==== ')
print(t_rn)

# Predictions
y_pred_rn = mlp.predict(X_test)

# Time to classify:
# Moment at we start
time_ini_rn = time()
# classify
scores_rn = cross_val_score(mlp, X, y, cv=2, scoring='accuracy')
# Moment at we end
time_fin_rn = time()
# Total time spent in classify
t_rn = time_fin_rn - time_ini_rn
print('==== Classify Time ====')
print(t_rn)

# Print Results
trueEstimates = np.count_nonzero((y_pred_rn - y_test) == 0)
totalEstimates = len(y_pred_rn)
print('==== WRONG PREDICTED ==== ')
print(totalEstimates - trueEstimates)
print('==== CORRECT PREDICTED ==== ')
print(trueEstimates)
print('==== CONFUSION MATRIX NN ==== ')
print(confusion_matrix(y_test, y_pred_rn))
print('==== PRECISION ====')
print(precision_score(y_test, y_pred_rn) * 100, '%')
print('==== ACCURACY ==== ')
print(accuracy_score(y_test, y_pred_rn) * 100, '%')
print('==== RECALL ==== ')
print(recall_score(y_test, y_pred_rn, average='binary') * 100, '%')
