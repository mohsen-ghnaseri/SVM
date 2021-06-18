# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 17:19:10 2021

@author: mohsen-pc
"""
#%% load library
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import classification_report
import itertools
from sklearn.metrics import confusion_matrix
#%% load data petal width  & petal length 
irisdata = datasets.load_iris()
X = irisdata.data[:, 2:4]
y = irisdata.target
  
#%% C is the SVM regularization parameter
C = 1.0 
#%%Create an Instance of SVM and Fit out the data
svc = svm.SVC(kernel='linear', C=C).fit(X, y)
# LinearSVC (linear kernel)
lin_svc = svm.LinearSVC(C=C).fit(X, y)
# SVC with RBF kernel
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
# SVC with polynomial (degree 3) kernel
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
#%% create a mesh to plot
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = 0.01
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
#%% part b
# title for the plots
titles = ['SVC with linear kernel',
	  'LinearSVC (linear kernel)',
	  'SVC with RBF kernel',
	  'SVC with polynomial kernel']


for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    plt.figure(figsize=(12,8)) 

    # Put the result into a color plot
    Za = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Za, cmap=plt.cm.Greens, alpha=0.8)
    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.brg)
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.xlim(xx.min(), xx.max())
    plt.title(titles[i])

plt.show()

#%% confusion matrix
font = {'family': 'serif','color': 'navy','weight': 'normal','size': 24}
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.figure(figsize = (7,7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(3)
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label',fontdict=font)
    plt.xlabel('Predicted label',fontdict=font)
#%% plot cofusion matrix and classification report
class_names = irisdata.target_names

for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].

    REP = clf.predict(X)
    conf_matrix1 =  confusion_matrix(y, REP)
    plot_confusion_matrix(conf_matrix1, class_names)
    plt.title(titles[i],fontdict=font)
    print(titles[i])
    print(classification_report(y,REP))

plt.show()
#%%
