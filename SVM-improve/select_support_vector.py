from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, sigmoid_kernel,linear_kernel
from sklearn import svm
from config import cfg
import tensorflow as tf
import toolbox
from sklearn.metrics import accuracy_score

def support_vector(kernel, label):
    clf = svm.SVC(C=cfg.C, kernel='precomputed')
    clf.fit(kernel, label)
    predict = clf.predict(kernel)
    accuracy = accuracy_score(label, predict)
    print(accuracy)
    print(clf.support_.shape)
    return clf.support_.copy()

def solve(data, label):
    kernel = rbf_kernel(data)
    rbf_support = support_vector(kernel, label)
    kernel = polynomial_kernel(data)
    pol_support = support_vector(kernel, label)
    kernel = sigmoid_kernel(data)
    sig_support = support_vector(kernel, label)
    kernel = linear_kernel(data)
    lin_support = support_vector(kernel, label)
    intersection = set(rbf_support) | set(pol_support) | set(sig_support) | set(lin_support)
    return intersection


