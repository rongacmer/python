from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
import svmtest
import random
from sklearn.datasets.samples_generator import make_blobs

def plot_svc_decision_function(model, ax = None, plot_support = True):
    if ax is None:
        ax = plt.gca()  #get current axes
    xlim = ax.get_xlim()  #获取X坐标取值范围
    ylim = ax.get_ylim() #获取Y坐标取值范围

    x = np.linspace(xlim[0], xlim[1], 30) #创建等差数列
    y = np.linspace(ylim[0], ylim[1], 30)

    Y,X = np.meshgrid(y,x) #生成网格图
    xy = np.vstack([X.ravel(), Y.ravel()]).T #ravel():数组降维 vstack():向量合并,T矩阵转置
    P = svmtest.dis_function(model,xy).reshape(X.shape)
    # P = model.decision_function(xy).reshape(X.shape) #函数距离
    ax.contour(X, Y, P, colors='k',levels = [-1,0,1], alpha=0.5, linestyles = ['--', '-', '--']) #等高线,高度为-1，0，1的等高线线
    # print(type(model.support_vector.A))
    if plot_support:
        ax.scatter(model.support_vector.A[:,0],
                   model.support_vector.A[:,1],
                   s=300, linewidth=1, color="b", facecolors='none')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.show()




def main():
    X,y = make_blobs(n_samples=20, centers= 2, cluster_std = 0.6, random_state = 10)
    for i in range(len(y)):
        if(y[i] == 0):
            y[i] = -1
    oS = svmtest.SMO(X, y, 1,0.000001,10000)
    # print(type(model.decision_function(X)))
    # model = svm.SVC(kernel='linear')
    # model.fit(X, y)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
    plot_svc_decision_function(oS)
    plt.show()

if __name__ == '__main__':
    main()