import numpy as np
import random
class optStruct:
    def __init__(self, data, labels, C, toler):
        self.X = data
        self.labels = labels
        self.C = C
        self.tol = toler
        self.m = data.shape[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        self.support_vector = None
        self.ecache = -1.0*labels.T.copy()
        self.K = np.mat(np.zeros((self.m, self.m)))
        self.K = self.X * self.X.T

def Kernels(X, A):
    return X * A.T

def calckEk(oS ,k):
    fxk = float(np.multiply(oS.labels,oS.alphas).T * oS.K[:,k] + oS.b)
    Ek = fxk - oS.labels[k]
    return Ek

def updateEk(oS, k):
    Ek = calckEk(oS, k)
    oS.ecache[0,k] = Ek

# def selectj(i, oS, Ei):
#     minE = Ei
#     maxE = Ei
#     minindex = i
#     maxindex = i
#     for k in range(oS.ecache.shape[1]):
#         if oS.ecache[0,k] < minE:
#             minE = oS.ecache[0,k]
#             minindex = k
#         if oS.ecache[0, k] > maxE:
#             maxE = oS.ecache[0, k]
#             maxindex = k
#     if Ei > 0:
#         return minindex,minE
#     else:
#         return maxindex,maxE

def innerL(i, oS):
    Ei = calckEk(oS, i)
    if (0 < oS.alphas[i,0] < oS.C and (oS.labels[i, 0] * Ei > oS.tol or oS.labels[i, 0] * Ei < -oS.tol)) or \
            (oS.alphas[i,0] <= oS.tol and oS.labels[i,0] * Ei < -oS.tol) or \
            (oS.alphas[i, 0] == oS.C and oS.labels[i, 0] * Ei > oS.tol):
        for j in range(oS.m):
            Ej = oS.ecache[0, j]
            alphai = oS.alphas[i].copy()
            alphaj = oS.alphas[j].copy()
            if oS.labels[i] != oS.labels[j]:
                L = max(0,alphaj - alphai)
                H = min(oS.C, oS.C + alphaj - alphai)
            else:
                L = max(0, alphaj + alphai - oS.C)
                H = min(oS.C, alphai + alphaj)
            if L == H:
                continue
            ela = oS.K[i, i] + oS.K[j, j] - 2 * oS.K[i, j]
            if ela <= 0 :
                continue
            oS.alphas[j] = oS.alphas[j] + oS.labels[j] * (Ei - Ej) / ela
            if oS.alphas[j] > H:
                oS.alphas[j] = H
            elif oS.alphas[j] < L:
                oS.alphas[j] = L
            oS.alphas[i] = oS.alphas[i] + oS.labels[i] * oS.labels[j] * (alphaj - oS.alphas[j])
            if(abs(oS.alphas[j] - alphaj) < oS.tol):
                continue
            b1 = -Ei - oS.labels[i] * oS.K[i, i] * (oS.alphas[i] - alphai) - oS.labels[j] * oS.K[j, i] * (oS.alphas[j] - alphaj) + oS.b
            b2 = -Ej - oS.labels[i] * oS.K[i, j] * (oS.alphas[i] - alphai) - oS.labels[j] * oS.K[j, j] * (oS.alphas[j] - alphaj) + oS.b
            if 0 < oS.alphas[i] < oS.C:
                oS.b = b1
            elif 0 < oS.alphas[j] < oS.C:
                oS.b = b2
            else:
                oS.b = (b1 + b2) / 2
            for i in range(oS.m):
                updateEk(oS,i)
            return 1
    return 0

def SMO(data, labels, C, toler, maxIter):
    data = np.mat(data)
    labels = np.mat(labels)
    oS= optStruct(data, labels.T, C, toler)
    iter = 0
    entireSet = True
    cnt = 0
    while iter < maxIter and (cnt > 0 or entireSet) :
        cnt = 0
        if entireSet:
            for i in range(oS.m):
                cnt += innerL(i, oS)
            iter += 1
        else:
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                cnt += innerL(i, oS)
            iter += 1
        if entireSet:
            entireSet = False
        elif cnt == 0:
            entireSet = True
    print(iter)
    flag = True
    for i in range(oS.m):
        if 0 < oS.alphas[i, 0] < C:
            if flag == True:
                oS.support_vector = oS.X[i]
                flag = False
            else:
                oS.support_vector = np.vstack([oS.support_vector, oS.X[i]])
    return oS

def dis_function(oS, X):
    X = np.mat(X)
    m = X.shape[0]
    dis = np.mat(np.zeros((1,m)))
    print(oS.alphas)
    for i in range(m):
        Ker = Kernels(oS.X, X[i,:])
        dis[0,i] = float(np.multiply(oS.labels,oS.alphas).T * Ker + oS.b)
    return dis
