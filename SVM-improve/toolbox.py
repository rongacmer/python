import numpy as np
import copy
from config import cfg
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def loadData(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = np.zeros((numberOfLines, 18))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split(',')
        returnMat[index, :] = (listFromLine[0:18])
        classLabelVector.append(float(listFromLine[-1]))
        index += 1
    X_scale = preprocessing.scale(returnMat)
    return X_scale, classLabelVector


def nearestPD(A):
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)
    H = np.dot(V.T, np.dot(np.abs(np.diag(s)), V)) #V输出
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3
    spacing = np.spacing(np.linalg.norm(A))

    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1
    return A3


def isPD(B):
    try:
        _ = np.linalg.cholesky(B) #LLT分解
        return True
    except np.linalg.LinAlgError:
        return False

def get_train_acc(kernel, label):
    clf = SVC(C=cfg.C, kernel='precomputed')
    clf.fit(kernel, label)
    predict = clf.predict(kernel)
    return accuracy_score(label, predict)

def fitness_function(kernel, yy, label):
    m, n = np.shape(yy)
    molecular = np.sum(np.multiply(kernel, yy)) #分子
    # print("dididid")
    Denominator = np.sqrt(np.sum(np.multiply(kernel, kernel))) #分母
    # print("casssss")
    alignment_value = molecular/(m*Denominator)
    # print("ddfadsf")
    return alignment_value
    # return np.sum(kernel * yy) / np.linalg.norm(kernel) / np.linalg.norm(yy)


def computer_b(dataMat,label, kernel):
    ############################求解b的版本1#########################
    # support_index = clf.support_
    # support_vectors = dataMat[support_index]
    # support_vectors_label = label[support_index]
    # alphasy = clf.dual_coef_[0]
    # k_value = kernel[support_index[0]][support_index]
    # b = support_vectors_label[support_index[0]] - np.dot(alphasy, k_value)
    # print("b:", float(b))
    # return float(b), alphasy, support_vectors
    #############################求解b的版本2#########################
    clf = SVC(C=1.0, kernel='precomputed')
    clf.fit(kernel, label)
    support_index = clf.support_
    data = dataMat[support_index]
    label = label[support_index]
    SV_index = np.intersect1d(np.where(clf.dual_coef_[0] != -1)[0], np.where(clf.dual_coef_[0] != 1)[0])
    alphasy = clf.dual_coef_[0][SV_index]
    support_vectors = data[SV_index]
    support_vectors_label = label[SV_index]
    k_value = kernel[SV_index[0]][SV_index]
    b = support_vectors_label[support_index[0]] - np.dot(alphasy, k_value)
    print("b:", float(b))
    return float(b), alphasy, support_vectors


def computer_acc(kernel, label):
    m, n = np.shape(kernel)
    train_kernel = kernel[0:int(cfg.train_percent*m), 0:int(cfg.train_percent*m)]
    train_label = label[0:int(cfg.train_percent*m)]
    test_kernel = kernel[int(cfg.train_percent*m):, 0:int(cfg.train_percent*m)]
    test_label = label[int(cfg.train_percent*m):]
    train_label = np.reshape(train_label, (len(train_label, )))
    clf = SVC(C=cfg.C, kernel='precomputed')
    clf.fit(train_kernel, train_label)
    predict = clf.predict(test_kernel)
    return accuracy_score(test_label, predict)

def computer_acc_test(kernel, label):
    m, n = np.shape(kernel)
    train_kernel = kernel[0:int(cfg.test_percent*m), 0:int(cfg.test_percent*m)]
    train_label = label[0:int(cfg.test_percent*m)]
    test_kernel = kernel[int(cfg.test_percent*m):, 0:int(cfg.test_percent*m)]
    test_label = label[int(cfg.test_percent*m):]
    # train_label = np.reshape(train_label, (len(train_label, )))
    clf = SVC(C=cfg.C, kernel='precomputed')
    clf.fit(train_kernel, train_label)
    predict = clf.predict(test_kernel)
    return accuracy_score(test_label, predict)

def computer_tr(clf, kernel, label):
    m, n = np.shape(kernel)
    tkernel = kernel[0:int(cfg.train_percent * m), 0:int(cfg.train_percent * m)]
    klabel = label[0:int(cfg.train_percent * m)]
    predict = clf.predict(tkernel)
    return fitness_function(klabel, predict)


def get_kernel_yy(kernel, m, label):
    ktrain = kernel[0:int(cfg.train_percent * m), 0:int(cfg.train_percent * m)]
    klabel = label[0:int(cfg.train_percent * m)]
    # ktrain = np.mat(ktrain)
    # klabel = np.mat(klabel)
    train_yy = np.dot(klabel, klabel.T)
    return ktrain, train_yy


def generate_data(data, kernel):
    k = len(kernel)
    data_new = []
    for i in range(k):
        for j in range(k):
            data_new.append([float(data[i][0]), float(data[i][1]), float(data[i][2]), float(data[i][3]),
                            float(data[i][4]), float(data[i][5]), float(data[i][6]), float(data[i][7]),
                            float(data[i][8]), float(data[i][9]), float(data[i][10]), float(data[i][11]),
                            float(data[i][12]), float(data[i][13]), float(data[i][14]), float(data[i][15]),
                            float(data[i][16]), float(data[i][17]), float(data[i][18]), float(data[i][19]),
                            float(data[i][20]), float(data[i][21]), float(data[i][22]), float(data[i][23]),
                            float(data[i][24]), float(data[i][25]), float(data[i][26]), float(data[i][27]),
                            float(data[i][28]), float(data[i][29]), float(data[i][30]), float(data[i][31]),
                            float(data[i][32]), float(data[i][33]),
                            float(data[j][0]), float(data[j][1]), float(data[j][2]), float(data[j][3]),
                            float(data[j][4]), float(data[j][5]), float(data[j][6]), float(data[j][7]),
                            float(data[j][8]), float(data[j][9]), float(data[j][10]), float(data[j][11]),
                            float(data[j][12]), float(data[j][13]), float(data[j][14]), float(data[j][15]),
                            float(data[j][16]), float(data[j][17]), float(data[j][18]), float(data[j][19]),
                            float(data[j][20]), float(data[j][21]), float(data[j][22]), float(data[j][23]),
                            float(data[j][24]), float(data[j][25]), float(data[j][26]), float(data[j][27]),
                            float(data[j][28]), float(data[j][29]), float(data[j][30]), float(data[j][31]),
                            float(data[j][32]), float(data[j][33]), float(kernel[i][j])])
    data_BP = np.mat(data_new)
    normalized_data_BP_ex = data_BP[:, 0:-1]
    label_BP = data_BP[:, -1]
    return normalized_data_BP_ex, label_BP


