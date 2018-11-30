import numpy as np
import copy
import xlrd
import mulsvmpso
import sys
from config import cfg
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import metrics


def loadData(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = np.zeros((numberOfLines, 60))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split(',')
        returnMat[index, :] = (listFromLine[0:60])
        classLabelVector.append(float(listFromLine[-1]))
        # if classLabelVector[index] == 2:
        #     classLabelVector[index] = -1
        index += 1
    X_scale = preprocessing.scale(returnMat)
    return X_scale, classLabelVector

def loadxcel(filename, kind):
    wb = xlrd.open_workbook(filename)
    sheet = wb.sheet_by_index(0)
    res = np.zeros((sheet.nrows, sheet.ncols))
    for i in range(sheet.nrows):
        res[i] = sheet.row_values(i)
    if kind == 'data':
        pass
    else:
        res = np.ravel(res.T)
    return res

def loadfold(i):
    path = '../Data/MCIcvsNC_fold'
    trainx_filename =path +str(i)+'_KeyFeatures_train_x.xlsx'
    testx_filename = path +str(i)+'_KeyFeatures_test_x.xlsx'
    trainy_filename = path +str(i)+'_train_y.xlsx'
    testy_filename = path +str(i)+'_test_y.xlsx'
    train_x = loadxcel(trainx_filename, 'data')
    test_x = loadxcel(testx_filename, 'data')
    train_y = loadxcel(trainy_filename, 'label')
    test_y = loadxcel(testy_filename, 'label')
    data = np.zeros((train_x.shape[0] + test_x.shape[0], train_x.shape[1]))
    label = np.zeros(train_y.shape[0] + test_y.shape[0])
    data[:train_x.shape[0]] = train_x
    data[train_x.shape[0]:] = test_x
    data = preprocessing.scale(data)
    label[:train_y.shape[0]] = train_y
    label[train_y.shape[0]:] = test_y
    return data, label, train_x.shape[0]

def nearestPD(A):
    B = (A + A.T) / 2
    spacing = np.spacing(np.linalg.norm(A))
    fail = 1
    while(fail):
        try:
            _, s, V = np.linalg.svd(B)
            fail = 0
        except:
            B = np.eye(A.shape[0]) * spacing + B
    H = np.dot(V.T, np.dot(np.abs(np.diag(s)), V)) #V输出
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3


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

def fitness_function(kernel, label, train_percent = None):
    m, n = np.shape(kernel)
    train_kernel = kernel.copy()
    train_label = label.copy()
    # train_kernel = kernel[0:int(cfg.test_percent * m), 0:int(cfg.test_percent * m)]
    # train_label = label[0:int(cfg.test_percent * m)]
    yy = np.mat(train_label).T * np.mat(train_label)
    molecular = np.sum(np.multiply(train_kernel, yy)) #分子
    # print("dididid")
    Denominator = np.sqrt(np.sum(np.multiply(train_kernel, train_kernel))) #分母
    # print("casssss")
    alignment_value = molecular/(m*Denominator)
    # print("ddfadsf")
    return alignment_value,0,0,1
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


def computer_acc(kernel, label, train_percent = None):
    m, n = np.shape(kernel)
    if train_percent:
        verify = train_percent
    else:
        verify = int(cfg.train_percent * m)
    test_assemble = int(verify * cfg.test_percent)
    train_kernel = kernel[0:test_assemble, 0:test_assemble]
    train_label = label[0:test_assemble]
    test_kernel = kernel[test_assemble:verify, 0:test_assemble]
    test_label = label[test_assemble:verify]
    # train_label = np.reshape(train_label, (len(train_label, )))
    clf = SVC(C=cfg.C, kernel='precomputed')
    clf.fit(train_kernel, train_label)
    predict = clf.predict(train_kernel)
    traccuray = accuracy_score(train_label, predict)
    predict = clf.predict(test_kernel)
    accuracy = accuracy_score(test_label, predict)
    # clf.fit(kernel[:verify,:verify], label[:verify])
    # predict = clf.predict(kernel[:verify,:verify])
    # traccuray = accuracy_score(label[:verify], predict)
    fitness = fitness_function(train_kernel, train_label)[0]
    support = len(clf.support_)
    percent = support / test_assemble
    return f_target(traccuray, accuracy, fitness, 1 - percent),percent, traccuray, accuracy


def computer_acc_test(kernel, label, train_percent = None):
    m, n = np.shape(kernel)
    verify = int(cfg.train_percent * m)
    if train_percent:
        test_assemble = train_percent
    else:
        test_assemble = int(verify * cfg.test_percent)
    train_kernel = kernel[0:test_assemble, 0:test_assemble]
    train_label = label[0:test_assemble]
    test_kernel = kernel[verify:, 0:test_assemble]
    test_label = label[verify:]
    # train_label = np.reshape(train_label, (len(train_label, )))
    clf = SVC(C=cfg.C, kernel='precomputed')
    clf.fit(train_kernel, train_label)
    predict = clf.predict(test_kernel)
    # predict = clf.decision_function(test_kernel)
    # auc = metrics.roc_auc_score(test_label, predict)
    return accuracy_score(test_label, predict),clf.support_,clf.dual_coef_

def computer_tr(kernel, label):
    m, n = np.shape(kernel)
    tkernel = kernel[0:int(cfg.test_percent * m), 0:int(cfg.test_percent * m)]
    klabel = label[0:int(cfg.test_percent * m)]
    clf = SVC(C=cfg.C, kernel='precomputed')
    clf.fit(tkernel, klabel)
    predict = clf.predict(tkernel)
    return accuracy_score(klabel, predict), clf.support_

def auc_measure(kernel, label,train_percent = None):
    m, n = np.shape(kernel)
    verify = int(cfg.train_percent * m)
    if train_percent:
        test_assemble = train_percent
    else:
        test_assemble = int(verify * cfg.test_percent)
    tkernel = kernel[0:test_assemble, 0:test_assemble]
    klabel = label[0:test_assemble]
    trkernel = kernel[test_assemble:verify, 0:test_assemble]
    trlabel = label[test_assemble:verify]
    allkernel = kernel[verify:, 0:test_assemble]
    alllabel = label[verify:]
    clf = SVC(C = cfg. C, kernel = 'precomputed')
    clf.fit(tkernel, klabel)
    predict = clf.decision_function(tkernel)
    auc = metrics.roc_auc_score(klabel, predict)
    predict = clf.decision_function(trkernel)
    verify_auc = metrics.roc_auc_score(trlabel, predict)
    predict = clf.decision_function(allkernel)
    t_auc = metrics.roc_auc_score(alllabel, predict)
    return f_target(auc, verify_auc, 0,0),auc, verify_auc, t_auc

def f_measure(kernel, label):
    m, n = np.shape(kernel)
    train_kernel = kernel[0:int(cfg.test_percent * m), 0:int(cfg.test_percent * m)]
    train_label = label[0:int(cfg.test_percent * m)]
    clf = SVC(C = cfg.C, kernel='precomputed')
    clf.fit(train_kernel, train_label)
    predict = clf.predict(train_kernel)
    tp, fp, fn, fm = 0, 0, 0, 0
    for i in range(len(train_label)):
        if predict[i] > 0 and train_label[i] > 0:
            tp += 1
        if predict[i] > 0 and train_label[i] < 0:
            fp +=1
        if predict[i] < 0 and train_label[i] > 0:
            fn +=1
        if predict[i] < 0 and train_label[i] < 0:
            fm +=1
    p = tp / (tp + fp)
    q = tp / (tp + fn)
    return (2 * p * q) / (p + q)

def get_kernel_yy(kernel, m, label, rate):
    ktrain = kernel[0:int(rate * m), 0:int(rate * m)]
    klabel = label[0:int(rate * m)]
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

def PR_SORT(decision, label):
    p_r = list()
    for i in range(len(decision)):
        p_r.append([decision[i], int(label[i])])
    p_r.sort(key = lambda x:x[0], reverse = True)
    res = []
    for i in p_r:
        res.append(i[1])
    return res

def pr_area(decision, label):
    label = PR_SORT(decision, label)
    pos = label.count(-1)
    neg = label.count(1)
    tp = 0
    fn = neg
    fp = 0
    area = 0
    left_p = 0
    left_r = 0
    for i, l in enumerate(label):
        if l == 1 :
            tp += 1
            fn -= 1
        elif l == -1:
            fp += 1
        else:
            raise ('erro')
        r = tp / (tp + fn)
        p = tp / (tp + fp)
        if left_p and left_r:
            area += (left_p + p) * (r - left_r) / 2
        left_r = r
        left_p = p
    return area


def PR_measure(kernel, label):
    m, n = np.shape(kernel)
    verify = int(cfg.train_percent * m)
    test_assemble = int(verify * cfg.test_percent)
    tkernel = kernel[0:test_assemble, 0:test_assemble]
    klabel = label[0:test_assemble]
    trkernel = kernel[test_assemble:verify, 0:test_assemble]
    trlabel = label[test_assemble:verify]
    allkernel = kernel[verify:, 0:test_assemble]
    alllabel = label[verify:]
    clf = SVC(C = cfg. C, kernel = 'precomputed')
    clf.fit(tkernel, klabel)
    decision = clf.decision_function(tkernel)
    precision, recall, _ = metrics.precision_recall_curve(klabel, decision)
    tPR = metrics.auc(recall, precision)
    decision = clf.decision_function(trkernel)
    precision, recall, _ = metrics.precision_recall_curve(trlabel, decision)
    PR = metrics.auc(recall, precision)
    # PR = pr_area(decision, trlabel)
    all_decision = clf.decision_function(allkernel)
    precision, recall, _ = metrics.precision_recall_curve(alllabel, all_decision)
    all_PR = metrics.auc(recall, precision)
    # all_PR = pr_area(all_decision, alllabel)
    return f_target(tPR, PR), all_PR, PR, tPR

def f_target(A, B, C, D):
    # print("pre_acc: %f" %(mulsvmpso.pre_acc))
    res = 0.1 * A + 0.3 * B +  0.1 * C/0.1 + 0.4 /(1 + np.e **(D - 1)) - 0.1 * np.abs(A - B)
    # spacing = sys.float_info.min
    # if B >= 0.8:
    #     res = 0.1 * A + 0.2 * B + 0.3 / (1 + np.e ** (- (spacing + np.abs(A - B)))) + 0.3 * D + 0.1 * C
    # else:
    #     res = 0.1 * A + 0.4 * B + 0.3/(1 + np.e ** (- (spacing + np.abs(A - B)))) + 0.1 * D + 0.1 * C
    return res

def decision_function(kernel, alpha, b):
    decision = alpha * kernel.T + b
    print(decision)
    return decision

def checkerror(y_true, predict, f_handle,decision = None):
    index = 0
    for i, j in zip(y_true, predict):
        if i != j:
            print(index,end = " ")
            f_handle.write(str(index) + " ")
        index += 1
    f_handle.write("\n\n")
    print("\n")

def predict(data,label,train_percent):
    clf = SVC(C = cfg.C, kernel = 'rbf',gamma = 'auto')
    clf.fit(data[:train_percent],label[:train_percent])
    return clf.predict(data[train_percent:])