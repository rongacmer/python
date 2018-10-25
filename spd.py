import numpy as np
def nearestPD(A):
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)
    H = np.dot(V.T, np.dot(np.abs(np.diag(s)), V)) #V输出
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    print(A3)
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

def main():
    A = np.mat([[-1,0,0],[0,-1,0],[0,0,-1]])

    print(nearestPD(A))

if __name__ == '__main__':
    main()