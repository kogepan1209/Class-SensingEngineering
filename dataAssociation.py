import numpy as np

#Z~(k+1|k)
def calc_Ztidlek1(zk1, ak1, x_hat):
    temp = np.dot(ak1, x_hat)
    Ztidlek1 = zk1 - temp
    return Ztidlek1

#S(k+1)
def calc_Sk1(ak1, Pk, Rk1):
    Sk1 = np.dot(ak1, Pk)
    Sk1 = np.dot(Sk1, ak1.T)
    Sk1 = Sk1 + Rk1
    return Sk1

#W(k+1)
def calc_Wk1(Pk, ak1, Sk1):
    Wk1 = np.dot(Pk, ak1.T)
    Wk1 = np.dot(Wk1, np.linalg.inv(Sk1))
    return Wk1

#P(k|k)
def calc_Pk1(Pk, Wk1, Sk1):
    temp = np.dot(Wk1, Sk1)
    temp = np.dot(temp, Wk1.T)
    Pk1 = Pk - temp
    return Pk1

#P(k+1|k)
def calc_Pk2(Pk, Qk1):
    Pk1 = Pk - Qk1
    return Pk1

#x^(k+1|k+1)
def calc_x_hatk1(x_hat, Wk1, Ztidlek1):
    temp = np.dot(Wk1, Ztidlek1)
    x_hatk1 = x_hat + temp
    return x_hatk1

#E(z)
def calc_epsilon(Z, S):
    epsilon = np.dot(Z.T, np.linalg.inv(S))
    epsilon = np.dot(epsilon, Z)
    return epsilon

#観測値z(k+1)（array）
def get_zk1(Z, k):
    zk1 = [Z[(k+1)+15]]
    return np.array(zk1)

#a(k+1)（array）
def get_ak1(A, k):
    ak1 = [A[(k+1)+15]]
    return np.array(ak1)

#雑音の共分散R(k+1)（array）
def get_Rk1(R, k):
    Rk1 = [R[(k+1)+15][(k+1)+15]]
    return np.array(Rk1)

#カルマンフィルタ
def KalmanFiltering(x_hat_init, Pk_init, Qk_init, Z, A, R):
    #k=-16のときの初期値
    x_hat = x_hat_init
    print('推定値の初期値x_hat(-16) =', x_hat)
    Pk = Pk_init
    print('推定誤差共分散の初期値P(-16) =\n', Pk)
    Qk = Qk_init
    print('プラント雑音共分散の初期値Q(-16) =\n', Qk)

    for k in range(-16, 15):
        zk1 = get_zk1(Z, k)
        ak1 = get_ak1(A, k)
        Rk1 = get_Rk1(R, k)

        #Z~(k+1|k)
        Ztidlek1 = calc_Ztidlek1(zk1, ak1, x_hat)
        print('観測予測誤差Z~(', k+1, ') =', Ztidlek1)

        #S(k+1)
        Sk1 = calc_Sk1(ak1, Pk, Rk1)
        print('観測予測誤差共分散S(', k+1, ') =', Sk1)

        #Z(k)
        print('観測値 Z(', k+1, ') =', zk1)

        #E(z)
        E = calc_epsilon(Ztidlek1, Sk1)
        print('epsilon E(', k+1, ') =', E)

        #W(k+1)
        Wk1 = calc_Wk1(Pk, ak1, Sk1)
        print('フィルタゲインW(', k+1, ') =\n', Wk1)

        #P(k|k)
        Pk = calc_Pk1(Pk, Wk1, Sk1)
        print('推定誤差共分散P(', k+1, ') =\n', Pk)

        #P(k+1|k)
        Pk = calc_Pk2(Pk, Qk)
        print('予測誤差共分散P(', k+1, '|', k, ') =\n', Pk)

        #x^(k+1)の計算
        x_hat = calc_x_hatk1(x_hat, Wk1, Ztidlek1)
        print('推定値x^(', k+1, ') =', x_hat)

#メイン
if __name__ == '__main__':
    #必要な行列の用意
    #観測回数k(31*1) -15 <- k <- 15
    k = np.arange(-15, 16)

    #観測値Z(31*1)
    Z = np.array([162.1746, 139.5805, 113.8133, 94.3372, 74.7258, 59.3817, 41.4117, 26.5951, 20.1832, 8.8816, 1.8636, -5.0213, -5.8861, -5.7711, -4.9332, -1.9845, 2.0593, 12.3849, 17.9044, 30.1826, 41.1677, 55.7128, 74.2944, 93.7607, 112.6638, 134.9818, 162.7143, 188.9610, 219.6236, 248.9036, 281.3082])

    #係数行列A(31*3)
    A = np.zeros((31, 3))
    for i in range(31):
        for j in range(3):
            if j == 0:
                A[i][j] = 1
            elif j == 1:
                A[i][j] = k[i]
            else:
                A[i][j] = k[i]*k[i]

    #観測雑音wの共分散行列R(31*31)
    R = np.zeros((31, 31))
    for i in range(31):
        for j in range(31):
            #対角成分
            if j == i:
                #kが奇数の場合
                if k[i] % 2 != 0:
                    R[i][j] = 1.0
                #kが偶数の場合
                else:
                    R[i][j] = 4.0
            #非対角成分
            else:
                R[i][j] = 0
    
    #推定値x_hat
    x_hat_init = np.zeros(3)

    #推定誤差共分散行列P
    Pk_init = np.zeros((3, 3))
    #対角成分（変更すると推定精度が変わる）
    np.fill_diagonal(Pk_init, 1000000)
    
    #プラント雑音共分散行列q
    Qk_init = np.zeros((3, 3))
    #対角成分（変更すると推定精度が変わる）
    np.fill_diagonal(Qk_init, 0)

    print('Start Kalman Filtering')
    KalmanFiltering(x_hat_init, Pk_init, Qk_init, Z, A, R)
