import numpy as np
import pandas as pd
import copy
import pickle
import matplotlib.pyplot as plt

def RunKLUCB(numRound, listHB, listLB, listU, TF_causal, TF_naive, TF_CF, listnOBS, listUobs):
    def MaxKL(mu_hat, ft, NaT, init_maxval=1):
        def BinoKL(mu_hat, mu):
            if mu_hat == mu:
                return 0
            else:
                result = mu_hat * np.log(mu_hat / mu) + (1 - mu_hat) * np.log((1 - mu_hat) / (1 - mu))
            return result

        def MaxBinarySearch(mu_hat, M, maxval):
            if M < 0:
                print(mu_hat, M, "ERROR")
            terminal_cond = 1e-8
            eps = 1e-12
            if mu_hat == 1:
                return 1
            elif mu_hat == 0:
                mu_hat += eps  # diff
            mu = copy.copy(mu_hat)

            iteridx = 0
            while 1:
                iteridx += 1
                mu_cand = (mu + maxval) / 2
                KL_val = BinoKL(mu_hat, mu_cand)
                diff = np.abs(KL_val - M)
                # print(mu, mu_hat, mu_cand,KL_val, M, diff)
                if diff < terminal_cond:
                    mu = mu_cand
                    return mu

                if KL_val < M:
                    mu = copy.copy(mu_cand)
                else:
                    maxval = copy.copy(mu_cand)

                if np.abs(mu - maxval) < terminal_cond:
                    return mu

                if iteridx > 20:
                    return mu
        maxval = copy.copy(init_maxval)
        M = ft / NaT
        mu = MaxBinarySearch(mu_hat, M, maxval)
        return mu

    def MaxKLInverse(mu_hat, C, init_maxval):
        def BinoKL(mu, mu_hat):
            if mu_hat == mu:
                return 0
            else:
                result = mu * np.log(mu / mu_hat) + (1 - mu) * np.log((1 - mu) / (1 - mu_hat))
            return result

        def MaxBinarySearch(mu_hat, C, maxval):
            terminal_cond = 1e-8
            eps = 1e-12
            if mu_hat == 1:
                return 1
            elif mu_hat == 0:
                mu_hat += eps  # diff
            mu = copy.copy(mu_hat)

            iteridx = 0
            while 1:
                iteridx += 1
                mu_cand = (mu + maxval) / 2
                KL_val = BinoKL(mu_cand, mu_hat)
                diff = np.abs(KL_val - C)
                # print(mu, mu_hat, mu_cand,KL_val, M, diff)
                if diff < terminal_cond:
                    mu = mu_cand
                    return mu

                if KL_val < C:
                    mu = copy.copy(mu_cand)
                else:
                    maxval = copy.copy(mu_cand)

                if np.abs(mu - maxval) < terminal_cond:
                    return mu

                if iteridx > 20:
                    return mu
        maxval = copy.copy(init_maxval)
        mu = MaxBinarySearch(mu_hat, C, maxval)
        return mu

    def ComputeDynamicMean(n, prevM, lastElem):
        M = ((n - 1) * prevM + lastElem) / n
        return M

    def UpdateAfterArm(dictNumArm, dictM, dictLastElem, armChosen, reward):
        dictNumArm[armChosen] += 1
        dictLastElem[armChosen] = reward
        dictM[armChosen] = ComputeDynamicMean(dictNumArm[armChosen], dictM[armChosen], reward)
        return [dictNumArm, dictM, dictLastElem]

    ''' Definition of variable '''
    dictNumArm = dict() # Number of pulling arm a
    dictM = dict() # Average of reward of arm a
    dictLastElem = dict() # previous reward of arm a
    listTFArmCorrect = list() # 1 if arm a = optimal arm // 0 otherwise.
    listCummRegret = list() # cummulative regret += E[Y|do(X=optimal)] - E[Y|do(X=a)]

    armOpt = np.argmax(listU)
    cummRegret = 0
    armDomain = np.arange(len(listU))

    for a in armDomain:
        if TF_naive == True:
            dictNumArm[a] = listnOBS[a]
            dictM[a] = listUobs[a]
            dictLastElem[a] = 0
        else:
            dictNumArm[a] = 0
            dictM[a] = 0
            dictLastElem[a] = 0

    ''' Initial pulling'''
    # Pulling all arm at once.
    for a in armDomain:
        u0,u1 = fU()
        reward = fY(u0,u1,a)

        dictNumArm, dictM, dictLastElem = UpdateAfterArm(dictNumArm,dictM,dictLastElem, a, reward)
        cummRegret += listU[armOpt] - listU[a]
        listCummRegret.append(cummRegret)
        if a == armOpt:
            listTFArmCorrect.append(1)
        else:
            listTFArmCorrect.append(0)

    ''' Run!'''
    f = lambda x: np.log(x) + 3 * np.log(np.log(x))
    list_listUpper = []
    for idxround in range(numRound-len(armDomain)):
        t = idxround + len(armDomain) + 1 # t=3,4,...,nRound+2 // total nRound.
        # Compute the mean reward
        listUpper = list() # Each arm's upper confidence.
        listHB_CF = np.zeros(len(armDomain))
        listLB_CF = np.zeros(len(armDomain))
        for a in armDomain:
            # Compute
            armDomain_noa = [aidx for aidx in armDomain if aidx != a]
            mu_hat = dictM[a] # Average rewards of arm a up to (t-1)
            ft = f(t)
            # print(t, a, mu_hat, ft, dictNumArm[a])
            upper_a = MaxKL(mu_hat,ft,dictNumArm[a],init_maxval=1) # argmax_u KL(mu_hat, u) < (ft/Na(t)) s.t. 0<= u <= 1.
            if TF_causal:
                upper_a = np.max([np.min([listHB[a], upper_a]),listLB[a]])
            elif TF_CF:
                f_delta = lambda n,delta: np.sqrt( (1/(2*n))*np.log(1/delta)  )
                HB_cf_sumval = 0
                LB_cf_sumval = 0
                for aidx in armDomain_noa:
                    px_obs = px[aidx]
                    px_dot = dictNumArm[aidx]/sum(dictNumArm.values())
                    C = -np.log(px_dot)

                    HB_cf = MaxKLInverse(min(mu_hat+f_delta(dictNumArm[a],0.99),0.01),C,init_maxval=1)
                    LB_cf = MaxKLInverse(max(mu_hat-f_delta(dictNumArm[a],0.01),0.01),C,init_maxval=0)

                    HB_cf_sumval += (HB_cf * px_obs)
                    LB_cf_sumval += (LB_cf * px_obs)

                listHB_CF[a] = min(listLB[a] + HB_cf_sumval,1.0)
                listLB_CF[a] = min(listLB[a] + LB_cf_sumval,1.0)

                # print(t, a, listLB_CF[a], mu_hat, listHB_CF[a])

                upper_a = np.max([np.min([listHB_CF[a], upper_a]), listLB_CF[a]])
            listUpper.append(upper_a)
            list_listUpper.append(listUpper)
        # print(t,listUpper)
        armChosen = np.argmax(listUpper)

        u0, u1 = fU()
        reward = fY(u0, u1, armChosen)

        # reward = fY(armChosen, u)
        # reward = np.random.binomial(1, listU[armChosen])
        dictNumArm, dictM, dictLastElem = UpdateAfterArm(dictNumArm, dictM, dictLastElem, armChosen, reward)

        cummRegret += (np.max(listU) - listU[armChosen])
        if armChosen == armOpt:
            listTFArmCorrect.append(1)
        else:
            listTFArmCorrect.append(0)
        listCummRegret.append(cummRegret)
    return np.asarray(listTFArmCorrect), np.asarray(listCummRegret), list_listUpper

def fU():
    U0 = np.random.binomial(1, 0.2)
    U1 = np.random.binomial(1, 0.5)
    return U0,U1

def fY(U0,U1,X):
    return (U0|U1)^X

def GenParam(nOBS):
    np.random.seed(1)
    N = nOBS
    U0 = np.random.binomial(1, 0.2, N)
    U1 = np.random.binomial(1, 0.5, N)
    X = np.random.binomial(1, 0.4, N) * (1 - (U0 ^ U1))
    numX = len(np.unique(X))
    px = [1-np.mean(X),np.mean(X)]
    Y = fY(U0, U1, X)

    OBS = pd.DataFrame({'X': X, 'Y': Y})

    X0 = np.array([0] * N)
    X1 = np.array([1] * N)
    Y0 = fY(U0, U1, X0)
    Y1 = fY(U0, U1, X1)

    print('Y0:', np.mean(Y0))
    print('Y1:', np.mean(Y1))
    print('Y|x0:', np.mean(OBS[OBS['X'] == 0]['Y']))
    print('Y|x1:', np.mean(OBS[OBS['X'] == 1]['Y']))

    # Bound
    l0 = np.mean(OBS[OBS['X'] == 0]['Y']) * (1 - np.mean(X))
    l1 = np.mean(OBS[OBS['X'] == 1]['Y']) * (np.mean(X))
    h0 = l0 + np.mean(X)
    h1 = l1 + (1 - np.mean(X))

    listU = [np.mean(Y0), np.mean(Y1)]
    listHB = [h0,h1]
    listLB = [l0,l1]

    print(l0, np.mean(Y0), h0)
    print(l1, np.mean(Y1), h1)
    return listU,listLB,listHB,px

if __name__ == "__main__":
    numRound = 5000
    numSim = 100
    nOBS = 5000

    listU, listLB, listHB,px = GenParam(nOBS)

    ''' PARAM 0, 181030 '''


    # ''' PARAM 1 '''
    # U = np.random.normal(loc=0.5,scale=1,size=nOBS) + np.random.rand(nOBS)
    # X = fX(U)
    # px = [1-np.mean(X),np.mean(X)]
    # X0 = np.array([0] * nOBS)
    # X1 = np.array([1] * nOBS)
    # Y = fY(X,U)
    # Y0 = fY(X0,U)
    # Y1 = fY(X1,U)
    # OBS = pd.DataFrame({'X':X,'Y':Y})
    # Yo_0 = np.array(OBS[OBS['X']==0]['Y'])
    # Yo_1 = np.array(OBS[OBS['X'] == 1]['Y'])
    #
    # listU = [np.mean(Y0),np.mean(Y1)]
    # listLB = [np.mean(Yo_0)*px[0],np.mean(Yo_1)*px[1]]
    # listHB = [listLB[0] + px[1-0], listLB[1] + px[1-1]]

    ''' PARAM 2 '''
    # U = np.random.normal(loc=0.5, scale=1, size=nOBS) + np.random.rand(nOBS)
    # numX = 2
    # bincut = [0.2,0.4]
    # def fX(u, numX, bincut):
    #     g = 2 * (U - 0.5) + 0.2 * np.random.normal(loc=0, scale=1, size=len(u))
    #     X = np.exp(g) / (np.exp(g) + 1)
    #     if numX == 2:
    #         X = np.round(X)
    #     else:
    #         mybin = [0] + bincut + [1]
    #         np.digitize(X, mybin)
    #     return X
    # X = fX(U,numX,bincut)
    # px = [1 - np.mean(X), np.mean(X)]
    # X0 = np.array([0] * nOBS)
    # X1 = np.array([1] * nOBS)
    # Y = fY(X, U)
    # Y0 = fY(X0, U)
    # Y1 = fY(X1, U)
    # OBS = pd.DataFrame({'X': X, 'Y': Y})
    # Yo_0 = np.array(OBS[OBS['X'] == 0]['Y'])
    # Yo_1 = np.array(OBS[OBS['X'] == 1]['Y'])
    #
    # listU = [np.mean(Y0), np.mean(Y1)]
    # listLB = [np.mean(Yo_0) * px[0], np.mean(Yo_1) * px[1]]
    # listHB = [listLB[0] + px[1 - 0], listLB[1] + px[1 - 1]]

    ''' PARAM 3 '''
    # U = np.random.normal(loc=0.5, scale=2, size=nOBS) + 0.5*np.random.rand(nOBS)
    # numX = 4
    # bincut = [40,50,90]
    # def fX(u, numX, bincut):
    #     rand_noise = 0.2 * np.random.normal(loc=0, scale=1, size=len(u))
    #     X = 2 * (U - 0.3) + ((U-1)**2) + 1  +  rand_noise
    #     X = np.exp(X) / (np.exp(X) + 1)
    #     if numX == 2:
    #         X = np.round(X)
    #     else:
    #         mybin = [0] + list(np.percentile(X,q=bincut)) + [1]
    #         X=np.digitize(X, mybin)
    #         X = X-1
    #     return X
    # X = fX(U,numX,bincut)
    # px = [list(X).count(idx)/nOBS for idx in range(numX-1)]
    # px = px + [1-sum(px)]
    #
    # X0 = np.array([0] * nOBS)
    # X1 = np.array([1] * nOBS)
    # X2 = np.array([2] * nOBS)
    # X3 = np.array([3] * nOBS)
    # # fY = lambda x, u: np.exp(-0.6 * x + 0.3 * u) / (np.exp(-0.6 * x + 0.3 * u) + 1)
    # Y = fY(X, U)
    # Y0 = fY(X0, U)
    # Y1 = fY(X1, U)
    # Y2 = fY(X2, U)
    # Y3 = fY(X3, U)
    # OBS = pd.DataFrame({'X': X, 'Y': Y})
    # Yo_0 = np.array(OBS[OBS['X'] == 0]['Y'])
    # Yo_1 = np.array(OBS[OBS['X'] == 1]['Y'])
    # Yo_2 = np.array(OBS[OBS['X'] == 2]['Y'])
    # Yo_3 = np.array(OBS[OBS['X'] == 3]['Y'])
    # listU = [np.mean(Y0), np.mean(Y1), np.mean(Y2), np.mean(Y3)]
    # listLB = [np.mean(Yo_0) * px[0], np.mean(Yo_1) * px[1], np.mean(Yo_2) * px[2], np.mean(Yo_3) * px[3]]
    # listHB = []
    # for idx in range(len(px)):
    #     temp = [px[i] for i in range(len(px)) if i != idx]
    #     listHB.append(listLB[idx] + sum(temp))

    ''' PARAM 4 '''
    # U = np.random.rand(nOBS)
    # standard = [100*np.percentile(U,q=30),100*np.percentile(U,q=10),100*np.percentile(U,q=40),100*np.percentile(U,q=20)]
    # px = np.random.dirichlet(standard)
    # X = np.random.choice(len(px),nOBS,replace=True,p=px)
    # X0 = np.array([0] * nOBS)
    # X1 = np.array([1] * nOBS)
    # X2 = np.array([2] * nOBS)
    # X3 = np.array([3] * nOBS)
    # # fY = lambda x, u: np.exp(-0.6 * x + 0.3 * u) / (np.exp(-0.6 * x + 0.3 * u) + 1)
    # Y = fY(X, U)
    # Y0 = fY(X0, U)
    # Y1 = fY(X1, U)
    # Y2 = fY(X2, U)
    # Y3 = fY(X3, U)
    # OBS = pd.DataFrame({'X': X, 'Y': Y})
    # Yo_0 = np.array(OBS[OBS['X'] == 0]['Y'])
    # Yo_1 = np.array(OBS[OBS['X'] == 1]['Y'])
    # Yo_2 = np.array(OBS[OBS['X'] == 2]['Y'])
    # Yo_3 = np.array(OBS[OBS['X'] == 3]['Y'])
    # listU = [np.mean(Y0), np.mean(Y1), np.mean(Y2), np.mean(Y3)]
    # listLB = [np.mean(Yo_0) * px[0], np.mean(Yo_1) * px[1], np.mean(Yo_2) * px[2], np.mean(Yo_3) * px[3]]
    # listHB = []
    # for idx in range(len(px)):
    #     temp = [px[i] for i in range(len(px)) if i != idx]
    #     listHB.append(listLB[idx] + sum(temp))

    ''' PARAM 5 '''
    # px = [0.1, 0.6, 0.3]
    # U = np.random.rand(nOBS)
    # X = np.random.choice(len(px),nOBS,replace=True,p=px)
    # X0 = np.array([0] * nOBS)
    # X1 = np.array([1] * nOBS)
    # X2 = np.array([2] * nOBS)
    # # fY = lambda x, u: np.exp(1*x - 2*u)/(np.exp(1*x - 2*u)+1)
    # Y = fY(X, U)
    # Y0 = fY(X0, U)
    # Y1 = fY(X1, U)
    # Y2 = fY(X2, U)
    #
    # OBS = pd.DataFrame({'X': X, 'Y': Y})
    # Yo_0 = np.array(OBS[OBS['X'] == 0]['Y'])
    # Yo_1 = np.array(OBS[OBS['X'] == 1]['Y'])
    # Yo_2 = np.array(OBS[OBS['X'] == 2]['Y'])
    # Yo_3 = np.array(OBS[OBS['X'] == 3]['Y'])
    # listU = [np.mean(Y0), np.mean(Y1), np.mean(Y2)]
    # listLB = [np.mean(Yo_0) * px[0], np.mean(Yo_1) * px[1], np.mean(Yo_2) * px[2]]
    #
    # listHB = []
    # for idx in range(len(px)):
    #     temp = [px[i] for i in range(len(px)) if i != idx]
    #     listHB.append(listLB[idx] + sum(temp))

    ''' MAB '''
    arrayTF = np.array([0] * numRound)
    arrayCUM = np.array([0] * numRound)
    matTF = np.zeros((numSim, numRound))
    matCUM = np.zeros((numSim, numRound))
    dictUP = dict()
    for k in range(numSim):
        print(k,'standard')
        TF, CUM, UP = RunKLUCB(numRound, listHB, listLB, listU, TF_causal=False, TF_naive=False, TF_CF=False, listnOBS=[100, 100],
                               listUobs=[0, 0])

        matTF[k, :] = TF
        matCUM[k, :] = CUM
        dictUP[k] = np.array(UP)
        arrayTF = arrayTF + TF
        arrayCUM = arrayCUM + CUM
    print("HB",listHB)
    arrayTF = arrayTF / numSim
    arrayCUM = arrayCUM / numSim

    arrayTF_C = np.array([0] * numRound)
    arrayCUM_C = np.array([0] * numRound)
    matTF_C = np.zeros((numSim, numRound))
    matCUM_C = np.zeros((numSim, numRound))
    dictUP_C = dict()
    for k in range(numSim):
        print(k, 'C')
        TF_C, CUM_C, UP_C = RunKLUCB(numRound, listHB, listLB, listU, TF_causal=True, TF_naive=False, TF_CF=False,
                               listnOBS=[100, 100], listUobs=[0, 0])
        matTF_C[k, :] = TF_C
        matCUM_C[k, :] = CUM_C
        dictUP_C[k] = np.array(UP_C)
        arrayTF_C = arrayTF_C + TF_C
        arrayCUM_C = arrayCUM_C + CUM_C
    print("HB", listHB)
    arrayTF_C = arrayTF_C / numSim
    arrayCUM_C = arrayCUM_C / numSim

    arrayTF_CF = np.array([0] * numRound)
    arrayCUM_CF = np.array([0] * numRound)
    matTF_CF = np.zeros((numSim, numRound))
    matCUM_CF = np.zeros((numSim, numRound))
    dictUP_CF = dict()
    for k in range(numSim):
        print(k, 'CF')
        TF_CF, CUM_CF, UP_CF = RunKLUCB(numRound, listHB, listLB, listU, TF_causal=False, TF_naive=False, TF_CF=True,
                               listnOBS=[100, 100], listUobs=[0, 0])
        matTF_CF[k, :] = TF_CF
        matCUM_CF[k, :] = CUM_CF
        dictUP_CF[k] = np.array(UP_CF)
        arrayTF_CF = arrayTF_CF + TF_CF
        arrayCUM_CF = arrayCUM_CF + CUM_CF
    print("HB", listHB)
    arrayTF_CF = arrayTF_CF / numSim
    arrayCUM_CF = arrayCUM_CF / numSim

    xDomain = np.arange(numRound)
    alpha = 0.05
    colorAlpha = 0.3
    linewidthval = 2
    tt = np.arange(0,numRound,200)

    plt.figure(1,figsize=(8,5))
    # plt.title('Cum.Reg')
    plt.plot(xDomain[tt], arrayCUM[tt], 'red', label='klUCB',linewidth=linewidthval)
    plt.plot(xDomain[tt], arrayCUM[tt], 'ro')
    # plt.fill_between(xDomain[tt], np.percentile(matCUM[:,tt], q=100 * alpha, axis=0),
    #                  np.percentile(matCUM[:,tt], q=100 * (1 - alpha), axis=0), facecolor='red', alpha=colorAlpha)
    plt.plot(xDomain[tt], arrayCUM_C[tt], 'blue', label='klUCB-C',linewidth=linewidthval)
    plt.plot(xDomain[tt], arrayCUM_C[tt], 'bo')
    # plt.fill_between(xDomain[tt], np.percentile(matCUM_C[:,tt], q=100 * alpha, axis=0),
    #                  np.percentile(matCUM_C[:,tt], q=100 * (1 - alpha), axis=0), facecolor='blue', alpha=colorAlpha)
    # plt.plot(xDomain[tt], arrayCUM_CF[tt], 'green', label='klUCB-CF',linewidth=linewidthval)
    # plt.plot(xDomain[tt], arrayCUM_CF[tt], 'go')
    # plt.fill_between(xDomain[tt], np.percentile(matCUM_CF[:,tt], q=100 * alpha, axis=0),
    #                  np.percentile(matCUM_CF[:,tt], q=100 * (1 - alpha), axis=0), facecolor='green', alpha=colorAlpha)
    plt.legend(loc='upper left',fontsize=15)
    plt.ylabel('Cum.Reg',fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    plt.figure(2)
    plt.title('Opt.Prob')
    plt.plot(xDomain, arrayTF, 'red', label='klUCB')
    plt.fill_between(xDomain, np.percentile(matTF, q=100 * alpha, axis=0),
                     np.percentile(matTF, q=100 * (1 - alpha), axis=0), facecolor='red', alpha=colorAlpha)
    plt.plot(xDomain, arrayTF_C, 'blue', label='klUCB-C')
    plt.fill_between(xDomain, np.percentile(matTF_C, q=100 * alpha, axis=0),
                     np.percentile(matTF_C, q=100 * (1 - alpha), axis=0), facecolor='blue', alpha=colorAlpha)
    # plt.plot(xDomain, arrayTF_CF, 'green', label='klUCB-CF')
    # plt.fill_between(xDomain, np.percentile(matTF_CF, q=100 * alpha, axis=0),
    #                  np.percentile(matTF_CF, q=100 * (1 - alpha), axis=0), facecolor='green', alpha=colorAlpha)
    plt.legend()
    plt.show()

    # pickle.dump(listHB,open('Result/listHB.pkl','wb'))
    # pickle.dump(listU, open('Result/listU.pkl', 'wb'))
    # pickle.dump(listLB, open('Result/listLB.pkl', 'wb'))
    # pickle.dump(dictUP,open('Result/dictUP.pkl','wb'))
    # pickle.dump(dictUP_C, open('Result/dictUP_C.pkl', 'wb'))
    # pickle.dump(dictUP_CF, open('Result/dictUP_CF.pkl', 'wb'))
    # pickle.dump(matTF, open('Result/MatTF.pkl', 'wb'))
    # pickle.dump(matTF_C, open('Result/MatTF_C.pkl', 'wb'))
    # pickle.dump(matTF_CF, open('Result/MatTF_CF.pkl', 'wb'))
    # pickle.dump(matCUM, open('Result/MatCUM.pkl', 'wb'))
    # pickle.dump(matCUM_C, open('Result/MatCUM_C.pkl', 'wb'))
    # pickle.dump(matCUM_CF, open('Result/MatCUM_CF.pkl', 'wb'))

    # k = 19
    # plt.plot(matCUM[k,:], 'b', label='klUCB')
    # plt.plot(matCUM_C[k,:], 'r', label='klUCB-C')
    # plt.plot(matCUM_CF[k,:], 'g', label='klUCB-CF')
    # plt.legend()
