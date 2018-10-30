import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
import copy
import itertools
from sklearn import preprocessing
import pickle
import operator
import matplotlib.pyplot as plt

def HideCovarDF(DF,selected_covariates):
    ## Resulting dataset
    DF = DF[selected_covariates]
    return DF

def IST_LabelEncoder(IST, ColName):
    le = preprocessing.LabelEncoder()

    COL = copy.copy(IST[ColName])
    list_label = list(pd.unique(COL))
    le.fit(list_label)

    COL = le.transform(COL)

    IST[ColName] = COL
    return IST

def LabelRecover(df, df_orig, colname):
    print(pd.unique(df[colname]))
    print(pd.unique(df_orig[colname]))
    # print( pd.concat([df[colname],df_orig[colname]],axis=1) )

def ReduceIST(IST, chosen_variables):
    # Exclude patients not having AF
    # IST = IST.loc[pd.isnull(IST['RATRIAL']) == False]
    # Exclude patients having no recovery information
    IST = IST.loc[(IST['FRECOVER']) != 'U']
    IST = IST.loc[(IST['FDEAD'])!= 'U']
    IST = IST[chosen_variables]
    return IST

def IndexifyDisc(IST, discrete_variables):
    for disc_val in discrete_variables:
        IST = IST_LabelEncoder(IST, disc_val)
    return IST

def ContToDisc(IST,continuous_variables):
    # Discretize the continuous variable
    # continuous_variable = ['RSBP', 'AGE', 'RDELAY']  # list(set(chosen_variables) - set(discrete_variables))
    for cont_val in continuous_variables:
        IST[cont_val] = BinaryCategorize(IST[cont_val])
    return IST

def BinaryCategorize(df):
    df_copy = copy.copy(df)
    df_q = list(df.quantile([0.5]))

    df_copy[df_copy <= df_q[0]] = 0
    df_copy[df_copy > df_q[0]] = 1

    return df_copy

def ComputeXYEffect(df,X,Y):
    return [np.mean(df[df[X]==0][Y]),np.mean(df[df[X]==1][Y])]

def GenOBS(EXP, seed_obs = 1):
    pxu = [0, 0, 0, 0, 0, 0, 1, 0.2, 0.1, 0.1, 0, 0]
    listSample = []
    for idx in range(len(EXP)):
        elem = EXP.iloc[idx]
        elem_treat = elem['RXASP']
        elem_sex = elem['SEX']
        elem_age = elem['AGE']
        elem_RCONSC = elem['RCONSC']

        u = int(6*elem_age + 3*elem_sex + elem_RCONSC + 1)
        x = np.random.binomial(1,pxu[u-1])
        if x == elem_treat:
            listSample.append(elem)
    OBS = pd.DataFrame(listSample)
    return OBS

def BoundsPl(OBS, OBS_Z, OBS_X, listPz, pl, zName, xName):
    zDomain = np.unique(OBS_Z)
    xDomain = np.unique(OBS_X)
    N = len(OBS)

    sum_prob_lb = 0
    zidx = 0
    for z in zDomain:
        z = int(z)
        pz = listPz[zidx]
        zidx += 1
        for x in xDomain:
            x = int(x)
            pl_xz = FunEval(pl,z)
            print(pl_xz, x)
            pl_xz = pl_xz[x]
            EY_xz = np.mean(OBS[(OBS[zName]==z) & (OBS[xName]==x)]['Y'])
            if pd.isnull(EY_xz):
                EY_xz = 0
            sum_prob_lb += EY_xz * pl_xz * pz

    sum_prob_ub = 0
    LB = copy.copy(max(sum_prob_lb, 0))

    zidx = 0
    for z in zDomain:
        zidx += 1
        for x in xDomain:
            x = int(x)
            pxz = len(OBS[(OBS[zName]==z) & (OBS[xName]==x)]) / N
            if pd.isnull(pxz):
                pxz = 0
            pi_nonx_z = FunEval(pl,z)
            pi_nonx_z = pi_nonx_z[1-x]
            sum_prob_ub += pxz * pi_nonx_z
    HB = LB + sum_prob_ub
    HB = min(HB, 1)
    return [LB, HB]

def EmpBoundsPl(OBS,zName,listPolicy,delta,N):
    delta = delta / len(listPolicy)
    fn = np.sqrt(((2 * N) ** (-1)) * (np.log(4) - np.log(delta)))

    Ys = list(OBS['Y'])
    Xs = list(OBS['RXASP'])
    Zs = np.array(OBS[zName])

    listLB = []
    listHB = []

    for plidx in range(len(listPolicy)):
        sumval = 0
        pl = listPolicy[plidx]
        for idx in range(N):
            y = Ys[idx]
            x = int(Xs[idx])
            z = Zs[idx]
            pl_xz = FunEval(pl, z)
            pl_xz = pl_xz[x]
            sumval += (y*pl_xz)
        LB_orig = sumval/N
        LB = max(0,LB_orig - fn)

        sumval = 0
        for idx in range(N):
            x = int(Xs[idx])
            z = Zs[idx]
            pl_xz = FunEval(pl, z)
            pl_xz = pl_xz[1-x]
            sumval += pl_xz
        HB_orig = LB_orig + sumval/N
        HB = HB_orig + fn
        HB = min(HB,1)

        listLB.append(LB)
        listHB.append(HB)

    return [listLB,listHB]

def EmpiricalComputeBound(OBS,X,delta,N):
    # sampleOBS = OBS.sample(n=N)
    sampleOBS = OBS.iloc[:N]
    delta = delta/2
    fn = np.sqrt(((2 * N) ** (-1)) * (np.log(4) - np.log(delta)))

    Px0 = len(sampleOBS[sampleOBS[X] == 0]) / N
    Px1 = len(sampleOBS[sampleOBS[X] == 1]) / N

    Lx0_before = np.mean((sampleOBS[X] == 0) * sampleOBS['Y'])
    Lx1_before = np.mean((sampleOBS[X] == 1) * sampleOBS['Y'])

    Lx0 = max(0,Lx0_before - fn)
    Lx1 = max(0,Lx1_before - fn)

    Hx0 = Lx0_before + Px1
    Hx1 = Lx1_before + Px0

    Hx0 = min(Hx0 + fn,1)
    Hx1 = min(Hx1 + fn,1)

    return [[Lx0, Lx1], [Hx0, Hx1]]

def CheckCase2(h_subopt, u_opt):
    if h_subopt < u_opt:
        return True
    else:
        return False

def CheckCase2BestArm(HB,U):
    def FindMax2(U):
        Ucopy = np.array(copy.copy(U))
        Ucopy = -np.sort(-Ucopy)
        u1, u2 = Ucopy[:2]
        return [u1, u2]
    u1,u2 = FindMax2(U)

    hx0, hx1 = HB
    Ux0, Ux1 = U

    if Ux0 < Ux1:
        if hx0 < (u1 + u2)/2:
            return True
        else:
            return False
    else:
        if hx1 < (u1 + u2)/2:
            return True
        else:
            return False

def RewardFun(x, s, r, a):
    y = x + s + 0.2 * (r - 1) - 2 * x * a * (1 - s)
    # y = x + 0.5*s + 0.2 * (r - 1) - x * a
    try:
        x = int(x)
        y = y + np.random.normal(0,1)
    except:
        y = y + np.random.normal(0, 1, len(x))
    y = 1 / (1 + np.exp(-y))
    return y

def ComputePz(EXP_Z, zName, EXP):
    # Assume z 1D
    zDomain = np.unique(EXP_Z)
    N = len(EXP)
    listPz = []
    for z in zDomain:
        pz = len(EXP[EXP[zName]==z])/N
        listPz.append(pz)
    return listPz

def FunEval(pl,z):
    z0, z1 = z
    pl_1z = pl(z0,z1)
    return [1-pl_1z, pl_1z]

def RunGenInstance():
    chosen_variables = ['SEX', 'AGE', 'RCONSC','RXASP']
    discrete_variables = ['SEX', 'RCONSC', 'RXASP']
    continuous_variables = ['AGE']
    necessary_set = ['RXASP', 'AGE', 'SEX', 'RCONSC']

    # ''' Temp experiment'''
    # temp_chosen = ['SEX', 'AGE', 'RCONSC', 'RXASP', 'FDEAD', 'FRECOVER']
    # temp_disc = ['SEX', 'RCONSC', 'RXASP', 'FDEAD', 'FRECOVER']
    #
    # IST = pd.read_csv('IST.csv')
    # IST = ReduceIST(IST,temp_chosen)
    # IST = IST.dropna()
    # IST_orig = copy.copy(IST)
    # IST = IndexifyDisc(IST, temp_disc)
    # IST['SEX'] = 1 - IST['SEX']
    # IST['RCONSC'] = 1 * (IST['RCONSC'] == 0) + 0 * (IST['RCONSC'] == 2) + 2 * (IST['RCONSC'] == 1)
    # IST = ContToDisc(IST,continuous_variables)
    # ''' TEMP OVER '''

    IST = pd.read_csv('IST.csv')
    IST = ReduceIST(IST, chosen_variables)
    IST_orig = copy.copy(IST)
    IST = IndexifyDisc(IST, discrete_variables)
    IST['SEX'] = 1 - IST['SEX']
    IST['RCONSC'] = 1 * (IST['RCONSC'] == 0) + 0 * (IST['RCONSC'] == 2) + 2 * (IST['RCONSC'] == 1)

    EXP = ContToDisc(IST, continuous_variables)
    EXP_SEX = list(EXP['SEX'])
    EXP_AGE = list(EXP['AGE'])
    EXP_RCONSC = list(EXP['RCONSC'])
    EXP_RXASP = list(EXP['RXASP'])
    EXP_Y = RewardFun(EXP['RXASP'],EXP['SEX'],EXP['RCONSC'],EXP['AGE'])
    EXP['Y'] = EXP_Y

    OBS = GenOBS(EXP)
    OBS_SEX = list(OBS['SEX'])
    OBS_AGE = list(OBS['AGE'])
    OBS_RCONSC = list(OBS['RCONSC'])
    OBS_RXASP = list(OBS['RXASP'])

    pickle.dump(EXP, open('sim_instance/EXP.pkl', 'wb'))
    pickle.dump(OBS, open('sim_instance/OBS.pkl', 'wb'))

    ''' KLUCB instance '''
    listnOBS = [len(OBS[OBS['RXASP'] == 0]), len(OBS[OBS['RXASP'] == 1])]
    listUobs= ComputeXYEffect(OBS, 'RXASP', 'Y')
    listU = ComputeXYEffect(EXP,'RXASP','Y')
    listLB, listHB = EmpiricalComputeBound(OBS, 'RXASP', delta=0.01, N=len(OBS))
    listLB100, listHB100 = EmpiricalComputeBound(OBS, 'RXASP', delta=0.01, N=100)
    print('KLUCB: Case 2?', CheckCase2(np.min(listHB), np.max(listU)))

    pickle.dump(listnOBS, open('sim_instance/listnOBS.pkl','wb'))
    pickle.dump(listUobs, open('sim_instance/listUobs.pkl','wb'))
    pickle.dump(listLB, open('sim_instance/listLB.pkl', 'wb'))
    pickle.dump(listLB100, open('sim_instance/listLB100.pkl', 'wb'))
    pickle.dump(listHB, open('sim_instance/listHB.pkl', 'wb'))
    pickle.dump(listHB100, open('sim_instance/listHB100.pkl', 'wb'))
    pickle.dump(listU, open('sim_instance/listU.pkl','wb'))

    ''' BestArm instance'''

    ''' DUCB instance '''
    # Define policies
    # pl1 = lambda z: 0.01 if z == 0 else 0.02
    # pl2 = lambda z: 0.05 if z == 0 else 0.1
    # pl3 = lambda z: 0.97 if z == 0 else 0.99
    # pl4 = lambda z: 0.1 if z == 0 else 0.05

    low_prob = 0.01
    high_prob = 0.99
    mid_prob = 0.5

    def pl1(z0, z1):
        if z0 == 0 and z1 == 0:
            return high_prob
        elif z0 == 0 and z1 == 1:
            return low_prob
        elif z0 == 0 and z1 == 2:
            return low_prob
        elif z0 == 1 and z1 == 0:
            return high_prob
        elif z0 == 1 and z1 == 1:
            return low_prob
        elif z0 == 1 and z1 == 2:
            return low_prob

    def pl2(z0, z1):
        if z0 == 0 and z1 == 0:
            return high_prob
        elif z0 == 0 and z1 == 1:
            return low_prob
        elif z0 == 0 and z1 == 2:
            return low_prob
        elif z0 == 1 and z1 == 0:
            return low_prob
        elif z0 == 1 and z1 == 1:
            return low_prob
        elif z0 == 1 and z1 == 2:
            return low_prob

    def pl3(z0, z1):
        if z0 == 0 and z1 == 0:
            return low_prob
        elif z0 == 0 and z1 == 1:
            return high_prob
        elif z0 == 0 and z1 == 2:
            return low_prob
        elif z0 == 1 and z1 == 0:
            return low_prob
        elif z0 == 1 and z1 == 1:
            return high_prob
        elif z0 == 1 and z1 == 2:
            return low_prob

    def pl4(z0, z1):
        if z0 == 0 and z1 == 0:
            return low_prob
        elif z0 == 0 and z1 == 1:
            return low_prob
        elif z0 == 0 and z1 == 2:
            return low_prob
        elif z0 == 1 and z1 == 0:
            return low_prob
        elif z0 == 1 and z1 == 1:
            return high_prob
        elif z0 == 1 and z1 == 2:
            return high_prob

    def pl5(z0, z1):
        if z0 == 0 and z1 == 0:
            return low_prob
        elif z0 == 0 and z1 == 1:
            return low_prob
        elif z0 == 0 and z1 == 2:
            return low_prob
        elif z0 == 1 and z1 == 0:
            return low_prob
        elif z0 == 1 and z1 == 1:
            return low_prob
        elif z0 == 1 and z1 == 2:
            return low_prob

    def pl6(z0, z1):
        if z0 == 0 and z1 == 0:
            return mid_prob
        elif z0 == 0 and z1 == 1:
            return mid_prob
        elif z0 == 0 and z1 == 2:
            return mid_prob
        elif z0 == 1 and z1 == 0:
            return mid_prob
        elif z0 == 1 and z1 == 1:
            return mid_prob
        elif z0 == 1 and z1 == 2:
            return mid_prob

    listPolicy = [pl1, pl2, pl3, pl4, pl5, pl6]

    listUpl = []
    for pl in listPolicy:
        m = 0
        n = 0
        for s,r,a in zip(EXP_SEX, EXP_RCONSC, EXP_AGE):
            x = np.random.binomial(1,FunEval(pl,[s,r])[1])
            y = RewardFun(x,s,r,a)
            n += 1
            m = ((n - 1) * m + y) / n
        listUpl.append(m)
    print(listUpl)

    listPz = []
    for z0 in [0,1]:
        for z1 in [0,1,2]:
            pz = len(OBS[(OBS['SEX'] == z0) & (OBS['RCONSC'] == z1)]) / len(OBS)
            listPz.append(pz)

    # listPz = ComputePz(OBS_SEX,'SEX',OBS)
    listLBpl, listHBpl = EmpBoundsPl(OBS, ['SEX','RCONSC'], listPolicy, 0.01, len(OBS))
    listLBpl100, listHBpl100 = EmpBoundsPl(OBS, ['SEX','RCONSC'], listPolicy, 0.01, 100)
    for plidx in range(len(listPolicy)):
        print('Case 2', 'Policy', plidx, CheckCase2(listHBpl100[plidx], np.max(listUpl)))

    pickle.dump(listUpl, open('sim_instance/listUpl.pkl', 'wb'))
    pickle.dump(listPz, open('sim_instance/listPz.pkl', 'wb'))
    pickle.dump(listLBpl, open('sim_instance/listLBpl.pkl','wb'))
    pickle.dump(listHBpl, open('sim_instance/listHBpl.pkl','wb'))
    pickle.dump(listHBpl100, open('sim_instance/listHBpl100.pkl', 'wb'))
    pickle.dump(listLBpl100, open('sim_instance/listLBpl100.pkl', 'wb'))

def RunKLUCB(numRound, EXP, listHB, listLB, listU, TF_causal, TF_naive, listnOBS, listUobs):
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

                if iteridx > 50:
                    return mu
        maxval = copy.copy(init_maxval)
        M = ft / NaT
        mu = MaxBinarySearch(mu_hat, M, maxval)
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

    for a in [0,1]:
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
    for a in [0,1]:
        # sex, age, r, x, y
        row_cond = (EXP['RXASP']==a)
        s, _, r, x, y = EXP[row_cond].sample(1).values[0]
        reward = y
        dictNumArm, dictM, dictLastElem = UpdateAfterArm(dictNumArm,dictM,dictLastElem, a, reward)
        cummRegret += listU[armOpt] - listU[a]
        listCummRegret.append(cummRegret)
        if a == armOpt:
            listTFArmCorrect.append(1)
        else:
            listTFArmCorrect.append(0)

    ''' Run!'''
    f = lambda x: np.log(x) + 3 * np.log(np.log(x))
    for idxround in range(numRound-2):
        t = idxround + 2 + 1 # t=3,4,...,nRound+2 // total nRound.
        # Compute the mean reward
        listUpper = list() # Each arm's upper confidence.
        for a in [0,1]:
            # Compute
            mu_hat = dictM[a] # Average rewards of arm a up to (t-1)
            ft = f(t)
            # print(t, a, mu_hat, ft, dictNumArm[a])
            upper_a = MaxKL(mu_hat,ft,dictNumArm[a],init_maxval=1) # argmax_u KL(mu_hat, u) < (ft/Na(t)) s.t. 0<= u <= 1.
            if TF_causal:
                upper_a = np.max([np.min([listHB[a], upper_a]),listLB[a]])
            listUpper.append(upper_a)
        # print(t,listUpper)
        armChosen = np.argmax(listUpper)
        # sex, age, r, x, y
        row_cond = (EXP['RXASP'] == armChosen)
        s, _, r, x, y = EXP[row_cond].sample(1).values[0]
        reward = y
        dictNumArm, dictM, dictLastElem = UpdateAfterArm(dictNumArm, dictM, dictLastElem, armChosen, reward)

        cummRegret += (np.max(listU) - listU[armChosen])
        if armChosen == armOpt:
            listTFArmCorrect.append(1)
        else:
            listTFArmCorrect.append(0)
        listCummRegret.append(cummRegret)
    return np.asarray(listTFArmCorrect), np.asarray(listCummRegret)

def RunBestArm(EXP, listLB, listU, listHB, TF_causal, TF_naive, listnOBS, listUobs):
    def ComputeDynamicMean(n, prevM, lastElem):
        return ((n - 1) * prevM + lastElem) / n

    def UpdateAfterArm(dictNumArm, dictM, dictLastElem, armChosen, reward):
        dictNumArm[armChosen] += 1
        dictLastElem[armChosen] = reward
        dictM[armChosen] = ComputeDynamicMean(dictNumArm[armChosen], dictM[armChosen], dictLastElem[armChosen])
        return [dictNumArm, dictM, dictLastElem]

    def UpperConfidence(t, delta, eps):
        val = (1 + np.sqrt(eps)) * np.sqrt((1 + eps) * np.log(np.log((1 + eps) * t) / delta) / (2 * t))
        return val

    def FindMax2Idx(U):
        u1, u2 = FindMax2(U)
        idx_u1 = list(U).index(u1)
        idx_u2 = list(U).index(u2)
        return [idx_u1, idx_u2]

    def FindMax2(U):
        Ucopy = np.array(copy.copy(U))
        Ucopy = -np.sort(-Ucopy)
        u1, u2 = Ucopy[:2]
        return [u1, u2]

    def CheckStopCondition(listMuEst, listUEst, ht, lt, TF_causal):
        if TF_causal == True:
            if max(min(listMuEst[ht] - listUEst[ht], listHB[ht]), listLB[ht]) > max(min(listMuEst[lt] + listUEst[lt], listHB[lt]),
                                                                            listLB[lt]):
                return True
            else:
                return False

        else:
            if listMuEst[ht] - listUEst[ht] > listMuEst[lt] + listUEst[lt]:
                return True
            else:
                return False

    # Note this parametrization satisfied the definition of U
    eps = 0.01
    delta = 0.01  # (1-delta) is a confidence interval
    optarm = np.argmax(listU)
    # f = lambda eps: np.log(1+eps)/np.e

    # Declaration of variable
    numArm = 2
    dictNumArm = dict()
    dictM = dict()
    dictLastElem = dict()

    # dictlistArmReward = dict()
    # dictNumArmH = dict()

    for a in [0,1]:
        if TF_naive == True:
            dictNumArm[a] = listnOBS[a]
            dictM[a] = listUobs[a]
        else:
            dictNumArm[a] = 0
            dictM[a] = 0
        dictLastElem[a] = 0

    # Initialization
    t = 0
    for a in [0,1]:
        t += 1
        # sex, age, r, x, y
        row_cond = (EXP['RXASP'] == a)
        s, _, r, x, y = EXP[row_cond].sample(1).values[0]
        reward = y
        dictNumArm, dictM, dictLastElem = UpdateAfterArm(dictNumArm, dictM, dictLastElem, a, reward)

    # Start
    # print("")
    # print("BestArm Start")
    while 1:
        t += 1
        listUpperEst = list()
        listMuEst = list()
        listUEst = list()
        for a in [0,1]:
            muEst_a = dictM[a]
            listMuEst.append(muEst_a)
            U_a = UpperConfidence(dictNumArm[a], delta/numArm, eps)
            listUEst.append(U_a)
            if TF_causal == True:
                listUpperEst.append(max(min(muEst_a + U_a, listHB[a]),listLB[a]))
            else:
                listUpperEst.append(muEst_a + U_a)
        # print(t,listMuEst,listUpperEst)
        ht, lt = FindMax2Idx(listMuEst)

        for a in [ht, lt]:
            row_cond = (EXP['RXASP'] == a)
            s, _, r, x, y = EXP[row_cond].sample(1).values[0]
            reward = y
            dictNumArm, dictM, dictLastElem = UpdateAfterArm(dictNumArm, dictM, dictLastElem, a, reward)
        if t % 1000 == 0:
            print(t, ht, listMuEst[ht] - listUEst[ht], listMuEst[lt] + listUEst[lt])
        if CheckStopCondition(listMuEst, listUEst, ht, lt, TF_causal) == True:
            break

    return t,ht,dictM

def RunDUCB(numRound, EXP, OBS, listPz, listPolicy, listHB, listLB, listU, TF_causal, TF_naive):
    def FunEval(pl, z):
        z0, z1 = z
        pl_1z = pl(z0, z1)
        return [1 - pl_1z, pl_1z]

    def ComputePxz(OBS, x, zMask):
        mask = zMask & (OBS['RXASP'] == x)
        pxz = len(OBS[mask]) / len(OBS)
        return pxz

    def ComputePz(OBS, zMask):
        pz = len(OBS[zMask]) / len(OBS)
        return pz

    def ComputeMatM(listPolicy, listPz):
        def ComputeMpq(p, q):
            zPossible = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
            def f1(x):
                return x * np.exp(x - 1) - 1

            sumProb = 0
            for zidx in range(len(listPz)):
                Pz = listPz[zidx]
                z = zPossible[zidx]
                for x in [0, 1]:
                    pxz = FunEval(p, z)
                    pxz = pxz[x]
                    qxz = FunEval(q, z)
                    qxz = qxz[x]
                    sumProb += (f1(pxz / qxz) * qxz * Pz)
            return (1 + np.log(1 + sumProb))

        N_poly = len(listPolicy)
        poly_idx_iter = list(itertools.product(list(range(N_poly)), list(range(N_poly))))
        M_mat = np.zeros((N_poly, N_poly))
        for k, j in poly_idx_iter:
            # if k != j:
            pk = listPolicy[k]
            pj = listPolicy[j]
            M_mat[k, j] = ComputeMpq(pk, pj)
        return M_mat

    def ComputeZkt(dictNumPolicy, nPolicy, M, k):
        sumval = 0
        for j in range(nPolicy):
            Mkj = M[k, j]
            sumval += (dictNumPolicy[j] / Mkj)
        return sumval

    def ComputeSk(dictNumPolicy, nPolicy, M, k, t):
        c1 = 1
        eps = 1e-8
        Zkt = ComputeZkt(dictNumPolicy, nPolicy, M, k)
        return (c1* t * np.log(t) / (Zkt ** 2)) ** (1 / (2 + eps))

    def PullArmFromPl(pl, z):
        probs = FunEval(pl, z)
        return np.random.binomial(1, probs[1])

    def ComputeVal(k, j, xjs, yjs, zjs, listPolicy, M, t):
        Mkj = M[k, j]
        blockval = 2 * np.log(t) * Mkj

        pik = listPolicy[k]
        pij = listPolicy[j]

        pikval = FunEval(pik, zjs)
        pikval_xjs = pikval[xjs]
        pijval = FunEval(pij, zjs)
        pijval_xjs = pijval[xjs]

        invval = pikval_xjs / pijval_xjs
        if invval <= blockval:
            result = (1 / Mkj) * yjs * invval
        else:
            result = 0
        return result

    def ComputeMu(dictdictlistPolicyData, dictNumPolicy, listPolicy, M, k, t):
        nPolicy = len(listPolicy)
        Zkt = ComputeZkt(dictNumPolicy, nPolicy, M, k)
        pik = listPolicy[k]
        sumval = 0
        for j in range(nPolicy):
            Mkj = M[k, j]
            blockval = 2 * np.log(t) * Mkj
            pij = listPolicy[j]
            Xj = dictdictlistPolicyData[j]['X']
            Yj = dictdictlistPolicyData[j]['Y']
            Zj = dictdictlistPolicyData[j]['Z']
            for s in range(len(Xj)):
                xjs = int(Xj[s])
                yjs = Yj[s]
                zjs = Zj[s]

                pikval = FunEval(pik, zjs)
                pikval_xjs = pikval[xjs]
                pijval = FunEval(pij, zjs)
                pijval_xjs = pijval[xjs]

                invval = pikval_xjs / pijval_xjs
                if invval <= blockval:
                    sumval = sumval + (1 / Mkj) * yjs * invval
                else:
                    sumval = sumval + 0
        mu = sumval / Zkt
        return mu

    def UpdateAfterArm(dictNumPolicy, dictlistNumPolicyArm, dictdictlistPolicyData, plidxChosen, armChosen, z,
                       rewardReceived):
        dictNumPolicy[plidxChosen] += 1
        dictlistNumPolicyArm[plidxChosen][armChosen] += 1
        try:
            dictdictlistPolicyData[plidxChosen]['Z'].append(z)
            dictdictlistPolicyData[plidxChosen]['X'].append(armChosen)
            dictdictlistPolicyData[plidxChosen]['Y'].append(rewardReceived)
        except:
            dictdictlistPolicyData[plidxChosen]['Z'] = np.vstack((dictdictlistPolicyData[plidxChosen]['Z'], z))
            dictdictlistPolicyData[plidxChosen]['X'].append(armChosen)
            dictdictlistPolicyData[plidxChosen]['Y'].append(rewardReceived)
        return [dictNumPolicy, dictlistNumPolicyArm, dictdictlistPolicyData]

    def FindMaxKey(dct):
        return max(dct.items(), key=operator.itemgetter(1))[0]

    def ComputeMaxCut(listPolicy,listPz, Mmat):
        allvallist = []
        allPossible = list(itertools.combinations(range(len(listPolicy)), 2))
        for idx in range(len(listPolicy)):
            allPossible.append([idx, idx])
        Mmat = ComputeMatM(listPolicy, listPz)
        zPossible = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

        for x in [0, 1]:
            for zidx in range(len(zPossible)):
                z = zPossible[zidx]
                for k, j in allPossible:
                    plk = listPolicy[k]
                    plj = listPolicy[j]
                    Mkj = Mmat[k, j]

                    plkval = FunEval(plk, z)
                    plkval_x = plkval[x]

                    pljval = FunEval(plj, z)
                    pljval_x = pljval[x]

                    val = np.exp((plkval_x / pljval_x) * (1 / (2 * Mkj)))

                    allvallist.append(val)
        return max(allvallist) + 2


    ''' Variable declaration '''
    dictNumPolicy = dict() # Number of choosing policy p
    dictlistNumPolicyArm = dict()
    dictdictlistPolicyData = dict()
    dictMu = dict()
    dictSk = dict()
    dictUpper = dict()

    dictZk = dict()
    nPolicy = len(listPolicy)

    listAvgLoss = list()
    listTFArmCorrect = list()
    listCummRegret = list()

    MatSumVal = np.zeros((nPolicy, nPolicy))

    cummRegret = 0
    sumLoss = 0

    uopt = np.max(listU)
    optpl = np.argmax(listU)
    for plidx in range(nPolicy):
        if TF_naive== True:
            dictMu[plidx] = np.mean(OBS['Y'])
            dictNumPolicy[plidx] = 0
            dictdictlistPolicyData[plidx] = dict()
            dictdictlistPolicyData[plidx]['X'] = list(OBS['RXASP'])
            dictdictlistPolicyData[plidx]['Y'] = list(OBS['Y'])
            dictdictlistPolicyData[plidx]['Z'] = np.array(OBS[['SEX','RCONSC']])

        else:
            dictNumPolicy[plidx] = 0
            dictMu[plidx] = 0
            dictdictlistPolicyData[plidx] = dict()
            dictdictlistPolicyData[plidx]['X'] = []
            dictdictlistPolicyData[plidx]['Y'] = []
            dictdictlistPolicyData[plidx]['Z'] = []
        dictSk[plidx] = 0
        dictUpper[plidx] = 0
        dictZk[plidx] = 0
        dictlistNumPolicyArm[plidx] = [0, 0]


    ''' Before start'''
    # Compute the M matrix
    Mmat = ComputeMatM(listPolicy, listPz)
    maxcut = ComputeMaxCut(listPolicy,listPz,Mmat)
    maxcut += 10

    # If naive
    if TF_naive == True:
        t = 0
        for x,y,z in zip(list(OBS['RXASP']), list(OBS['Y']), np.array(OBS[['SEX','RCONSC']])):
            plidxChosen = np.random.choice(nPolicy)
            t += 1
            for k in range(nPolicy):
                MatSumVal[k][plidxChosen] += ComputeVal(xjs=int(x), yjs=y, zjs=z, k=k, j=plidxChosen,
                                                            listPolicy=listPolicy, M=Mmat, t=t)

    ''' Initial pulling by random choice'''
    t = 1
    # Observe zj
    s, a, r, x, y = EXP.sample(1).values[0]
    z = []
    zObs = np.array([s,r])
    # Choose random expert j
    plidxChosen = np.random.choice(nPolicy)
    pl = listPolicy[plidxChosen] # randomly chosen policy
    # Play an arm from pl (xj)
    armChosen = PullArmFromPl(pl, zObs)
    # Receive rewards yj
    row_cond = (EXP['RXASP']==armChosen) & (EXP['SEX']==zObs[0]) & (EXP['RCONSC'] == zObs[1])
    s, a, r, x, y = EXP[row_cond].sample(1).values[0]
    reward = y

    # Update information
    dictNumPolicy, dictlistNumPolicyArm, dictdictlistPolicyData = \
        UpdateAfterArm(dictNumPolicy, dictlistNumPolicyArm, dictdictlistPolicyData, plidxChosen, armChosen, zObs, reward)

    for k in range(nPolicy):
        # Zkt = ComputeZkt(dictNumPolicy, nPolicy, Mmat, k)
        MatSumVal[k][plidxChosen] += ComputeVal(xjs=armChosen, yjs=reward, zjs=zObs, k=k, j=plidxChosen,
                                                listPolicy=listPolicy, M=Mmat, t=t)

    for k in range(nPolicy):
        dictZk[k] = ComputeZkt(dictNumPolicy,nPolicy,Mmat,k)
        if TF_naive == True:
            dictMu[k] = 0
            for j in range(nPolicy):
                dictMu[k] += MatSumVal[k][j]
            dictMu[k] /= dictZk[k]
        else:
            dictMu[k] = ComputeMu(dictdictlistPolicyData,dictNumPolicy,listPolicy,Mmat,k,t)
        dictSk[k] = ComputeSk(dictNumPolicy,nPolicy,Mmat,k,t)
        dictUpper[k] = dictMu[k] + dictSk[k]
        if TF_causal:
            dictUpper[k] = max(min(dictUpper[k],listHB[k]),listLB[k])

    cummRegret += uopt - listU[plidxChosen]
    listCummRegret.append(cummRegret)
    if plidxChosen == optpl:
        listTFArmCorrect.append(1)
        sumLoss += 0
        avgLoss = sumLoss / t
        listAvgLoss.append(avgLoss)
    else:
        listTFArmCorrect.append(0)
        sumLoss += 1
        avgLoss = sumLoss / t
        listAvgLoss.append(avgLoss)

    ''' Play! '''
    for t in range(2, numRound+1):
        # Observe Z
        s, a, r, x, y = EXP.sample(1).values[0]
        zObs = np.array([s, r])

        # Choose policy
        plidxChosen = FindMaxKey(dictUpper)
        pl = listPolicy[plidxChosen]  # randomly chosen policy
        # Play an arm from pl (xj)
        armChosen = PullArmFromPl(pl, zObs)
        # Receive rewards
        row_cond = (EXP['RXASP']==armChosen) & (EXP['SEX']==zObs[0]) & (EXP['RCONSC'] == zObs[1])
        s, a, r, x, y = EXP[row_cond].sample(1).values[0]
        reward = y

        # Update information
        dictNumPolicy, dictlistNumPolicyArm, dictdictlistPolicyData = \
            UpdateAfterArm(dictNumPolicy, dictlistNumPolicyArm, dictdictlistPolicyData, plidxChosen, armChosen, zObs,
                           reward)
        for k in range(nPolicy):
            # Zkt = ComputeZkt(dictNumPolicy, nPolicy, Mmat, k)
            MatSumVal[k][plidxChosen] += ComputeVal(xjs=armChosen, yjs=reward, zjs=zObs, k=k, j=plidxChosen,
                                                    listPolicy=listPolicy, M=Mmat, t=t)

        for k in range(nPolicy):
            dictZk[k] = ComputeZkt(dictNumPolicy, nPolicy, Mmat, k)
            if t <= maxcut and TF_naive == False:
                dictMu[k] = ComputeMu(dictdictlistPolicyData, dictNumPolicy, listPolicy, Mmat, k, t)
            elif t <= maxcut and TF_naive == True:
                dictMu[k] = ComputeMu(dictdictlistPolicyData, dictNumPolicy, listPolicy, Mmat, k, t)
            elif t > maxcut or (t > maxcut and TF_naive == True):
                dictMu[k] = 0
                for j in range(nPolicy):
                    dictMu[k] += MatSumVal[k][j]
                dictMu[k] /= dictZk[k]
            dictSk[k] = ComputeSk(dictNumPolicy, nPolicy, Mmat, k, t)
            dictUpper[k] = dictMu[k] + dictSk[k]
            if TF_causal:
                dictUpper[k] = max(min(dictUpper[k],listHB[k]),listLB[k])

        cummRegret += uopt - listU[plidxChosen]
        listCummRegret.append(cummRegret)
        if plidxChosen== optpl:
            listTFArmCorrect.append(1)
            sumLoss += 0
            avgLoss = sumLoss / t
            listAvgLoss.append(avgLoss)
        else:
            listTFArmCorrect.append(0)
            sumLoss += 1
            avgLoss = sumLoss / t
            listAvgLoss.append(avgLoss)

    return [np.asarray(listTFArmCorrect), np.asarray(listCummRegret)]

def Pickle2Mat(dir_mat, mat_name, mydata):
    import scipy.io
    scipy.io.savemat(dir_mat, mdict={mat_name: mydata})


if __name__ == "__main__":
    numRound = 5000
    numSim = 10
    ''' Data Load '''
    EXP = pickle.load(open('sim_instance/EXP.pkl','rb'))
    OBS = pickle.load(open('sim_instance/OBS.pkl','rb'))

    TF_KLUCB = False
    TF_Best = False
    TF_DUCB = True

    TF_plot = True

    if TF_KLUCB:
        listHB = pickle.load(open('sim_instance/listHB.pkl','rb'))
        listLB = pickle.load(open('sim_instance/listLB.pkl', 'rb'))
        listHB100 = pickle.load(open('sim_instance/listHB100.pkl', 'rb'))
        listLB100 = pickle.load(open('sim_instance/listLB100.pkl', 'rb'))
        listU = pickle.load(open('sim_instance/listU.pkl','rb'))
        listnOBS = pickle.load(open('sim_instance/listnOBS.pkl','rb'))
        listUobs = pickle.load(open('sim_instance/listUobs.pkl','rb'))
        print('KLUCB: Case 2?', CheckCase2(np.min(listHB), np.max(listU)))

        ''' RUN KLUCB '''
        print('-'*30,'KLUCB','-'*30)
        arrayTFArmCorrect = np.array([0] * numRound)
        arrayCummRegret = np.array([0] * numRound)
        MatTFArmCorrect = np.zeros((numSim, numRound))
        MatCummRegret = np.zeros((numSim, numRound))

        for k in range(numSim):
            print('KLUCB',k)
            listTFArmCorrect, listCummRegret = RunKLUCB(numRound, EXP, listHB, listLB, listU, TF_causal=False, TF_naive=False, listnOBS=listnOBS, listUobs=listUobs)
            MatTFArmCorrect[k,:] = listTFArmCorrect
            MatCummRegret[k,:] = listCummRegret
            arrayTFArmCorrect = arrayTFArmCorrect + listTFArmCorrect
            arrayCummRegret = arrayCummRegret + listCummRegret

        MeanTFArmCorrect = arrayTFArmCorrect / numSim
        MeanCummRegret = arrayCummRegret / numSim
        pickle.dump(MeanTFArmCorrect,open('Result/OPT_KLUCB.pkl','wb'))
        pickle.dump(MeanCummRegret, open('Result/REG_KLUCB.pkl', 'wb'))
        pickle.dump(MatTFArmCorrect, open('Result/MAT_OPT_KLUCB.pkl','wb'))
        pickle.dump(MatCummRegret, open('Result/MAT_REG_KLUCB.pkl', 'wb'))

        ''' RUN C-KLUCB '''
        print('-' * 30, 'C-KLUCB', '-' * 30)
        arrayTFArmCorrect = np.array([0] * numRound)
        arrayCummRegret = np.array([0] * numRound)
        MatTFArmCorrect = np.zeros((numSim, numRound))
        MatCummRegret = np.zeros((numSim, numRound))
        for k in range(numSim):
            print('C-KLUCB', k)
            listTFArmCorrect, listCummRegret = RunKLUCB(numRound, EXP, listHB, listLB, listU, TF_causal=True,
                                                        TF_naive=False, listnOBS=listnOBS, listUobs=listUobs)
            MatTFArmCorrect[k, :] = listTFArmCorrect
            MatCummRegret[k, :] = listCummRegret
            arrayTFArmCorrect = arrayTFArmCorrect + listTFArmCorrect
            arrayCummRegret = arrayCummRegret + listCummRegret

        MeanTFArmCorrect = arrayTFArmCorrect / numSim
        MeanCummRegret = arrayCummRegret / numSim
        pickle.dump(MeanTFArmCorrect, open('Result/OPT_CKLUCB.pkl', 'wb'))
        pickle.dump(MeanCummRegret, open('Result/REG_CKLUCB.pkl', 'wb'))
        pickle.dump(MatTFArmCorrect, open('Result/MAT_OPT_CKLUCB.pkl', 'wb'))
        pickle.dump(MatCummRegret, open('Result/MAT_REG_CKLUCB.pkl', 'wb'))

        ''' RUN KLUCB-'''
        print('-' * 30, 'KLUCB-', '-' * 30)
        arrayTFArmCorrect = np.array([0] * numRound)
        arrayCummRegret = np.array([0] * numRound)
        MatTFArmCorrect = np.zeros((numSim, numRound))
        MatCummRegret = np.zeros((numSim, numRound))
        for k in range(numSim):
            print('KLUCB-', k)
            listTFArmCorrect, listCummRegret = RunKLUCB(numRound, EXP, listHB, listLB, listU, TF_causal=False,
                                                        TF_naive=True, listnOBS=[len(OBS),len(OBS)], listUobs=listUobs)
            MatTFArmCorrect[k, :] = listTFArmCorrect
            MatCummRegret[k, :] = listCummRegret
            arrayTFArmCorrect = arrayTFArmCorrect + listTFArmCorrect
            arrayCummRegret = arrayCummRegret + listCummRegret

        MeanTFArmCorrect = arrayTFArmCorrect / numSim
        MeanCummRegret = arrayCummRegret / numSim
        pickle.dump(MeanTFArmCorrect, open('Result/OPT_KLUCB-.pkl', 'wb'))
        pickle.dump(MeanCummRegret, open('Result/REG_KLUCB-.pkl', 'wb'))
        pickle.dump(MatTFArmCorrect, open('Result/MAT_OPT_KLUCB-.pkl', 'wb'))
        pickle.dump(MatCummRegret, open('Result/MAT_REG_KLUCB-.pkl', 'wb'))

        ''' RUN C-KLUCB-'''
        print('-' * 30, 'C-KLUCB-100', '-' * 30)
        arrayTFArmCorrect = np.array([0] * numRound)
        arrayCummRegret = np.array([0] * numRound)
        MatTFArmCorrect = np.zeros((numSim, numRound))
        MatCummRegret = np.zeros((numSim, numRound))
        for k in range(numSim):
            print('C-KLUCB-100', k)
            listTFArmCorrect, listCummRegret = RunKLUCB(numRound, EXP, listHB100, listLB100, listU, TF_causal=True,
                                                        TF_naive=False, listnOBS=[100,100], listUobs=listUobs)
            MatTFArmCorrect[k, :] = listTFArmCorrect
            MatCummRegret[k, :] = listCummRegret
            arrayTFArmCorrect = arrayTFArmCorrect + listTFArmCorrect
            arrayCummRegret = arrayCummRegret + listCummRegret

        MeanTFArmCorrect = arrayTFArmCorrect / numSim
        MeanCummRegret = arrayCummRegret / numSim
        pickle.dump(MeanTFArmCorrect, open('Result/OPT_CKLUCB-100.pkl', 'wb'))
        pickle.dump(MeanCummRegret, open('Result/REG_CKLUCB-100.pkl', 'wb'))
        pickle.dump(MatTFArmCorrect, open('Result/MAT_OPT_CKLUCB-100.pkl', 'wb'))
        pickle.dump(MatCummRegret, open('Result/MAT_REG_CKLUCB-100.pkl', 'wb'))

        if TF_plot == True:
            OPT_KLUCB = pickle.load(open('Result/OPT_KLUCB.pkl','rb'))
            REG_KLUCB = pickle.load(open('Result/REG_KLUCB.pkl', 'rb'))
            MAT_KLUCB = pickle.load(open('Result/MAT_REG_KLUCB.pkl','rb'))

            OPT_CKLUCB = pickle.load(open('Result/OPT_CKLUCB.pkl', 'rb'))
            REG_CKLUCB = pickle.load(open('Result/REG_CKLUCB.pkl', 'rb'))
            MAT_CKLUCB = pickle.load(open('Result/MAT_REG_CKLUCB.pkl', 'rb'))

            OPT_KLUCB_N = pickle.load(open('Result/OPT_KLUCB-.pkl','rb'))
            REG_KLUCB_N = pickle.load(open('Result/REG_KLUCB-.pkl', 'rb'))
            MAT_KLUCB_N = pickle.load(open('Result/MAT_REG_KLUCB-.pkl', 'rb'))

            OPT_CKLUCB_100 = pickle.load(open('Result/OPT_CKLUCB-100.pkl','rb'))
            REG_CKLUCB_100 = pickle.load(open('Result/REG_CKLUCB-100.pkl', 'rb'))
            MAT_CKLUCB_100 = pickle.load(open('Result/MAT_REG_CKLUCB-100.pkl', 'rb'))

            xDomain = np.arange(numRound)
            alpha = 0.05
            colorAlpha = 0.1
            plt.title('KLUCB')
            plt.plot(xDomain,REG_KLUCB,'red',label='KLUCB')
            plt.fill_between(xDomain, np.percentile(MAT_KLUCB, q=100 * alpha, axis=0),
                             np.percentile(MAT_KLUCB, q=100 * (1 - alpha), axis=0), facecolor='red', alpha=colorAlpha)
            plt.plot(xDomain,REG_CKLUCB,'blue',label='C-KLUCB')
            plt.fill_between(xDomain, np.percentile(MAT_CKLUCB, q=100 * alpha, axis=0),
                             np.percentile(MAT_CKLUCB, q=100 * (1 - alpha), axis=0), facecolor='blue', alpha=colorAlpha)
            plt.plot(xDomain,REG_KLUCB_N,'green',label='KLUCB-')
            plt.fill_between(xDomain, np.percentile(MAT_KLUCB_N, q=100 * alpha, axis=0),
                             np.percentile(MAT_KLUCB_N, q=100 * (1 - alpha), axis=0), facecolor='green', alpha=colorAlpha)
            plt.plot(xDomain,REG_CKLUCB_100,'orange',label='C-KLUCB-100')
            plt.fill_between(xDomain, np.percentile(MAT_CKLUCB_100, q=100 * alpha, axis=0),
                             np.percentile(MAT_CKLUCB_100, q=100 * (1 - alpha), axis=0), facecolor='orange', alpha=colorAlpha)
            plt.ylim(0-3,max(max(REG_KLUCB),max(REG_CKLUCB_100)) + 20)
            plt.legend()

        # plt.plot(OPT_KLUCB, label='KLUCB')
        # plt.plot(OPT_CKLUCB, label='C-KLUCB')
        # plt.plot(OPT_KLUCB_N, label='KLUCB-')
        # plt.plot(OPT_CKLUCB_100, label='C-KLUCB-100')

    if TF_Best:
        listHB = pickle.load(open('sim_instance/listHB.pkl', 'rb'))
        listLB = pickle.load(open('sim_instance/listLB.pkl', 'rb'))
        listHB100 = pickle.load(open('sim_instance/listHB100.pkl', 'rb'))
        listLB100 = pickle.load(open('sim_instance/listLB100.pkl', 'rb'))
        listU = pickle.load(open('sim_instance/listU.pkl', 'rb'))
        listnOBS = pickle.load(open('sim_instance/listnOBS.pkl', 'rb'))
        listUobs = pickle.load(open('sim_instance/listUobs.pkl', 'rb'))
        print('BestArm: Case 2 (true)?', CheckCase2BestArm(listHB, listU))

        ''' RUN BestArm '''
        print('-' * 30, 'Best Arm', '-' * 30)
        array_t = np.array([0] * numSim)
        array_ht = np.array([0] * numSim)

        for k in range(numSim):
            print('BestArm', k)
            t, ht, _ = RunBestArm(EXP, listLB, listU, listHB, TF_causal=False, TF_naive=False, listnOBS = listnOBS, listUobs=listUobs)
            array_t[k] = t
            array_ht[k] = ht

        pickle.dump(array_t, open('Result/t_BestArm.pkl', 'wb'))
        pickle.dump(array_ht, open('Result/ht_BestArm.pkl', 'wb'))

        ''' RUN C-BestArm '''
        print('-' * 30, 'C-Best Arm', '-' * 30)
        array_t = np.array([0] * numSim)
        array_ht = np.array([0] * numSim)

        for k in range(numSim):
            print('C-BestArm', k)
            t, ht, _ = RunBestArm(EXP, listLB, listU, listHB, TF_causal=True, TF_naive=False, listnOBS = listnOBS, listUobs=listUobs)
            array_t[k] = t
            array_ht[k] = ht

        pickle.dump(array_t, open('Result/t_C-BestArm.pkl', 'wb'))
        pickle.dump(array_ht, open('Result/ht_C-BestArm.pkl', 'wb'))

        ''' RUN BestArm-Naive '''
        print('-' * 30, 'Best Arm-', '-' * 30)
        array_t = np.array([0] * numSim)
        array_ht = np.array([0] * numSim)

        for k in range(numSim):
            print('BestArm-N', k)
            t, ht, _ = RunBestArm(EXP, listLB, listU, listHB, TF_causal=False, TF_naive=True, listnOBS = [1000,1000], listUobs=listUobs)
            array_t[k] = t
            array_ht[k] = ht

        pickle.dump(array_t, open('Result/t_BestArm-.pkl', 'wb'))
        pickle.dump(array_ht, open('Result/ht_BestArm-.pkl', 'wb'))

        ''' RUN C-BestArm-100'''
        print('-' * 30, 'C-Best Arm-100', '-' * 30)
        array_t = np.array([0] * numSim)
        array_ht = np.array([0] * numSim)

        for k in range(numSim):
            print('C-BestArm-100', k)
            t, ht, _ = RunBestArm(EXP, listLB100, listU, listHB100, TF_causal=True, TF_naive=False,
                                  listnOBS=listUobs, listUobs=listUobs)
            array_t[k] = t
            array_ht[k] = ht

        pickle.dump(array_t, open('Result/t_C-BestArm-100.pkl', 'wb'))
        pickle.dump(array_ht, open('Result/ht_C-BestArm-100.pkl', 'wb'))

        if TF_plot == True:
            array_t = pickle.load(open('Result/t_BestArm.pkl', 'rb'))
            array_ht = pickle.load(open('Result/ht_BestArm.pkl', 'rb'))
            array_tC = pickle.load(open('Result/t_C-BestArm.pkl', 'rb'))
            array_htC = pickle.load(open('Result/ht_C-BestArm.pkl', 'rb'))
            array_tN = pickle.load(open('Result/t_BestArm-.pkl', 'rb'))
            array_htN = pickle.load(open('Result/ht_BestArm-.pkl', 'rb'))
            array_tCN = pickle.load(open('Result/t_C-BestArm-100.pkl', 'rb'))
            array_htCN = pickle.load(open('Result/ht_C-BestArm-100.pkl', 'rb'))

            print('LUCB',np.mean(array_t),np.mean(array_ht))
            print('C-LUCB',np.mean(array_tC), np.mean(array_htC))
            print('LUCB-N',np.mean(array_tN), np.mean(array_htN))
            print('C-LUCB-100',np.mean(array_tCN), np.mean(array_htCN))

            data = {'LUCB':array_t, 'C-LUCB':array_tC, 'LUCB-':array_tN, 'C-LUCB-100':array_tCN}
            data = pd.DataFrame(data)
            plt.title("Best Arm")
            data.boxplot(column=['LUCB','C-LUCB','LUCB-','C-LUCB-100'],grid=False)

    if TF_DUCB:
        listHB = pickle.load(open('sim_instance/listHBpl.pkl', 'rb'))
        listLB = pickle.load(open('sim_instance/listLBpl.pkl', 'rb'))
        listHB100 = pickle.load(open('sim_instance/listHBpl100.pkl', 'rb'))
        listLB100 = pickle.load(open('sim_instance/listLBpl100.pkl', 'rb'))
        listU = pickle.load(open('sim_instance/listUpl.pkl', 'rb'))
        listPz = pickle.load(open('sim_instance/listPz.pkl','rb'))
        # listPz = [len(EXP[EXP['SEX']==z]['Y'])/len(EXP) for z in [0,1]]
        #
        # # Define policies
        # pl1 = lambda z: 0.01 if z == 0 else 0.02
        # pl2 = lambda z: 0.05 if z == 0 else 0.1
        # pl3 = lambda z: 0.97 if z == 0 else 0.99
        # pl4 = lambda z: 0.1 if z == 0 else 0.05
        # listPolicy = [pl1, pl2, pl3, pl4]

        low_prob = 0.01
        high_prob = 0.99
        mid_prob = 0.5


        def pl1(z0, z1):
            if z0 == 0 and z1 == 0:
                return high_prob
            elif z0 == 0 and z1 == 1:
                return low_prob
            elif z0 == 0 and z1 == 2:
                return low_prob
            elif z0 == 1 and z1 == 0:
                return high_prob
            elif z0 == 1 and z1 == 1:
                return low_prob
            elif z0 == 1 and z1 == 2:
                return low_prob


        def pl2(z0, z1):
            if z0 == 0 and z1 == 0:
                return high_prob
            elif z0 == 0 and z1 == 1:
                return low_prob
            elif z0 == 0 and z1 == 2:
                return low_prob
            elif z0 == 1 and z1 == 0:
                return low_prob
            elif z0 == 1 and z1 == 1:
                return low_prob
            elif z0 == 1 and z1 == 2:
                return low_prob


        def pl3(z0, z1):
            if z0 == 0 and z1 == 0:
                return low_prob
            elif z0 == 0 and z1 == 1:
                return high_prob
            elif z0 == 0 and z1 == 2:
                return low_prob
            elif z0 == 1 and z1 == 0:
                return low_prob
            elif z0 == 1 and z1 == 1:
                return high_prob
            elif z0 == 1 and z1 == 2:
                return low_prob


        def pl4(z0, z1):
            if z0 == 0 and z1 == 0:
                return low_prob
            elif z0 == 0 and z1 == 1:
                return low_prob
            elif z0 == 0 and z1 == 2:
                return low_prob
            elif z0 == 1 and z1 == 0:
                return low_prob
            elif z0 == 1 and z1 == 1:
                return high_prob
            elif z0 == 1 and z1 == 2:
                return high_prob


        def pl5(z0, z1):
            if z0 == 0 and z1 == 0:
                return low_prob
            elif z0 == 0 and z1 == 1:
                return low_prob
            elif z0 == 0 and z1 == 2:
                return low_prob
            elif z0 == 1 and z1 == 0:
                return low_prob
            elif z0 == 1 and z1 == 1:
                return low_prob
            elif z0 == 1 and z1 == 2:
                return low_prob


        def pl6(z0, z1):
            if z0 == 0 and z1 == 0:
                return mid_prob
            elif z0 == 0 and z1 == 1:
                return mid_prob
            elif z0 == 0 and z1 == 2:
                return mid_prob
            elif z0 == 1 and z1 == 0:
                return mid_prob
            elif z0 == 1 and z1 == 1:
                return mid_prob
            elif z0 == 1 and z1 == 2:
                return mid_prob


        listPolicy = [pl1, pl2, pl3, pl4, pl5, pl6]

        for plidx in range(len(listPolicy)):
            print('Case 2', 'Policy', plidx, CheckCase2(listHB[plidx], np.max(listU)))

        '''RUN DUCB'''
        print('-' * 30, 'DUCB', '-' * 30)
        arrayTFArmCorrect = np.array([0] * numRound)
        arrayCummRegret = np.array([0] * numRound)
        MatTFArmCorrect = np.zeros((numSim, numRound))
        MatCummRegret = np.zeros((numSim, numRound))
        for k in range(numSim):
            print('DUCB', k)
            listTFArmCorrect, listCummRegret = RunDUCB(numRound, EXP, OBS, listPz, listPolicy, listHB, listLB, listU, TF_causal=False, TF_naive=False)
            MatTFArmCorrect[k, :] = listTFArmCorrect
            MatCummRegret[k, :] = listCummRegret
            arrayTFArmCorrect = arrayTFArmCorrect + listTFArmCorrect
            arrayCummRegret = arrayCummRegret + listCummRegret

        MeanTFArmCorrect = arrayTFArmCorrect / numSim
        MeanCummRegret = arrayCummRegret / numSim
        pickle.dump(MeanTFArmCorrect, open('Result/OPT_DUCB.pkl', 'wb'))
        pickle.dump(MeanCummRegret, open('Result/REG_DUCB.pkl', 'wb'))
        pickle.dump(MatTFArmCorrect, open('Result/MAT_OPT_DUCB.pkl', 'wb'))
        pickle.dump(MatCummRegret, open('Result/MAT_REG_DUCB.pkl', 'wb'))

        '''RUN C-DUCB'''
        print('-' * 30, 'C-DUCB', '-' * 30)
        arrayTFArmCorrect = np.array([0] * numRound)
        arrayCummRegret = np.array([0] * numRound)
        MatTFArmCorrect = np.zeros((numSim, numRound))
        MatCummRegret = np.zeros((numSim, numRound))
        for k in range(numSim):
            print('C-DUCB', k)
            listTFArmCorrect, listCummRegret = RunDUCB(numRound, EXP, OBS, listPz, listPolicy, listHB, listLB, listU,
                                                       TF_causal=True, TF_naive=False)
            MatTFArmCorrect[k, :] = listTFArmCorrect
            MatCummRegret[k, :] = listCummRegret
            arrayTFArmCorrect = arrayTFArmCorrect + listTFArmCorrect
            arrayCummRegret = arrayCummRegret + listCummRegret

        MeanTFArmCorrect = arrayTFArmCorrect / numSim
        MeanCummRegret = arrayCummRegret / numSim
        pickle.dump(MeanTFArmCorrect, open('Result/OPT_CDUCB.pkl', 'wb'))
        pickle.dump(MeanCummRegret, open('Result/REG_CDUCB.pkl', 'wb'))
        pickle.dump(MatTFArmCorrect, open('Result/MAT_OPT_CDUCB.pkl', 'wb'))
        pickle.dump(MatCummRegret, open('Result/MAT_REG_CDUCB.pkl', 'wb'))

        '''RUN DUCB-N'''
        print('-' * 30, 'DUCB-N', '-' * 30)
        arrayTFArmCorrect = np.array([0] * numRound)
        arrayCummRegret = np.array([0] * numRound)
        MatTFArmCorrect = np.zeros((numSim, numRound))
        MatCummRegret = np.zeros((numSim, numRound))
        for k in range(numSim):
            print('DUCB-N', k)
            listTFArmCorrect, listCummRegret = RunDUCB(numRound, EXP, OBS, listPz, listPolicy, listHB, listLB, listU,
                                                       TF_causal=False, TF_naive=True)
            MatTFArmCorrect[k, :] = listTFArmCorrect
            MatCummRegret[k, :] = listCummRegret
            arrayTFArmCorrect = arrayTFArmCorrect + listTFArmCorrect
            arrayCummRegret = arrayCummRegret + listCummRegret

        MeanTFArmCorrect = arrayTFArmCorrect / numSim
        MeanCummRegret = arrayCummRegret / numSim
        pickle.dump(MeanTFArmCorrect, open('Result/OPT_DUCB-.pkl', 'wb'))
        pickle.dump(MeanCummRegret, open('Result/REG_DUCB-.pkl', 'wb'))
        pickle.dump(MatTFArmCorrect, open('Result/MAT_OPT_DUCB-.pkl', 'wb'))
        pickle.dump(MatCummRegret, open('Result/MAT_REG_DUCB-.pkl', 'wb'))

        '''RUN C-DUCB-100'''
        print('-' * 30, 'C-DUCB-100', '-' * 30)
        arrayTFArmCorrect = np.array([0] * numRound)
        arrayCummRegret = np.array([0] * numRound)
        MatTFArmCorrect = np.zeros((numSim, numRound))
        MatCummRegret = np.zeros((numSim, numRound))
        for k in range(numSim):
            print('C-DUCB-100', k)
            listTFArmCorrect, listCummRegret = RunDUCB(numRound, EXP, OBS, listPz, listPolicy, listHB100, listLB100, listU,
                                                       TF_causal=True, TF_naive=False)
            MatTFArmCorrect[k, :] = listTFArmCorrect
            MatCummRegret[k, :] = listCummRegret
            arrayTFArmCorrect = arrayTFArmCorrect + listTFArmCorrect
            arrayCummRegret = arrayCummRegret + listCummRegret

        MeanTFArmCorrect = arrayTFArmCorrect / numSim
        MeanCummRegret = arrayCummRegret / numSim
        pickle.dump(MeanTFArmCorrect, open('Result/OPT_CDUCB-100.pkl', 'wb'))
        pickle.dump(MeanCummRegret, open('Result/REG_CDUCB-100.pkl', 'wb'))
        pickle.dump(MatTFArmCorrect, open('Result/MAT_OPT_CDUCB-100.pkl', 'wb'))
        pickle.dump(MatCummRegret, open('Result/MAT_REG_CDUCB-100.pkl', 'wb'))

        if TF_plot == True:
            OPT_DUCB = pickle.load(open('Result/OPT_DUCB.pkl', 'rb'))
            REG_DUCB = pickle.load(open('Result/REG_DUCB.pkl', 'rb'))
            MAT_DUCB = pickle.load(open('Result/MAT_REG_DUCB.pkl','rb'))

            OPT_CDUCB = pickle.load(open('Result/OPT_CDUCB.pkl', 'rb'))
            REG_CDUCB = pickle.load(open('Result/REG_CDUCB.pkl', 'rb'))
            MAT_CDUCB = pickle.load(open('Result/MAT_REG_CDUCB.pkl', 'rb'))

            OPT_DUCB_N = pickle.load(open('Result/OPT_DUCB-.pkl', 'rb'))
            REG_DUCB_N = pickle.load(open('Result/REG_DUCB-.pkl', 'rb'))
            MAT_DUCB_N = pickle.load(open('Result/MAT_REG_DUCB-.pkl', 'rb'))

            OPT_CDUCB_100 = pickle.load(open('Result/OPT_CDUCB-100.pkl', 'rb'))
            REG_CDUCB_100 = pickle.load(open('Result/REG_CDUCB-100.pkl', 'rb'))
            MAT_CDUCB_100 = pickle.load(open('Result/MAT_REG_CDUCB-100.pkl', 'rb'))

            xDomain = np.arange(numRound)
            alpha = 0.05
            colorAlpha = 0.1
            plt.title('D-UCB')
            plt.plot(xDomain,REG_DUCB, 'red', label='D-UCB')
            plt.fill_between(xDomain,np.percentile(MAT_DUCB,q=100*alpha,axis=0),
                             np.percentile(MAT_DUCB,q=100*(1-alpha),axis=0),facecolor='red',alpha=colorAlpha)
            plt.plot(xDomain,REG_CDUCB, 'blue', label='C-D-UCB')
            plt.fill_between(xDomain, np.percentile(MAT_CDUCB, q=100 * alpha, axis=0),
                             np.percentile(MAT_CDUCB, q=100 * (1 - alpha), axis=0), facecolor='blue', alpha=colorAlpha)
            plt.plot(xDomain,REG_DUCB_N,'green', label='D-UCB-')
            plt.fill_between(xDomain, np.percentile(MAT_DUCB_N, q=100 * alpha, axis=0),
                             np.percentile(MAT_DUCB_N, q=100 * (1 - alpha), axis=0), facecolor='green', alpha=colorAlpha)
            plt.plot(xDomain,REG_CDUCB_100,'orange', label='C-D-UCB-100')
            plt.fill_between(xDomain, np.percentile(MAT_CDUCB_100, q=100 * alpha, axis=0),
                             np.percentile(MAT_CDUCB_100, q=100 * (1 - alpha), axis=0), facecolor='orange', alpha=colorAlpha)
            plt.ylim(0 - 3, max(max(REG_DUCB), max(REG_CDUCB_100)) + 20)
            plt.legend()



# def FunEval(pl, z):
#     z0, z1 = z
#     pl_1z = pl(z0,z1)
#     return [1 - pl_1z, pl_1z]
#
#
# def ComputeMatM(listPolicy, listPz):
#     def ComputeMpq(p, q):
#         zPossible = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
#
#         def f1(x):
#             return x * np.exp(x - 1) - 1
#
#         sumProb = 0
#         for zidx in range(len(listPz)):
#             Pz = listPz[zidx]
#             z = zPossible[zidx]
#             for x in [0, 1]:
#                 pxz = FunEval(p, z)
#                 pxz = pxz[x]
#                 qxz = FunEval(q, z)
#                 qxz = qxz[x]
#                 sumProb += (f1(pxz / qxz) * qxz * Pz)
#         return (1 + np.log(1 + sumProb))
#
#     N_poly = len(listPolicy)
#     poly_idx_iter = list(itertools.product(list(range(N_poly)), list(range(N_poly))))
#     M_mat = np.zeros((N_poly, N_poly))
#     for k, j in poly_idx_iter:
#         # if k != j:
#         pk = listPolicy[k]
#         pj = listPolicy[j]
#         M_mat[k, j] = ComputeMpq(pk, pj)
#     return M_mat
# #
# allvallist = []
# allPossible = list(itertools.combinations(range(len(listPolicy)), 2))
# Mmat = ComputeMatM(listPolicy, listPz)
# #
# for x in [0,1]:
#     for z0 in [0,1]:
#         for z1 in [0,1,2]:
#             for k,j in allPossible:
#                 plk = listPolicy[k]
#                 plj = listPolicy[j]
#                 Mkj = Mmat[k,j]
#
#                 plkval = FunEval(plk,[z0,z1])
#                 plkval_x = plkval[x]
#
#                 pljval = FunEval(plj, [z0,z1])
#                 pljval_x = pljval[x]
#
#                 val = np.exp((plkval_x / pljval_x)*(1/(2*Mkj)))
#
#                 allvallist.append(val)
#
#
# listPxz = []
# for z0 in [0,1]:
#     for z1 in [0,1,2]:
#         pxz = len(OBS[(OBS['SEX'] == z0) & (OBS['RCONSC'] == z1) & (OBS['RXASP']==1)]) / len(OBS)
#         listPxz.append(pxz)