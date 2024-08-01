import numpy as np
from scipy.optimize import root
import math
from scipy.stats import norm
from scipy import stats

def init_mu(prob, old = False):
    d = len(prob)
    prob = prob[0]/(prob[0] + prob[1:])
    if (old):
        return -norm.ppf(prob)
    else:
        return -math.sqrt(2) * norm.ppf(prob)

def softmax_colwise(Z):
    tmp = np.repeat(np.max(Z, axis=1), len(Z[0])).reshape((len(Z), -1))
    Zexp = np.exp(Z - tmp)
    Zexp_rsum = np.sum(Zexp, axis=1)
    r = Zexp / np.repeat(Zexp_rsum, len(Zexp[0])).reshape((len(Zexp), -1))
    return r


def E_softmax_MC(mu, beta, n_MC = 2000, seed = None, old = False):
    # if (old):
    #     return E_softmax_MC_old(mu, beta, n_MC, seed)
    if isinstance(mu, np.ndarray):
        mu = mu.tolist()
    mu.insert(0, 0)             #u1 = 0
    if (seed is not None):
        np.random.seed(seed)    # set.seed(seed)
    d = len(mu)
    Z = np.random.normal(0, 1, size=(n_MC, d)) + np.tile(mu, (n_MC, 1)) # argmax(z+u) = pk, ���������㷨ģ��
    pi_x = softmax_colwise(Z * beta)                                    # �ز�������
    val = np.mean(pi_x, axis=0)              # val = colMeans(pi_x)
    Inkk = np.zeros(shape=(n_MC, d, d))      # Inkk = array(0, dim = c(n_MC, d, d))
    pi_x_1st = np.zeros(shape=(n_MC, d, d))  #     pi_x_1st = array(0, dim = c(n_MC, d, d))
    pi_x_2nd = np.zeros(shape=(n_MC, d, d)) #     pi_x_2nd = array(0, dim = c(n_MC, d, d))
    for i in range(n_MC):
        Inkk[i] = np.identity(d)
    for i in range(d):
        pi_x_1st[:, i, :] = pi_x
        pi_x_2nd[:, :, i] = pi_x
    jac = (Inkk - pi_x_1st) * pi_x_2nd
    jac = np.mean(jac, axis=0)*beta          #jac = apply(jac, c(2, 3), mean) * beta
    return dict(val = val, jac = jac)

def get_solve_mu(prob, beta, n_MC = 2000, seed = 11, old = False):
    # if (seed != None):
    #     np.random.seed(seed)
    # force(prob)
    # force(beta)
    # force(n_MC)
    # force(seed)
    # force(old)

    # betaĬ��Ϊ1000, n_MCĬ��Ϊ5000
    def f_val(mu):
        out = E_softmax_MC(mu, beta, n_MC=n_MC, seed=seed, old=old)
        val = out['val']
        return val[1:] - prob[1:]
    def f_jac(mu):
        out = E_softmax_MC(mu, beta, n_MC = n_MC, seed = seed, old = old)
        jac = out['jac']
        # if (old):       #
        #     jac = jac[1:, 1:]

        return jac[1:, 1:]
    return dict(f_val = f_val, f_jac = f_jac)

def solve_nominal_mu(prob, beta = 1000, n_MC = 10000, seed = 101, inits = None, eps = 1e-04, old = False):
    n_MC = [5000, 10000, 15000, 20000, 25000, 30000]
    best_precis = 100
    best_r = None

    mu0 = init_mu(prob, old = old)      # initalization
    d = len(mu0)                        # d = len(mu) - 1
    if (inits is None):                 #
        p_seq = np.arange(0, 5+1)
        l = len(p_seq)
        inits = np.tile(mu0, (l, 1)) #
        tmp = 2**p_seq
        tmp = np.tile(tmp, (d, 1)).T
        inits = inits / tmp
        inits[l-1,] = 0
    else:
        # if len(inits.shape) == 1:
        #     inits = inits.reshape((1,-1))
        p_seq = np.arange(0, 5 + 1)
        l = len(p_seq)
        inits = np.tile(inits[1:], (l, 1))  # ��mu0��������
        tmp = 2 ** p_seq
        tmp = np.tile(tmp, (d, 1)).T
        inits = inits / tmp
        inits[l - 1,] = 0
    l = len(inits)

    for m in n_MC:
        flist = get_solve_mu(prob=prob, beta=beta, n_MC=m,
                 seed=seed, old=old)    # get functions f_val and f_jac
        if (best_precis < 0.1):
            r = root(flist['f_val'], best_r['root'], jac=flist['f_jac'], method='hybr') #method='hybr') #
            #ifbest = r$estim.precis < best_precis
            estim_precis = np.sum(np.abs(r.fun))
            ifbest = estim_precis < best_precis
            assert(ifbest is not None)
            if (ifbest):
                best_precis = estim_precis
                best_r = {'root':r.x, 'fun':r.fun, 'estim_precis':estim_precis}

        if (best_precis < eps):     # precision < 0.0001, it's time to over
            break
        #
        for i in range(l):
            r = root(flist['f_val'], inits[i], jac= flist['f_jac'], method='hybr')#method='hybr')
            estim_precis = np.sum(np.abs(r.fun))
            ifbest = estim_precis < best_precis
            assert(ifbest != None)
            if ifbest:
                best_precis = estim_precis
                best_r = {'root':r.x, 'fun':r.fun, 'estim_precis':estim_precis}
            if (best_precis < eps):
                break
        if (best_precis < eps):
            break
    return best_r

def get_cat_mu(freq_list, beta = 1000, n_MC = 5000, seed = 101, eps = 1e-04, verbose = False, old = False, inits=None):
    '''
    :param freq_list: each fequency of value
    :param beta: default 1000
    :param n_MC: default 5000
    :param seed:
    :param eps:
    :param verbose:
    :param old:
    :return:
    '''
    #inits = None
    quiet_solve = True
    if (verbose):
        print("Starting categorical marginal estimation: ")
    d = sum([len(ii) for ii in freq_list])
    lf = len(freq_list)
    # if (old):
    #     mu = np.zeros((d - lf,))
    # else:
    mu = np.zeros((d,))
    start = 0
    # if (old):
    #     fs = np.zeros((d - lf,))
    # else:
    fs = np.zeros((d,))
    est_precis = np.zeros((lf,))
    init_tmp = None
    for i in range(len(freq_list)):
        # 1.estimate means of cat featured.
        if inits is not None:
            init_tmp = inits[start:start+len(freq_list[i])]
        out = solve_nominal_mu(freq_list[i], beta = beta, n_MC = n_MC, seed = seed, inits = init_tmp, eps = eps)
        ri = out
        mui = ri['root']
        index = np.arange(len(mui)) + start + 1
        mu[index] = mui                 # mu0 = 0
        fs[index] = ri['fun']
        est_precis[i] = ri['estim_precis']
        start = start + 1 + len(mui)
        if (verbose):
            print(f"{i} of {lf} categorical marginal estimations finished")
    if (any(est_precis > eps)):
        print("Some nominal mean estimation does not reach desired precision")
    return dict(mu = mu, f_root = fs, est_precis = est_precis)