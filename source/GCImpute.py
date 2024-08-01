import copy

import pandas as pd
import numpy as np
# from tools import set_seed
from collections import Counter
from scipy.stats import norm
from scipy.stats import truncnorm
import copy
import math

import statsmodels.api as sm
from scipy.stats import norm
from scipy import stats
import math

def cov2corr( A ):
    """
    covariance matrix to correlation matrix.
    """
    d = np.sqrt(A.diagonal())
    A = ((A.T/d).T)/d
    #A[ np.diag_indices(A.shape[0]) ] = np.ones( A.shape[0] )
    return A

#-1
ord_labels = {
    'heart': [2-1, 6-1, 9-1], 'creditg': [18, 19, 20, 1, 6, 7],
    'cmc': [5, 6, 9], 'abalone': [], 'credita': [1, 8, 9, 11], 'colic': [1, 2, 23, 7, 8, 10, 11, 15, 17]}


def call_gcimpute_nominal(X, cat_index,
                          # arg1
                          verbose = False, read_mu = False,
                          oracle_mu= False, oracle_cor = False,
                          trunc_method = 'Iterative', old=False, n_MI = 0, fast_MI=True,
                          # arg2
                          mu=None, corr=None, mu_loc=None, seed_num=None):
    '''
    ExtendGC
    :param X: Input
    :param cat_index: indexs for categorical features
    :param verbose:
    :param read_mu:
    :param oracle_mu:
    :param oracle_cor:
    :param trunc_method:
    :param old:
    :param n_MI:
    :param fast_MI:
    :param mu:
    :param corr:
    :param mu_loc:
    :param seed_num:
    :return:
    '''
    if oracle_mu:
        mu_use = mu
    else:
        mu_use = None

    if read_mu:
        assert(((mu_loc != None) and (seed_num != None)))  # stopifnot(! is.null(mu_loc) | ! is.null(seed_num) )
        readed = pd.read_csv(mu_loc, header=None)          # readed = as.matrix(read.csv(mu_loc, row.names = 1))
        mu_use = readed[seed_num, -1]
        m = np.isnan(mu_use)
        if any(m):
            mu_use = mu_use[~m]
        time_add = readed[seed_num, 1]
    else:
        time_add = 0

    if oracle_cor:
        corr_use = corr
    else:
        corr_use = None

    if verbose:
        print(f"is mu provided? {mu_use != None}")
        print(f"is corr provided? {corr_use != None}")

    # extend GC
    est = impute_nominal_cont_gc(X=X,
                                 cat_index=cat_index,
                                 mu=mu_use,
                                 corr=corr_use,
                                 verbose=verbose,
                                 trunc_method=trunc_method,
                                 n_MI=n_MI, fast_MI=fast_MI,
                                 old=old)

    return dict(Ximp=est['Ximp'], Ximp_MI=est['Ximp_MI'], cat_index=cat_index,
                est=est, time_add=time_add)


def get_cat_index_freq(X_cat):
    p_cat = len(X_cat[0])
    freq, nlevel = [], []
    for j in range(p_cat):     # 统计每个字段的频率
        count_res = Counter(X_cat[:, j][~np.isnan(X_cat[:, j])])
        sorted_index = np.argsort(list(count_res.keys()))
        keys = np.array(list(count_res.keys()))[sorted_index]
        values = np.array(list(count_res.values()))[sorted_index]
        freq.append(values / sum(values))
        nlevel.append(len(count_res))
    return {'freq':freq, 'nlevel':nlevel }

def get_cat_index_freq_with_Laplacian(X_cat, cat_range, eps=1):
    cat_range = list(cat_range.values())
    p_cat = len(X_cat[0])
    freq, nlevel = [], []
    for j in range(p_cat):  #
        nlevel.append(cat_range[j][1] - cat_range[j][0] + 1)
        freq_per = np.zeros(int(nlevel[j]))
        freq_per[:] = eps
        count_res = Counter(X_cat[:, j][~np.isnan(X_cat[:, j])])
        for k,v in count_res.items():
            freq_per[int(k-1)] += v
        freq.append(freq_per / sum(freq_per))  #
    return {'freq': freq, 'nlevel': nlevel}

def relabel(x, label):
  #stopifnot(check_cat_label(x))
  #x = as.factor(x)
  label = np.sort(label)
  x_labels = list(set(x))
  label_to_label = dict()
  for ii,xx in enumerate(x_labels):
    label_to_label[label[ii]] = (x == xx)
  for kk, vv in label_to_label.items():
    x[vv] = kk
  return x

def cat_to_integers(x):
    '''
    :param x: Input
    :return: relabeled input, such like[1, 3, 5, 10] -> [1,2,3,4], index starting from 1, due to features of R
    '''
    XX = x.copy()
    XX = np.unique(XX)
    XX = XX[~np.isnan(XX)]
    XX.sort()

    xlevels = []
    for i, v in enumerate(XX):
        x[x == v] = i+1
        xlevels.append(i+1)
    return dict(x=x, xlevels=xlevels)

def index_int_to_logi(index, l):
    out = np.zeros((l,))
    out = out.astype(bool)
    out[index] = True
    return out

def logi_to_index_int(index):
    return np.where(index)[0]


def get_solve_mu(prob, beta, n_MC = 2000, seed = 11, old = False):
    # if (seed != None):
    #     np.random.seed(seed)
    # force(prob)
    # force(beta)
    # force(n_MC)
    # force(seed)
    # force(old)

    # beta默认为1000, n_MC默认为5000
    def f_val(mu):
        out = E_softmax_MC(mu, beta, n_MC=n_MC, seed=seed, old=old)
        val = out['val']
        return val[1:] - prob[1:]
    def f_jac(mu):
        out = E_softmax_MC(mu, beta, n_MC = n_MC, seed = seed, old = old)
        jac = out['jac']


        return jac[1:, 1:]
    return dict(f_val = f_val, f_jac = f_jac)






def check_cat_label(x):
    x = x[~np.isnan(x)]
    xmax = max(x)
    xmin = min(x)
    nlevel = len(set(x))
    return (xmin == 1 and xmax == nlevel)

def create_cat_index_list(cat_index_level):
    cat_index_list = dict()
    start = 0
    for i in range(len(cat_index_level)):
        l = cat_index_level[i]
        if not np.isnan(l):
            cat_index_list[i] = list(range(start, (start+int(l.item()))))
            start = start+int(l.item())
        else:
            start = start+1
    return cat_index_list

def reduct_to_vector(ll):
    cat_index_list = []
    for i, v in ll.items():
        cat_index_list += v
    return cat_index_list

def adjust_index_list(index_list):
    '''
        convert a dict index_list to a index vector of mu
    '''
    # {'2':[0,1,2,3], '6':[0,1,2], '12':[0,1,2]} ->  {'2':[0,1,2,3], '6':[4,5,6], '12':[7,8,9]}
    index_list = copy.deepcopy(index_list)
    start = 0
    for i,v in index_list.items():
        # vals = index_list[i]
        if not isinstance(v, np.ndarray):
            v = np.array(v)
        index_list[i] = (v - v[0] + start).tolist()
        start = start + len(v)
    return index_list


def get_cat_bounds(X_cat, mu, cat_index_list, check=False, old = False, cat_range=None):
  '''
  :param X_cat: categorical values,
  :param mu: means
  :param cat_index_list: index of key:arange of values
  :param check:
  :param old:
  :return:
  '''
  # TODO check values of X_cat
  # if (old) return(get_cat_bounds_old(X_cat, mu, cat_index_list, check))
  d_cat = sum([len(ii) for ii in cat_index_list.values()])
  assert(d_cat == len(mu))
  p_cat = len(cat_index_list)
  assert(p_cat == len(X_cat[0]))

  n = len(X_cat)
  lower = np.zeros((n, d_cat))
  lower[:] = np.nan
  upper = lower.copy()

  # cat_range
  cat_range_values = list(cat_range.values())

  # cat_index_list -> {2:[0,1,2], 4:[3, 4,5]}
  incat_index_list = adjust_index_list(cat_index_list) # transform the dict to a vector, which contains the index of mu

  for j, k in enumerate(cat_index_list.keys()):
    if cat_range is not None:
      start_from = cat_range_values[j][0]
    else:
      start_from = 1

    index_o = ~np.isnan(X_cat[:, j])
    x_cat_obs = X_cat[index_o, j]
    n_obs = len(x_cat_obs)
    # initialize to (0, Inf): at argmax, we want (-inf, inf), at other loc, we want (mu_j - mu_argmax, inf)
    # index in 1,...,d_cat
    index_cat = incat_index_list[k]
    dj_cat = len(index_cat)
    l_o = np.zeros((n_obs, dj_cat))  # initiate the range [0, inf] to lower and upper
    u_o = l_o.copy() + np.inf

    # adjust for mean
    # length d_cat
    mu_j = mu[index_cat]            #
    # z_argmax - z_{-argmax} + mu_j[argmax] - mu_j[-argmax] >=0
    # thus z_argmax - z_{-argmax} >= mu_j[-argmax] - mu_j[argmax] (RHS computed below)
    mu_diff = np.tile(mu_j, (n_obs, 1))  - np.tile(mu_j[(x_cat_obs-start_from).astype(np.int64).tolist()], (dj_cat, 1)).T
    l_o = l_o + mu_diff
    # no constraints at argmax, thus -Inf lower
    #argmax_coor = np.array(list(enumerate(x_cat_obs)))
    l_o[np.arange(len(x_cat_obs)).tolist(), (x_cat_obs-start_from).astype(np.int64).tolist()] = -np.inf
    #l_o[np.ix_(np.arange(len(x_cat_obs)), x_cat_obs)] = -np.inf

    #lower[index_o, index_cat[0]:index_cat[-1]+1] = l_o #lower[index_o, index_cat] = l_o
    #upper[index_o,  index_cat[0]:index_cat[-1]+1] = u_o #upper[index_o, index_cat] = u_o
    lower[np.ix_(index_o, index_cat)] = l_o
    upper[np.ix_(index_o, index_cat)] = u_o
    # if (check):
    #     for i in range(n):
    #         ind1 = np.isnan(lower[i,])
    #         ind2 = get_cat_slicing_index(X_cat[i,], incat_index_list, keep = 'missing', d_cat=d_cat)$cat
    #         if (!all(ind1==ind2)) stop('something wrong!')
  return dict(lower = lower, upper = upper)

def range_transform(X, type):
  n = len(X)
  p = len(X[0])
  if (type == "continuous" and p > 0):      # the strictly monotone function of tranforming continuous variables to laten Zs
    r_val = X.copy()
    for j in range(p):
      X_tmp = X[:,j]
      fun = sm.distributions.ECDF(X_tmp[~np.isnan(X_tmp)])
      r_val[:, j][~np.isnan(X_tmp)] = fun(X_tmp[~np.isnan(X_tmp)])
    #r_val = qnorm(r_val * n/(n + 1))
    r_val = norm.ppf(r_val*n/(n+1))
    return dict(Z = r_val)

  if (type == "ordinal" and p > 0):
    d = np.zeros((p, ))         #d = numeric(p)
    r_lower = X.copy()
    r_upper = X.copy()
    for j in range(p):
      x = np.sort(np.unique(X[:, j]))
      x = x[~np.isnan(x)]
      assert(len(x) > 1)
      d[j] =  min(x[1:] -x[:-1]) / 2
      X_tmp = X[:, j]
      fun = sm.distributions.ECDF(X_tmp[~np.isnan(X_tmp)])

      X_tmp2, X_tmp3 = X_tmp.copy(), X_tmp.copy()
      X_tmp2[~np.isnan(X_tmp)] = fun(X_tmp[~np.isnan(X_tmp)] - d[j])
      X_tmp3[~np.isnan(X_tmp)] = fun(X_tmp[~np.isnan(X_tmp)] + d[j])
      r_lower[:, j] = X_tmp2
      r_upper[:, j] = X_tmp3

    r_lower = norm.ppf(r_lower)
    r_upper = norm.ppf(r_upper)
    return dict(Lower = r_lower, Upper = r_upper)

def range_transform_windows(X, type, windows=None):
  n, p = len(X), len(X[0])
  if (type == "continuous" and p > 0):      # the strictly monotone function of tranforming continuous variables to laten Zs
    r_val = X.copy()
    for j in range(p):
      X_tmp = X[:,j]
      l = len(windows[:, j])
      #fun = sm.distributions.ECDF(X_tmp[~np.isnan(X_tmp)])
      fun = sm.distributions.ECDF(windows[:, j])
      r_val[:, j][~np.isnan(X_tmp)] = fun(X_tmp[~np.isnan(X_tmp)])
    #r_val = qnorm(r_val * n/(n + 1))
    r_val = norm.ppf(r_val*l/(l+1))
    r_val[r_val==0] =  l/(l+1)/2
    return dict(Z = r_val)

  if (type == "ordinal" and p > 0):
    d = np.zeros((p, ))         #d = numeric(p)
    r_lower = np.empty(X.shape)
    r_lower[:] = np.nan
    r_upper = np.empty(X.shape)
    r_upper[:] = np.nan

    for j in range(p):
      # dealing with each dimension
      # x = np.sort(np.unique(X[:, j]))
      # x = x[~np.isnan(x)]
      # assert(len(x) > 1)
      # d[j] =  min(x[1:] -x[:-1]) / 2 # interval of int

      X_tmp = X[:, j]
      fun = sm.distributions.ECDF(windows[:,j]) # emprical cdf function
      unique_val = np.unique(windows[:, j])
      X_tmp2, X_tmp3 = X_tmp.copy(), X_tmp.copy()
      
      if unique_val.shape[0] > 1:
          #fun = sm.distributions.ECDF(X_tmp[~np.isnan(X_tmp)])
          threshold = np.min(np.abs(unique_val[1:] - unique_val[:-1])) / 2.0
          obs_indices = ~np.isnan(X_tmp)
          X_tmp2[obs_indices] = norm.ppf(fun(X_tmp[obs_indices] - threshold)) # lower
          X_tmp3[obs_indices] = norm.ppf(fun(X_tmp[obs_indices] + threshold)) # upper
          r_lower[:, j] = X_tmp2
          r_upper[:, j] = X_tmp3

          u_lower = stats.norm.cdf(r_lower[:, j][obs_indices])  # u_lower = pnorm(r_lower[obs_indices])
          u_upper = stats.norm.cdf(r_upper[:, j][obs_indices])  # u_upper = pnorm(r_upper[obs_indices])
          if (min(u_upper - u_lower) <= 0):
              # loc = which.min(u_upper - u_lower)
              loc = np.where((u_upper - u_lower) == min(u_upper - u_lower))[0]
              print(f"index=  {j}, loc = {loc}")
              print("Min of upper - lower", u_upper[loc] - u_lower[loc])
              print("where upper is", u_upper[loc], "and lower is", u_lower[loc])
              exit(-1)
          if (min(u_lower) < 0):
              loc = np.where(u_lower < 0 )[0]
              print(f"Invalid min of lower {min(u_lower)}, index = {j}, loc = {loc}")
              exit(-1)
          if (max(u_upper) > 1):
              loc = np.where(u_upper > 0)[0]
              print(f"Invalid max of upper {max(u_upper)},  index = {j}, loc = {loc}")
              exit(-1)

      else:
          X_tmp2[~np.isnan(X_tmp)] = -np.inf  # lower
          X_tmp3[~np.isnan(X_tmp)] = np.inf  # upper
          r_lower[:, j] = X_tmp2
          r_upper[:, j] = X_tmp3
    return dict(Lower = r_lower, Upper = r_upper)


def moments_truncnorm_vec(mu, std, a, b, tol = 1e-06, mean_only = False):
  alpha = (a - mu)/std
  beta = (b - mu)/std
  Z = norm.cdf(beta) - norm.cdf(alpha)
  p = len(Z)
  if (any(~np.isfinite(Z)) or min(Z) < 0):
    print("Invalid input")
    exit(-1)
  condition = np.logical_and(Z > tol, Z < 1)
  work_loc = np.where(condition)[0] # work_loc = which((Z > tol) & (Z < 1))
  trivial_loc = Z == 1
  fail_loc = Z <= tol
  pdf_beta = norm.pdf(beta[work_loc]) # pdf_beta = dnorm(beta[work_loc])
  pdf_alpha = norm.pdf(alpha[work_loc]) # pdf_alpha = dnorm(alpha[work_loc])
  if (any(~np.isfinite(pdf_alpha)) or any(~np.isfinite(pdf_beta))):
    print("Invalid input")
    exit(-1)
  mean_ = np.zeros(p)
  r1 = (pdf_beta - pdf_alpha)/Z[work_loc]
  mean_[work_loc] = mu[work_loc] - r1 * std[work_loc]
  mean_[fail_loc] = np.inf
  mean_[trivial_loc] = mu[trivial_loc]
  out = dict(mean = mean_)
  if (mean_only == False):
    loc_list = dict()
    r2_list = dict()
    beta_work = beta[work_loc]
    alpha_work = alpha[work_loc]
    Z_work = Z[work_loc]
    loc = beta_work >= np.inf
    if (any(loc)):
      r2_list["inf_beta"] = (-alpha_work[loc] * pdf_alpha[loc])/Z_work[loc]
      loc_list["inf_beta"] = loc
    loc = alpha_work <= -np.inf
    if (any(loc)):
      r2_list["inf_alpha"] = (beta_work[loc] * pdf_beta[loc])/Z_work[loc]
      loc_list["inf_alpha"] = loc
    loc = (beta_work < np.inf) & (alpha_work > -np.inf)
    if (any(loc)):
      r2_list["finite"] = (beta_work[loc] * pdf_beta[loc] - \
        alpha_work[loc] * pdf_alpha[loc])/Z_work[loc]
      loc_list["finite"] = loc
    std_ = np.zeros((p, ))
    for name in loc_list.keys():
      loc = loc_list[name]
      abs_loc = work_loc[loc]
      std_[abs_loc] = std[abs_loc] * np.sqrt(1 - r2_list[name] - (r1[loc]**2))
    std_[fail_loc] = np.inf
    std_[trivial_loc] = std[trivial_loc]
    out["std"] = std_
  return out

def get_cat_slicing_index(x_cat, cat_index_list, keep = 'observed', d_cat=None):
  if ( isinstance(keep, str)):
    if (keep == 'observed'):
      index_incat = ~np.isnan(x_cat)
    elif (keep == 'missing'):
      index_incat = np.isnan(x_cat)
    else:
      print('invalid char keep')
      exit(-1)
  elif (isinstance(keep[0], bool) and len(keep) == len(x_cat)):
    index_incat = keep
  else:
    print('invalid keep')
    exit(-1)
  if (d_cat == None):
    d_cat= sum(list(map(len, cat_index_list.values())))
  index_cat = np.zeros((d_cat,)).astype(np.bool_)
  if (any(index_incat)):
    # intindex_cat = purrr::reduce(cat_index_list[index_incat], c)
    intindex_cat = [jj for ii in cat_index_list[index_incat].values() for jj in ii]
    index_cat[intindex_cat] = True
  return dict(incat=index_incat, cat=index_cat)


def x_to_A(x, cat_index_list, d_cat=None, adjust=True, test=True, old = False):
  if (adjust):
    cat_index_list = adjust_index_list(cat_index_list)
  if (test):
    if (d_cat is not None):
      assert(d_cat == sum(list(map(len, cat_index_list.values()))))

  if (d_cat is None):
    d_cat = sum(list(map(len, cat_index_list.values())))
  if (any(np.isnan(x))):
    print('invalid x')
    exit(-1)

  if (old):
    index_notbase = get_cat_slicing_index(x, cat_index_list, keep = x!=1, d_cat = d_cat)
    if (any(index_notbase['incat'])):
      A = np.identity(d_cat) # A = diag(nrow = d_cat, ncol = d_cat)
      # for each xi != 1
      for i in np.where(index_notbase['incat'])[0]:
        index = cat_index_list[[i]]
        x_index = x[i]-1
        Ai = -np.identity(len(index))
        Ai[:,x_index] = 1
        A[index,index] = Ai
    else:
      A = None
  else:
    A = np.identity(d_cat)
    p = len(x)
    for i, (kk, vv) in enumerate(cat_index_list.items()):
      index = vv
      Ai = -np.identity(len(index))
      Ai[:,int(x[i]-1)] = 1
      #A[index[0]:index[-1]+1,index[0]:index[-1]+1] = Ai
      A[np.ix_(index, index)] = Ai
  return A

def Z_to_original_trunc(Z, X_cat, cat_index_list, old = False):
  n = len(Z)
  for i in range(n):
    z = Z[i]
    x_cat = X_cat[i]
    obs_indices = ~np.isnan(z)
    cat_obs = ~np.isnan(x_cat)
    if (any(cat_obs)):
      cat_indexs = dict()
      for ii, (kk, vv) in enumerate(cat_index_list.items()):
        if cat_obs[ii]:
            cat_indexs[kk] = vv
      A = x_to_A(x = x_cat[cat_obs], cat_index_list = cat_indexs, old = old)
      # if (any(A != None)):
        # z[obs_indices] = A %*% z[obs_indices] #matrix product
      if A is not None:
        z[obs_indices] = A @ z[obs_indices]
      Z[i] = z
  return Z


def initZ_noncat(Lower, Upper, X_cat, cat_in_d, cat_index_list, method = 'univariate_mean', old = False):
  Z_init = initZ_interval_truncated(Lower, Upper, method = method)
  Zord = Z_init[:, ~cat_in_d]
  Zcat = Z_init[:, cat_in_d]
  Zcat = Z_to_original_trunc(Zcat, X_cat, cat_index_list, old=old)
  return dict(Zord = Zord, Zcat = Zcat)

def initZ_interval_truncated(Lower, Upper, seed = None, method = "univariate_mean"):
  if (seed != None):
    np.random.seed(seed) #set.seed(seed)
  r_upper = Upper.copy()
  r_lower = Lower.copy()
  n = len(r_upper)  #n = dim(r_upper)[1]
  k = len(r_upper[0]) # k = dim(r_upper)[2]
  Z = np.zeros((n, k))  # Z = matrix(NA, n, k)
  Z[:] = np.nan
  obs_indices = ~np.isnan(r_lower)
  u_lower = stats.norm.cdf(r_lower[obs_indices]) # u_lower = pnorm(r_lower[obs_indices])
  u_upper = stats.norm.cdf(r_upper[obs_indices]) # u_upper = pnorm(r_upper[obs_indices])
  if (min(u_upper - u_lower) <= 0):
    # loc = which.min(u_upper - u_lower)
    loc = np.where((u_upper - u_lower) == min(u_upper-u_lower))[0]
    print("Min of upper - lower", u_upper[loc] - u_lower[loc])
    print("where upper is", u_upper[loc], "and lower is", u_lower[loc])
    exit(-1)
  if (min(u_lower) < 0):
    print("Invalid min of lower", min(u_lower))
    exit(-1)
  if (max(u_upper) > 1):
    print("Invalid max of upper", max(u_upper))
    exit(-1)

  if method == "sampling":
    # Z[obs_indices] = norm.ppf(purrr::map2_dbl(u_lower, u_upper,runif, n = 1))
    Z[obs_indices] = norm.ppf(np.random.normal(u_lower, u_upper, seed=1))
  elif method == "univariate_mean":
    l = np.sum(obs_indices)
    out = moments_truncnorm_vec(mu=np.zeros(l), std=1 + np.zeros(l), a = r_lower[obs_indices], b = r_upper[obs_indices], mean_only = True)
    Z[obs_indices] = out['mean']
  return Z


def initZ(Lower, Upper, X, cat_index, ord_in_noncat, cat_in_d, c_index, dord_index, dcat_index,
          cat_index_list, Z_cont=None, m=1, method = 'univariate_mean', old = False):
  X_cat = X[:, cat_index]
  X_noncat = X[:,~cat_index]

  if (any(c_index) and Z_cont == None):
    Z_cont = range_transform(X_noncat[:, ~ord_in_noncat], type='continuous')
    Z_cont = Z_cont['Z']
  elif (Z_cont == None):
    assert(all(~c_index))
  else:
    assert(len(Z_cont[0]) == sum(c_index))

  n = len(X)
  d = len(dcat_index)

  def call_initZ_noncat():
    Zinit = initZ_noncat(Lower, Upper, X_cat, cat_in_d, cat_index_list, old = old, method = method)
    return Zinit

  def setZ(Zinit):
    Z = np.zeros((n,d))  # Z = matrix(NA, n, d)
    Z[:] = np.nan
    Z[:, dord_index] = Zinit['Zord']
    Z[:, dcat_index] = Zinit['Zcat']
    Z[:, c_index] = Z_cont
    return Z

  if (m == 1):  #默认情况m=1, 此处只考虑m=1
    Zinit = call_initZ_noncat()
    out = setZ(Zinit)
  # else:
  #   Zinits = purrr::map(1:m, ~ call_initZ_noncat())
  #   out = purrr::map(Zinits, setZ)
  return out

def initZ_windows(Lower, Upper, X, cat_index, ord_in_noncat, cat_in_d, c_index, dord_index, dcat_index,
          cat_index_list, window,  Z_cont=None, m=1, method = 'univariate_mean', old = False):
  X_cat = X[:, cat_index]
  X_noncat = X[:,~cat_index]

  if (any(c_index) and Z_cont == None):
    Z_cont = range_transform_windows(X_noncat[:, ~ord_in_noncat], type='continuous', windows=window)
    Z_cont = Z_cont['Z']
  elif (Z_cont == None):
    assert(all(~c_index))
  else:
    assert(len(Z_cont[0]) == sum(c_index))

  n = len(X)
  d = len(dcat_index)

  def call_initZ_noncat():
    Zinit = initZ_noncat(Lower, Upper, X_cat, cat_in_d, cat_index_list, old = old, method = method)
    return Zinit

  def setZ(Zinit):
    Z = np.zeros((n,d))  # Z = matrix(NA, n, d)
    Z[:] = np.nan
    Z[:, dord_index] = Zinit['Zord']
    Z[:, dcat_index] = Zinit['Zcat']
    Z[:, c_index] = Z_cont
    return Z

  if (m == 1):  #默认情况m=1, 此处只考虑m=1
    Zinit = call_initZ_noncat()
    out = setZ(Zinit)
  # else:
  #   Zinits = purrr::map(1:m, ~ call_initZ_noncat())
  #   out = purrr::map(Zinits, setZ)
  return out

def regularize_corr(sigma, corr_min_eigen = 0.001, verbose = False, prefix = ""):
  eigenvalues, eigenvectors = np.linalg.eig(sigma)
  o_eigen = dict(values=eigenvalues, vectors=eigenvectors)

  if (min(o_eigen['values']) < corr_min_eigen):
    values = o_eigen['values']
    values[values < corr_min_eigen] = corr_min_eigen
    sigma = cov2corr(o_eigen['vectors'] @ np.diag(values) @  o_eigen['vectors'].T) # sigma = cov2cor(o_eigen['vectors'] @ np.diag(values) @  o_eigen['vectors'].T)
    if (verbose):
      print(f"{prefix} small eigenvalue in the copula correlation")
  return sigma

def project_to_nominal_corr(sigma, cat_index_list, eps = 1e-05):
  p = len(sigma[0])
  A = np.identity(p)
  for i, (kk, vv) in enumerate(cat_index_list.items()):
    index = list(vv)
    eigenvalues, eigenvectors = np.linalg.eig(sigma[index[0]:index[-1]+1, index[0]:index[-1]+1])
    o_eigen = dict(values=eigenvalues, vectors=eigenvectors)
    eigen_values = o_eigen['values']
    if (min(eigen_values) < 1e-05):
      print("Projection skipped: small eigenvalue in a categorical block")
      A[np.ix_(index, index)] = np.identity(len(index))
    else:
      m = o_eigen['vectors'] @ np.sqrt(np.diag(1/eigen_values)) @ o_eigen['vectors'].T
      A[index[0]:index[-1]+1, index[0]:index[-1]+1] = m

  sigma = A @ sigma @ A.T
  for i, (kk, vv) in enumerate(cat_index_list.items()):
    index = list(vv)
    l = len(index)
    sigma[index[0]:index[-1]+1, index[0]:index[-1]+1] = np.identity(l)
  return sigma

def is_symmetric(matrix):
    '''
    :param matrix:
    :return: Is it symmetric?
    '''
    if matrix.shape[0] != matrix.shape[1]:
        return False
    return np.allclose(matrix, matrix.T)

def est_z_row_ord(z, lower, upper, sigma_oo, ord_indices = None, obs_indices = None,
  ord_obs_indices = None, obs_in_ord = None, ord_in_obs = None,
  n_sample = 5000):
  if (obs_indices == None):
    obs_indices = ~np.isnan(z)
  if ((ord_obs_indices == None) or (obs_in_ord == None) or (ord_in_obs == None)):
    if (ord_indices == None):
      print("provide ord_indices")
      exit(-1)
    ord_obs_indices = ord_indices & obs_indices
    ord_in_obs = ord_obs_indices[obs_indices]
    obs_in_ord = ord_obs_indices[ord_indices]
  p = len(obs_indices)
  if (sum(obs_indices) > 1 and any(ord_obs_indices)):
    sigma_ord_ord = sigma_oo[np.ix_(ord_in_obs, ord_in_obs)]
    cont_in_obs = ~ord_in_obs
    if (z!= None and any(cont_in_obs)):
      z_obs = z[obs_indices]
      sigma_cont_ord = sigma_oo[np.ix_(cont_in_obs, ord_in_obs)]
      sigma_cont_cont = sigma_oo[np.ix_(cont_in_obs, cont_in_obs)]
      tot_m = np.concatenate((z_obs[cont_in_obs], sigma_cont_ord), axis=1)
      sol_m = np.linalg.solve(sigma_cont_cont, tot_m)
      cond_mean = sigma_cont_ord.T @ sol_m[:, 1]
      cond_cov = sigma_ord_ord - sigma_cont_ord.T @ sol_m[:, 1:]
    else:
      cond_mean = np.zeros((sum(ord_obs_indices), ))
      cond_cov = sigma_ord_ord
    if (not is_symmetric(cond_cov)):
      diff = max(abs(cond_cov - cond_cov.T))
      if (diff > 1e-04):
        print(f"max diff:{diff}")
      cond_cov = (cond_cov + cond_cov.T)/2
    cond_mean = cond_mean.tolist()
    lower_use = lower[obs_in_ord]
    upper_use = upper[obs_in_ord]
    lmu = len(cond_mean)
    ll = len(lower_use)
    lu = len(upper_use)
    if (lmu != len(cond_cov[0]) or lmu != ll or lmu != lu):
      print("inconsistent shapes")
      exit(-1)
    out_ = get_trunc_2dmoments(list(cond_mean), cond_cov, lower_use, upper_use, n_sample = n_sample)
    if (z == None):
      z_new = np.zeros((p,))
      z_new[:] = np.nan
    else:
      z_new = z
    z_new[ord_obs_indices] = out_['mean']
    cov_all = np.zeros((p, p))
    cov_all[np.ix_(ord_obs_indices, ord_obs_indices)] = out_['cov']
    out = dict(mean = z_new, cov = cov_all, var = None)
  else:
    out = dict(mean = z, var = np.zeros((p,)))
  return out

def sample_z_row_ord(z, lower, upper, sigma_oo, ord_indices = None, obs_indices = None,
                     ord_obs_indices = None, obs_in_ord = None, ord_in_obs = None,  n_sample = 5000):
  if (obs_indices == None):
    obs_indices = ~np.isnan(z)
  if (ord_obs_indices == None or (obs_in_ord == None) or (ord_in_obs == None)):
    assert((ord_indices is not None))
    ord_obs_indices = ord_indices & obs_indices
    ord_in_obs = ord_obs_indices[obs_indices]
    obs_in_ord = ord_obs_indices[ord_indices]
  p = len(obs_indices)
  if (sum(obs_indices) > 1 and any(ord_obs_indices)):
    sigma_ord_ord = sigma_oo[np.ix_(ord_in_obs, ord_in_obs)]
    cont_in_obs = ~ord_in_obs
    if ((z is not None) and any(cont_in_obs)):
      z_obs = z[obs_indices]
      sigma_cont_ord = sigma_oo[np.ix_(cont_in_obs, ord_in_obs)]
      sigma_cont_cont = sigma_oo[np.ix_(cont_in_obs, cont_in_obs)]
      tot_m = np.concatenate((z_obs[cont_in_obs], sigma_cont_ord), axis=1)
      sol_m = np.linalg.solve(sigma_cont_cont, tot_m)
      cond_mean = sigma_cont_ord.T @ sol_m[:, 0]
      cond_cov = sigma_ord_ord - sigma_cont_ord.T @ sol_m[:, 1:]
    else:
      cond_mean = np.zeros((sum(ord_obs_indices),)) # cond_mean = numeric(sum(ord_obs_indices))
      cond_cov = sigma_ord_ord
    if (not is_symmetric(cond_cov)):
      diff = max(abs(cond_cov - cond_cov.T))
      if (diff > 1e-04):
        print(f"max diff:{diff}")
      cond_cov = (cond_cov + cond_cov.T)/2
    cond_mean = c(cond_mean)
    lower_use = lower[obs_in_ord]
    upper_use = upper[obs_in_ord]
    lmu = len(cond_mean)
    ll = len(lower_use)
    lu = len(upper_use)
    if (lmu != len(cond_cov[0]) or lmu != ll or lmu != lu):
      print("inconsistent shapes")
      exit(-1)
    #"out_ = get_trunc_2dmoments(c(cond_mean), cond_cov,\n                               lower_use, upper_use,\n                               n_sample=n_sample)"
    # zsample = TruncatedNormal::rtmvnorm(n = n_sample, mu = c(cond_mean),
    #                                     sigma = cond_cov, lb = lower_use, ub = upper_use)

    aa = (lower_use - cond_mean) / cond_cov
    bb = (upper_use - cond_mean) / cond_cov
    zsample = truncnorm.rvs(aa, bb, loc=cond_mean, scale=cond_cov, size=1000)
    z_new = np.tile(z, (n_sample, 1)) # z_new = matrix(z, n_sample, p, byrow = TRUE)
    z_new[:, ord_obs_indices] = zsample
  else:
    z_new = z.reshape((n_sample,p)) #z_new = matrix(z, n_sample, p, byrow = TRUE)
  return z_new



def latent_operation_row(task, z, lower, upper, d_index, dcat_index, corr, cat_input = None, trunc_method = "Iterative", n_sample = 5000,
          n_update = 1, n_MI = 1):
  sigma = corr
  p = len(sigma[0])
  out_return = dict()
  mis_indices = np.isnan(z)
  obs_indices = ~mis_indices
  ord_indices = d_index
  ord_obs_indices = ord_indices & obs_indices
  ord_in_obs = ord_obs_indices[obs_indices]
  obs_in_ord = ord_obs_indices[ord_indices]
  sigma_oo = sigma[np.ix_(obs_indices, obs_indices)]
  sigma_om = sigma[np.ix_(obs_indices, mis_indices)]
  sigma_mm = sigma[np.ix_(mis_indices, mis_indices)]
  n_obs = np.sum(obs_indices)
  if any(mis_indices):
    ans = np.linalg.solve(sigma_oo, np.concatenate((np.identity(n_obs), sigma_om), axis=1))
    sigma_oo_inv = ans[:, :n_obs]
    J_mo = ans[:, n_obs:].T
  else:
    sigma_oo_inv = np.linalg.inv(sigma_oo)
    J_mo = None

  if (task == "em"):
    z_obs = z[obs_indices]
    negloglik = np.linalg.det(sigma_oo) + np.sum(z_obs * sigma_oo_inv @ z_obs)
    negloglik = negloglik + p * np.log(2 * np.pi)
    loglik = -negloglik/2
    out_return["loglik"] = loglik.copy()

  if (any(dcat_index == None)):
    dcat_index = np.zeros((p,)).astype(np.bool_)

  cat_obs_indices = dcat_index & obs_indices
  cat_in_obs = cat_obs_indices[obs_indices]

  if (any(cat_obs_indices)):
    assert(cat_input != None)
    x_cat = cat_input["x_cat"]
    cat_index_list = cat_input["cat_index_list"]
    cat_obs = ~np.isnan(x_cat)
    cat_indexs_tmp = dict()
    for ii, (kk, vv) in enumerate(cat_index_list.items()):
        if cat_obs[ii]:
            cat_indexs_tmp[kk] = vv
    A = x_to_A(x = x_cat[cat_obs], cat_index_list =cat_indexs_tmp,
               d_cat = sum(cat_obs_indices), old = cat_input["old"])
  else:
    A = None
  if (trunc_method == "TruncatedNormal" or trunc_method == "Sampling_TN"):
    trunc_method = "Sampling"
  #"if (trunc_method == 'Iterative'){\n    out_ref <- est_z_row_ord(z, lower, upper,\n                             obs_indices = obs_indices,\n                             ord_obs_indices = ord_obs_indices,\n                             ord_in_obs = ord_in_obs,\n                             obs_in_ord = obs_in_ord,\n                             sigma_oo = A_sigma_tA_at_cat(sigma_oo, A, cat_index=ord_in_obs),\n                             method = 'TruncatedNormal',\n                             n_sample = n_sample)\n    z = out_ref$mean\n    if (!is.null(A)) z[ord_obs_indices] = A %*% z[ord_obs_indices]\n    #out$var = diag(out_ref$cov)\n  }"
  if task in ["em", "fillup"]:
    if trunc_method == 'Iterative':
      if (A is not None):
        #"sigma_oo_inv = A_sigma_tA_at_cat(sigma_oo_inv, A,\n                                               cat_index=ord_in_obs, A_at_left = FALSE)"
        sigma_oo_inv = np.linalg.inv(A_sigma_tA_at_cat(sigma_oo, A, cat_index = cat_in_obs))
        z[cat_obs_indices] = A @ z[cat_obs_indices]
      def f_sigma_oo_inv_z(zz):
        return sigma_oo_inv @ zz

      assert(all(np.diag(sigma_oo_inv) >= 0))
      out = update_z_row_ord(z, lower, upper, obs_indices=obs_indices,
                             ord_obs_indices=ord_obs_indices, ord_in_obs=ord_in_obs,
                             obs_in_ord=obs_in_ord, f_sigma_oo_inv_z=f_sigma_oo_inv_z,
                             sigma_oo_inv_diag=np.diag(sigma_oo_inv), n_update=n_update)
    elif trunc_method == 'Sampling':
      # out = est_z_row_ord(z, lower, upper, obs_indices=obs_indices,
      #                     ord_obs_indices=ord_obs_indices, ord_in_obs=ord_in_obs,
      #                     obs_in_ord=obs_in_ord, sigma_oo=A_sigma_tA_at_cat(sigma_oo, A, cat_index=cat_in_obs),
      #                     n_sample=n_sample)
      pass
    else:
      print("invalid trunc_method")
      exit(-1)
    z = out['mean']
    if any(np.isnan(z[obs_indices])):
      print("invalid Zobs")
      exit(-1)
    if 'cov' not in out.keys():
      if (out['var'] is None):
        print("wrong return from trunc ordinal update")
        exit(-1)
      cov_ordinal = np.diag(out['var'])
    else:
      cov_ordinal = out['cov']

    if (A is not None):
      z[cat_obs_indices] = A @ z[cat_obs_indices]
      cov_ordinal = A_sigma_tA_at_cat(cov_ordinal, A,
                                      cat_index = cat_obs_indices)
    if task == 'em':
      out_return["Z"] = z.copy()
      out_return["var_ordinal"] = np.diag(cov_ordinal)
    elif task == "fillup":
      out_return["var_ordinal"] = np.diag(cov_ordinal)
  z_obs = z[obs_indices]
  zimp = z
  if (any(mis_indices)):
    zimp[mis_indices] = J_mo @ z_obs
  if (any(np.isnan(zimp))):
    print("invalid imputation")
    exit(-1)
  if (task in["em", "fillup"]):
    out_return["Zimp"] = zimp.copy()
  if (task == "sample"):
    if (trunc_method == "Sampling"):
      zsample = sample_z_row_ord(z, lower, upper, obs_indices = obs_indices,
                                  ord_obs_indices = ord_obs_indices, ord_in_obs = ord_in_obs,
                                  obs_in_ord = obs_in_ord, sigma_oo = A_sigma_tA_at_cat(sigma_oo,
                                                                                        A, cat_index = cat_in_obs), n_sample = n_MI)
      if (A != None):
        zsample[:, cat_obs_indices] = zsample[:, cat_obs_indices] @ A.T
    else:
      assert(n_MI == 1)
      zsample = z.reshape((1, -1))
    if (any(mis_indices)):
      mis_cond_cov = sigma_mm - J_mo @ sigma_om
      pmis = sum(mis_indices)
      zbar = np.random.multivariate_normal(np.zeros((pmis,)), mis_cond_cov, size=n_MI)
      mis_cond_mean = zsample[:, obs_indices] @ J_mo.T
      assert(zbar.shape == mis_cond_mean.shape)
      zsample[:, mis_indices] = zbar + mis_cond_mean
    out_return["Zimp_sample"] = zsample.copy()

  if (task == "em"):
    C = cov_ordinal
    if (any(mis_indices)):
      C[np.ix_(mis_indices, mis_indices)] = C[np.ix_(mis_indices, mis_indices)] + sigma_mm - J_mo @ sigma_om
      if (np.sum(np.diag(cov_ordinal)) > 0):
        cov_missing_obs_ord = J_mo[:, ord_in_obs] @ cov_ordinal[np.ix_(ord_obs_indices, ord_obs_indices)]
        C[np.ix_(mis_indices, ord_obs_indices)] = C[np.ix_(mis_indices, ord_obs_indices)] + cov_missing_obs_ord
        C[np.ix_(ord_obs_indices, mis_indices)] = C[np.ix_(ord_obs_indices, mis_indices)] + cov_missing_obs_ord.T
        C[np.ix_(mis_indices, mis_indices)] = C[np.ix_(mis_indices, mis_indices)] + cov_missing_obs_ord @ J_mo[:, ord_in_obs].T
    out_return["C"] = C.copy()
  return out_return


def latent_operation(task, Z, Lower, Upper, d_index, dcat_index, cat_input,
          corr, window_size, trunc_method = "Iterative", n_update = 1, n_sample = 5000,
          n_MI = 1, ):
  n = len(Z)
  p = len(Z[0])
  Z_lower = Lower
  Z_upper = Upper
  assert(isinstance(d_index[0], np.bool_))
  if task == "em":
    out = dict(Z=Z.copy(), Zimp=Z.copy(), loglik=0, C=np.zeros((p, p)), var_ordinal=np.zeros((n, p)))
  elif task == "fillup":
    out = dict(Zimp=Z.copy(), var_ordinal=np.zeros((n, p)))
  elif task == "sample":
    out = dict(Zimp_sample = np.zeros((n, p, n_MI)))

  if (cat_input != None):
    cat_input_row = dict(x_cat = None, cat_index_list = cat_input['cat_index_list'],
                         cat_index_all = cat_input['cat_index_all'], old = cat_input['old'])
  else:
    cat_input_row = None
  for i in range(n):
    if cat_input_row is not None:
      cat_input_row["x_cat"] = cat_input['X_cat'][i]


    row_out = latent_operation_row(task, Z[i], Z_lower[i], Z_upper[i], d_index = d_index, dcat_index = dcat_index,
    cat_input = cat_input_row, corr = corr, trunc_method = trunc_method,
    n_update = n_update, n_sample = n_sample, n_MI = n_MI)

    if task == "em":
      for name in ["Z", "Zimp", "var_ordinal"]:
        out[name][i,] = row_out[name]
      for name in ["loglik", "C"]:
        out[name] = out[name] + row_out[name] / window_size
    elif task == "fillup":
      for name in ["Zimp", "var_ordinal"]:
        out[name][i,] = row_out[name]
    elif task == "sample":
        out["Zimp_sample"][i] = row_out["Zimp_sample"]
  if (task == "em"):
    out["corr"] = np.cov(out["Zimp"],  rowvar=False) + out["C"]  # out["corr"] = np.cov(out["Zimp"]) + out["C"]
    if np.any(np.isnan(out["corr"])):
      print("invalid correlation")
      exit(-1)
  return out

def em_mixedgc(Z, Lower, Upper, d_index, window_size, dcat_index = None, cat_input = None,
          start = None, trunc_method = "Iterative", n_sample = 5000,
          n_update = 1, maxit = 50, eps = 0.01, verbose = False, runiter = 0,
          corr_min_eigen = 0.01, scale_to_corr = True,):
  Z_lower = Lower
  Z_upper = Upper

  p = len(d_index)
  c_index = ~d_index

  # R means sigma
  if (start == None):
    Z_meanimp = Z.copy()
    Z_meanimp[np.isnan(Z_meanimp)] = 0   #
    R = np.cov(Z_meanimp, rowvar=False)  # without initialization, caculating cov of the latent vector z
    del Z_meanimp #rm(Z_meanimp)
  else:
    R = start['R']

  eigenvalues, eigenvectors = np.linalg.eig(R) # o_eigen = eigen(R) # caculating eigenvalues and eigenvectors
  o_eigen = dict(values=eigenvalues, vectors=eigenvectors)

  if (min(o_eigen['values']) < 0):
    print("Bad initialization: Tiny negative eigenvalue potentially due to colinearity")
    values = o_eigen['values']
    values[values < corr_min_eigen] = corr_min_eigen
    R = cov2corr(o_eigen['vectors'] @ np.diag(values) @ o_eigen['vectors'].T) # R = cov2cor(o_eigen['vectors'] @ np.diag(values) @ o_eigen['vectors'].T)
  R = regularize_corr(R, corr_min_eigen = corr_min_eigen,
                      verbose = verbose, prefix = "Bad initialization potentially due to colinearity: ")
  assert(min(np.linalg.eig(R)[0]) > 0)
  if (cat_input != None):
    R = project_to_nominal_corr(R, cat_input['cat_index_list'])

  Zimp = Z
  l = 0
  loglik = None

  while True:
    l = l + 1
    est_iter = latent_operation("em", Z, Z_lower, Z_upper,
                                d_index = d_index, dcat_index = dcat_index, cat_input = cat_input,
                                corr = R,  window_size=window_size, trunc_method = trunc_method, n_sample = n_sample,
                                n_update = n_update)
    Z = est_iter['Z']
    Zimp = est_iter['Zimp']
    R1 = est_iter['corr']

    if (scale_to_corr):
      R1 = cov2corr(R1) # R1 = cov2cor(R1)
      R1 = regularize_corr(R1, corr_min_eigen = corr_min_eigen,
                           verbose = verbose)
      if (cat_input is not None):
        R1 = project_to_nominal_corr(R1, cat_input['cat_index_list'])

    err = np.linalg.norm(R1 - R, ord='fro')/np.linalg.norm(R,ord='fro') # err = norm(R1 - R, type = "F")/norm(R, type = "F"), Frobenius范数
    if loglik is None:
      loglik = [est_iter['loglik']]
    else:
      #loglik = loglik + est_iter['loglik']
      loglik.append(est_iter['loglik'])

    R = R1
    if (verbose):
      print("Iteration ", l, ": ", "copula parameter change ",
                   round(err, 4), ", likelihood ", np.round(est_iter['loglik'], 4))
    if (runiter == 0):
      if (err < eps):
        break
      if (l > maxit):
        print("Max iter reached in EM")
        break
    else:
      if (l >= runiter):
        break
  return dict(corr=R, loglik=loglik, Z=Z, Zimp=est_iter['Zimp'])

def A_sigma_tA_at_cat(sigma, A, cat_index = None, A_at_left = True):
  if (A is None):
    return sigma
  p = len(sigma[0])
  if (cat_index is None):
    A_all = A
  else:
    if (len(cat_index) != p):
      print("invalid cat_index and sigma")
      exit(-1)
    A_all = np.identity(p)        #diag(p)
    A_all[np.ix_(cat_index, cat_index)] = A
  if (A_at_left):
    return A_all @ sigma @ A_all.T
  else:
    return A_all.T @ sigma @ A_all


# similar to _em_step_body_row(Z_row, r_lower_row, r_upper_row, sigma, num_ord_updates=2)
def update_z_row_ord(z, lower, upper, obs_indices, ord_obs_indices, ord_in_obs,
  obs_in_ord, f_sigma_oo_inv_z, sigma_oo_inv_diag, n_update = 1):
  p = len(z)
  num_ord = len(lower)
  var_ordinal = np.zeros((p,))

  # OBSERVED ORDINAL ELEMENTS
  # when there is an observed ordinal to be imputed and another observed dimension, impute this ordinal
  if (sum(obs_indices) > 1 and any(ord_obs_indices)):
    ord_obs_iter = np.where(ord_obs_indices)[0]
    ord_in_obs_iter = np.where(ord_in_obs)[0]
    obs_in_ord_iter = np.where(obs_in_ord)[0]
    for i in range(n_update):
      sigma_oo_inv_z = f_sigma_oo_inv_z(z[obs_indices])

      #
      new_std = np.sqrt(1/sigma_oo_inv_diag[ord_in_obs_iter])
      new_mean = z[ord_obs_iter] - (new_std**2) * sigma_oo_inv_z[ord_in_obs_iter]
      a = lower[obs_in_ord_iter]
      b = upper[obs_in_ord_iter]
      if (isinstance(new_std[0], np.complex_)):
          print("invalid new_std")
          out_trunc = moments_truncnorm_vec(mu=new_mean, std=new_std, a=a, b=b)
          exit(-1)
      out_trunc = moments_truncnorm_vec(mu = new_mean, std = new_std, a = a, b = b)
      mean_ = out_trunc['mean']
      std_ = out_trunc['std']


      old_mean = z[ord_obs_iter]
      loc = ~np.isfinite(mean_)
      mean_[loc] = old_mean[loc]
      z[ord_obs_iter] = mean_
      std_[~np.isfinite(std_)] = 0
      var_ordinal[ord_obs_iter] = std_**2
  return dict(mean = z, var = var_ordinal)


# def get_trunc_2dmoments(mean, cov, lower, upper, n_sample = 5000):
#   p = len(mean)
#   if (p == 1):
#     out = moments_truncnorm(c(mean), c(cov), c(lower), c(upper))
#     return dict(mean = out['mean'], cov=np.array([out['std']**2])
#   method = "Sampling"
#   if method == "Sampling":
#     aa = (lower - mean) / cov
#     bb = (upper - mean) / cov
#     z = truncnorm.rvs(aa, bb, loc=mean, scale=cov, size=n_sample) # z = TruncatedNormal::rtmvnorm(n=n_sample, mu=mean, sigma=cov, lb=lower, ub=upper)
#     if (len(z.shape) != 2 or len(z) != p):
#       print("unexpected sample dimension")
#       exit(-1)
#     r = dict(mean=colMeans(z), cov=np.corrcoef(z))
#   elif method == "Diagonal":
#     r = moments_truncnorm_vec(mu=mean, std=sqrt(diag(cov)),
#                               a=lower, b=upper)
#     r = dict(mean=r['mean'], cov=np.diag(r['std'] ** 2))
#   else:
#     print(f"Invalid method vlaue: {method}")
#     exit(-1)
#   return r

def nominal_z_to_x_col(z, old = False):
  argmax = np.argmax(z) + 1 # 对齐
  if (old):
    if (z[argmax]<0):
      argmax = 1
    else:
      argmax = argmax + 1
  return argmax

def nominal_z_to_x_col_from_zero(z):
  argmax = np.argmax(z)# 对齐
  return argmax


def Ximp_transform_cat(Z_cat, X_cat, cat_index_list, old = False, cat_range=None):
  if (len(Z_cat[0]) != sum(map(len, cat_index_list.values()))):
    print('something wrong')
    exit(-1)
  cat_index_list = adjust_index_list(cat_index_list)
  cat_range_values = list(cat_range.values())
  Ximp_cat = X_cat.copy()
  for j, (k,v) in enumerate(cat_index_list.items()):
    start_from = cat_range_values[j][0]
    index_m = np.isnan(X_cat[:, j])
    if any(index_m) == True:
        index_cat = list(v)  # index_cat = cat_index_list[j]
        zmis = Z_cat[np.ix_(index_m,index_cat)]
        if start_from == 1:
          Ximp_cat[index_m, j] = np.apply_along_axis(nominal_z_to_x_col, axis=1, arr=zmis) # apply(zmis, 1, nominal_z_to_x_col)
        else:
          Ximp_cat[index_m, j] = np.apply_along_axis(nominal_z_to_x_col_from_zero, axis=1,
                                                   arr=zmis)  # apply(zmis, 1, nominal_z_to_x_col)

  return Ximp_cat

def Ximp_transform(Z, X, d_index):
  n = len(Z)
  p = len(Z[0])
  Ximp = X.copy()
  c_index = ~d_index
  for j in np.where(d_index)[0]:
    miss_ind = np.isnan(X[:, j])
    n_obs = n - sum(miss_ind)
    xmis_loc = np.floor(norm.cdf(Z[miss_ind, j]) * n_obs) # xmis_loc = pmax(math.ceil(norm.cdf(Z[np.ix_(miss_ind, j)]) * n_obs), 1)
    xmis_loc[xmis_loc < 1] = 1
    Ximp[miss_ind, j] = np.sort(X[~miss_ind, j])[xmis_loc.astype(np.int_)]

  for j in np.where(c_index)[0]:
    miss_ind = np.isnan(X[:, j])
    Ximp[miss_ind, j] = np.percentile(X[~miss_ind, j], norm.cdf(Z[miss_ind, j])*100)
  return Ximp

def Ximp_transform_windows(Z, X, d_index, windows):
  n = len(Z)
  p = len(Z[0])
  Ximp = X.copy()
  c_index = ~d_index

  # int type
  for j in np.where(d_index)[0]:
    miss_ind = np.isnan(X[:, j])
    n_obs = n - sum(miss_ind)
    xmis_loc = np.floor(norm.cdf(Z[miss_ind, j]) * n_obs) # xmis_loc = pmax(math.ceil(norm.cdf(Z[np.ix_(miss_ind, j)]) * n_obs), 1)
    xmis_loc[xmis_loc < 1] = 1
    #Ximp[miss_ind, j] = np.sort(X[~miss_ind, j])[xmis_loc.astype(np.int_)]
    Ximp[miss_ind, j] = np.sort(windows[:,j])[xmis_loc.astype(np.int_)]

  # continous type
  for j in np.where(c_index)[0]:
    miss_ind = np.isnan(X[:, j])
    #Ximp[miss_ind, j] = np.percentile(X[~miss_ind, j], norm.cdf(Z[miss_ind, j])*100)
    Ximp[miss_ind, j] = np.percentile(windows[:,j], norm.cdf(Z[miss_ind, j]) * 100)
  return Ximp

def latent_to_observed(Zimp, X, mu, cat_labels, ord_in_noncat,
                               cat_index, cat_index_all, cat_index_list, old=False):
  d_cat = len(cat_index_all)
  n = len(Zimp)
  Z_cat = Zimp[:,cat_index_all] + np.tile(mu, (n, 1))
  Ximp = X.copy()
  X_cat = X[:, cat_index]
  X_noncat = X[:, ~cat_index]
  Ximp[:, cat_index] = Ximp_transform_cat(Z_cat = Z_cat, X_cat = X_cat,
                                        cat_index_list = cat_index_list, old = old) #***!

  tmp_index = list(range(len(Zimp[0])))
  list(map(tmp_index.remove, cat_index_all))
  Ximp[:, ~cat_index] = Ximp_transform(Z=Zimp[:, tmp_index], X=X_noncat, d_index=ord_in_noncat)
  # Ximp[:, ~cat_index] = Ximp_transform(Z = Zimp[:, ~cat_index_all], X = X_noncat, d_index = ord_in_noncat)

  cat_index_int = np.where(cat_index)[0]
  for j in cat_index_int:
    Ximp[:, j] = relabel(Ximp[:,j], cat_labels[j])
  return Ximp

def latent_to_observed_windows(Zimp, X, mu, cat_labels, ord_in_noncat,
                               cat_index, cat_index_all, cat_index_list, windows, old=False, cat_range=None):
  d_cat = len(cat_index_all)
  n = len(Zimp)
  Z_cat = Zimp[:,cat_index_all] + np.tile(mu, (n, 1))
  Ximp = X.copy()
  X_cat = X[:, cat_index]
  X_noncat = X[:, ~cat_index]
  Ximp[:, cat_index] = Ximp_transform_cat(Z_cat = Z_cat, X_cat = X_cat,
                                        cat_index_list = cat_index_list, old = old, cat_range=cat_range) #***!

  tmp_index = list(range(len(Zimp[0])))
  list(map(tmp_index.remove, cat_index_all))
  Ximp[:, ~cat_index] = Ximp_transform_windows(Z=Zimp[:, tmp_index], X=X_noncat, d_index=ord_in_noncat, windows=windows)
  # Ximp[:, ~cat_index] = Ximp_transform(Z = Zimp[:, ~cat_index_all], X = X_noncat, d_index = ord_in_noncat)

  # cat_index_int = np.where(cat_index)[0]
  # for j in cat_index_int:
  #   Ximp[:, j] = relabel(Ximp[:,j], cat_labels[j])
  return Ximp

def impute_nominal_cont_gc(X, cat_index,
                            mu=None, corr=None, n_MI=0, fast_MI=False,
                            maxit=50, eps=0.01, nlevels=20, runiter=0, verbose=False,
                            seed=None, init_trunc_sampling=False,
                            trunc_method='Iterative', n_sample=5000, n_update=1, Z=None, old=False,):


    # if seed != None:
    #     set_seed(seed)
    n = len(X)
    p = len(X[0])
    dim = (n, p)

    # logical values to int
    if isinstance(cat_index[0], bool):   #
        if len(cat_index) != p:  #
            print('invalid cat_index')
            exit(-1)
        if isinstance(cat_index, np.ndarray) == False:
            cat_index = np.array(cat_index)
        cat_index_int = np.where(cat_index==True)[0].tolist()
    elif min(cat_index) < 0 or max(cat_index) > p-1:
        print('invalid cat_index')
        exit(-1)
    else:
        cat_index_int = cat_index

    # statistical freq
    # cat_freq = get_cat_index_freq(X[:, cat_index_int])
    # min_nl = min(cat_freq['nlevel'])
    # if (min_nl <= 1):
    #     print('some categorical var has no more than 1 level')
    #     exit(-1)
    # elif (min_nl == 2):
    #     idx = np.where(np.array(cat_freq['nlevel']) > 2)[0].tolist()
    #     cat_index_int = np.array(cat_index_int)
    #     cat_index_int = cat_index_int[idx]
    #     pcat = len(cat_freq['nlevel'])
    #     print(f'{pcat-len(idx)} of pcat categoricals have only two categories: treated as binary')


    # relabel
    cat_labels = dict()
    for j in cat_index_int:
        relabel = cat_to_integers(X[:, j])
        cat_labels[j] = relabel['xlevels']
        for i in range(len(relabel['x'])):
            X[:, j][i] = relabel['x'][i]

    # int to logical value
    cat_index = index_int_to_logi(cat_index_int, p)

    X_cat = X[:, cat_index]
    X_noncat = X[:, ~cat_index]

    # Do not allow empty row
    df = pd.DataFrame(X)
    if not all(df.apply(lambda x:np.sum(~np.isnan(x)), axis=1)):
        print("remove empty row")
        exit(-1)

    # Do not allow column with only one level
    unique_res = df.apply(lambda x:len(np.unique(x[~np.isnan(x)]))<=1, axis=0)
    if any(unique_res):
        print(f'remove column with only 0 or 1 unique value, {np.where(unique_res)[0]}')
        exit(-1)

    cat_level = np.zeros((p,))
    cat_freq = get_cat_index_freq(X_cat)
    # if (old):
    #     cat_level[cat_index] = cat_freq['nlevel'] - 1
    # else:
    cat_level[cat_index] = cat_freq['nlevel']  # ***!
    cat_level[~cat_index] = np.nan
    cat_index_list = create_cat_index_list(cat_level)  # create the index of values and keys, such like {'1':[0,1,2], '3':[3,4,5,6]}

    cat_index_all = reduct_to_vector(cat_index_list)
    d_cat = len(cat_index_all)
    d = len(X_noncat[0]) + d_cat

    # TEST POINT 1 !!!
    # the index of categorical values
    dcat_index = np.zeros((d,)).astype(bool)
    dcat_index[cat_index_all] = True

    #
    dord_index = np.zeros((d,)).astype(bool)
    df = pd.DataFrame(X_noncat)
    ord_in_noncat = df.apply(lambda x: len(np.unique(x)) <= nlevels, axis=0).to_numpy()
    dord_index[~dcat_index] = ord_in_noncat

    # discrete values
    d_index = dcat_index | dord_index
    dord = np.sum(d_index)
    d_index_int, dcat_index_index = np.where(d_index==True)[0].tolist(), np.where(dcat_index==True)[0].tolist()
    cat_in_d = index_int_to_logi(index=[d_index_int.index(ii) for ii in dcat_index_index], l=dord)

    c_index = ~d_index
    cat_input = dict(X_cat=X_cat,
                     cat_index_list=cat_index_list,
                     cat_index_all=cat_index_all, old=old)

    ### 2.1
    # categorical dimensions
    if (any(dcat_index)):
        if ( mu == None):
            # estimating means of categorical variables
            mu_est = get_cat_mu(cat_freq['freq'], old = old, verbose = verbose)  # *?
            mu = mu_est['mu']
    else:
        mu = None

    ### 2.2
    # 初始化上下边界, initialize bounds of lower and upper
    Lower = np.zeros((n, dord))
    Lower[:] = np.nan
    Upper = Lower.copy()

    #
    bounds = get_cat_bounds(X_cat, mu, cat_index_list, check=True, old=old)  # ***
    Lower[:, cat_in_d] = bounds['lower']
    Upper[:, cat_in_d] = bounds['upper']

    #
    bounds = range_transform(X_noncat[:, ord_in_noncat], type = 'ordinal')

    if bounds is not None:
        Lower[:,~cat_in_d] = bounds['Lower']
        Upper[:,~cat_in_d] = bounds['Upper']


    # initialize Z
    if (Z == None):
        Z = initZ(Lower, Upper, X, cat_index, ord_in_noncat, cat_in_d, c_index,
                  dord_index, dcat_index, cat_index_list, Z_cont=None, m=1, method = 'univariate_mean', old = old)

    # initialize corr
    if (corr == None):
        fit_em = em_mixedgc(Z, Lower, Upper,
        d_index=d_index, dcat_index=dcat_index,
        cat_input = cat_input,
        maxit = maxit, eps = eps, runiter=runiter, verbose=verbose,
        trunc_method = trunc_method, n_sample=n_sample, n_update=n_update,)  # ***!

        Zimp = fit_em['Zimp']
        corr = fit_em['corr']
        loglik = fit_em['loglik']
        Z = fit_em['Z']
    else:
        out = latent_operation('fillup',
        Z, Lower, Upper,
        d_index=d_index, dcat_index=dcat_index,
        cat_input = cat_input,
        corr = corr,
        n_update = n_update, n_sample = n_sample, trunc_method = trunc_method)  # ***!
        Zimp = out['Zimp']
        loglik = None


    # Impute X using Imputed Z
    Ximp = latent_to_observed(Zimp, X, mu, cat_labels,
                              cat_index=cat_index, ord_in_noncat=ord_in_noncat,
                              cat_index_all=cat_index_all, cat_index_list=cat_index_list, old=old)
    #
    if (n_MI > 0):
      def call_sample(Z, trunc_method='Iterative', n_MI=1):
        out = latent_operation('sample', Z, Lower, Upper, d_index=d_index, dcat_index=dcat_index,
        cat_input = cat_input, corr = corr, n_update = n_update, n_sample = n_sample, trunc_method = trunc_method, n_MI = n_MI)
        Zfill = out['Zimp_sample']
        return Zfill

      if (fast_MI):
        Z_cont = Z[:, c_index]
        Zinits = initZ(Lower, Upper, X, cat_index, ord_in_noncat, cat_in_d, c_index, dord_index, dcat_index,
        cat_index_list, Z_cont=Z_cont, m=n_MI, method = 'sampling', old = old)
        Zimps = map(call_sample, Zinits)  # no sampling


      def call_ZtoX(Zimp):
        ximp = latent_to_observed(Zimp, X, mu, cat_labels,
                                  cat_index=cat_index, ord_in_noncat=ord_in_noncat,
                                  cat_index_all=cat_index_all, cat_index_list=cat_index_list, old=old)
        return ximp
      Ximp_MI = map(call_ZtoX, Zimps)
    else:
      Ximp_MI = None

    # return values
    noncat_index =  np.setdiff1d(np.arange(p), cat_index_int)
    ord_index = noncat_index[ord_in_noncat]
    var_types = dict(continuous=noncat_index[~ord_in_noncat], ordinal = ord_index,
                     categorical = cat_index_int)
    return dict(Ximp=Ximp, Ximp_MI=Ximp_MI,
                 corr=corr, mu=mu,
                 loglik=loglik,
                 var_types=var_types,
                 cat_index_list=cat_index_list,
                 Z=Z)

def em_fix(Z,  Z_lower, Z_upper,
            d_index, dcat_index, cat_input,n_sample,
            n_update,
            corr, window_size, trunc_method="Iterative", scale_to_corr=True, verbose=False, corr_min_eigen=0.01,):
    err_list = []
    l=0
    R = corr
    while True:
        l = l + 1
        est_iter = latent_operation("em", Z, Z_lower, Z_upper,
                                    d_index=d_index, dcat_index=dcat_index, cat_input=cat_input,
                                    corr=R, window_size=window_size, trunc_method=trunc_method, n_sample=n_sample,
                                    n_update=n_update)
        Z = est_iter['Z']
        Zimp = est_iter['Zimp']
        R1 = est_iter['corr']

        if (scale_to_corr):
            R1 = cov2corr(R1)  # R1 = cov2cor(R1)
            R1 = regularize_corr(R1, corr_min_eigen=corr_min_eigen,
                                 verbose=verbose)
            if (cat_input is not None):
                R1 = project_to_nominal_corr(R1, cat_input['cat_index_list'])

        err = np.linalg.norm(R1 - R, ord='fro') / np.linalg.norm(R, ord='fro')  # err = norm(R1 - R, type = "F")/norm(R, type = "F"), Frobenius范数
        err_list.append(err)

        R = R1


        if (err < 0.01):
            break
        if (l > 10):
            break
        loglik = None
    return dict(corr=R, loglik=loglik, Z=Z, Zimp=est_iter['Zimp'])