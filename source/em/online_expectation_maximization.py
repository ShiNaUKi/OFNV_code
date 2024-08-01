from transforms.online_transform_function import OnlineTransformFunction
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from em.expectation_maximization import ExpectationMaximization
from em.embody import _em_step_body_, _em_step_body, _em_step_body_row
from GCImpute import impute_nominal_cont_gc, get_cat_index_freq, cat_to_integers, \
    index_int_to_logi, create_cat_index_list, reduct_to_vector, logi_to_index_int, \
     get_cat_bounds, range_transform, initZ, em_mixedgc, latent_operation,\
   range_transform_windows, latent_to_observed, adjust_index_list, nominal_z_to_x_col, \
    Ximp_transform, initZ_windows, latent_to_observed_windows, get_cat_index_freq_with_Laplacian, \
    nominal_z_to_x_col_from_zero, em_fix
from ExtendGC_func.estimate_mu import get_cat_mu

import pandas as pd

class OnlineExpectationMaximization(ExpectationMaximization):
    def __init__(self, cont_indices, cat_indices, ord_indices, window_size=200, sigma_init=None):
        self.transform_function = OnlineTransformFunction(cont_indices, ord_indices, window_size=window_size)
        self.window_size = window_size
        # 索引数据
        self.cont_indices = cont_indices
        self.ord_indices = ord_indices
        self.cat_indices = cat_indices
        # we assume boolean array of indices
        p = len(cont_indices)
        # By default, sigma corresponds to the correlation matrix of the permuted dataset (ordinals appear first, then continuous)
        if sigma_init is not None:
            self.sigma = sigma_init
        else:
            self.sigma = np.identity(p)
        # track what iteration the algorithm is on for use in weighting samples
        self.iteration = 1

    def partial_fit_and_predict(self, X_batch, max_workers=4, num_ord_updates=2, decay_coef=0.5, sigma_update=True, marginal_update = True, sigma_out=False):
        """
        Updates the fit of the copula using the data in X_batch and returns the 
        imputed values and the new correlation for the copula

        Args:
            X_batch (matrix): data matrix with entries to use to update copula and be imputed
            max_workers (positive int): the maximum number of workers for parallelism 
            num_ord_updates (positive int): the number of times to re-estimate the latent ordinals per batch
            decay_coef (float in (0,1)): tunes how much to weight new covariance estimates
        Returns:
            X_imp (matrix): X_batch with missing values imputed
        """

        if marginal_update:
            self.transform_function.partial_fit(X_batch)
        res = self._fit_covariance(X_batch, max_workers, num_ord_updates, decay_coef, sigma_update, sigma_out)
        if sigma_out:
            Z_batch_imp, sigma = res
        else:
            Z_batch_imp = res

        Z_imp_rearranged = np.empty(X_batch.shape)
        Z_imp_rearranged[:,self.ord_indices] = Z_batch_imp[:,:np.sum(self.ord_indices)]
        Z_imp_rearranged[:,self.cont_indices] = Z_batch_imp[:,np.sum(self.ord_indices):]
        X_imp = np.empty(X_batch.shape)
        X_imp[:,self.cont_indices] = self.transform_function.partial_evaluate_cont_observed(Z_imp_rearranged, X_batch)
        X_imp[:,self.ord_indices] = self.transform_function.partial_evaluate_ord_observed(Z_imp_rearranged, X_batch)
        #if not update:
            #self.transform_function.window = old_window
            #self.transform_function.update_pos = old_update_pos 
         #   pass
        if sigma_out:
            return Z_imp_rearranged,X_imp, sigma
        else:
            return Z_imp_rearranged,X_imp

    def _fit_covariance(self, X_batch, max_workers=4, num_ord_updates=2, decay_coef=0.5, update=True, sigma_out=False, seed=1):
        """
        Updates the covariance matrix of the gaussian copula using the data 
        in X_batch and returns the imputed latent values corresponding to 
        entries of X_batch and the new sigma

        Args:
            X_batch (matrix): data matrix with which to update copula and with entries to be imputed
            max_workers: the maximum number of workers for parallelism 
            num_ord_updates: the number of times to restimate the latent ordinals per batch
            decay_coef (float in (0,1)): tunes how much to weight new covariance estimates
        Returns:
            sigma (matrix): an updated estimate of the covariance of the copula
            Z_imp (matrix): estimates of latent values in X_batch
        """
        Z_ord_lower, Z_ord_upper = self.transform_function.partial_evaluate_ord_latent(X_batch)
        Z_ord = self._init_Z_ord(Z_ord_lower, Z_ord_upper, seed) # 根据Z_ord = norm.ppf(uniform(norm.cdf(Z_ord_lower), norm.cdf(Z_ord_upper)))
        Z_cont = self.transform_function.partial_evaluate_cont_latent(X_batch)
        # Latent variable matrix with columns sorted as ordinal, continuous
        Z = np.concatenate((Z_ord, Z_cont), axis=1)

        batch_size, p = Z.shape
        # track previous sigma for the purpose of early stopping
        prev_sigma = self.sigma
        Z_imp = np.zeros((batch_size, p))
        C = np.zeros((p, p))
        if max_workers==1:
            C, Z_imp, Z = _em_step_body(Z, Z_ord_lower, Z_ord_upper, prev_sigma, num_ord_updates)
        else:
            divide = batch_size/max_workers * np.arange(max_workers+1)
            divide = divide.astype(int)
            args = [(Z[divide[i]:divide[i+1],:], Z_ord_lower[divide[i]:divide[i+1],:], Z_ord_upper[divide[i]:divide[i+1],:], prev_sigma, num_ord_updates) for i in range(max_workers)]
            # divide each batch into max_workers parts instead of n parts
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                res = pool.map(_em_step_body_, args)
                for i,(C_divide, Z_imp_divide, Z_divide) in enumerate(res):
                    Z_imp[divide[i]:divide[i+1],:] = Z_imp_divide
                    Z[divide[i]:divide[i+1],:] = Z_divide # not necessary if we only do on EM iteration 
                    C += C_divide
        C = C/batch_size
        sigma = np.cov(Z_imp, rowvar=False) + C
        sigma = self._project_to_correlation(sigma)

        if update:
            self.sigma = sigma*decay_coef + (1 - decay_coef)*prev_sigma
            prev_sigma = self.sigma
            self.iteration += 1
        if sigma_out:
            if update:
                sigma = self.get_sigma()
            else:
                sigma = self.get_sigma(sigma*decay_coef + (1 - decay_coef)*prev_sigma)
            return Z_imp, sigma
        else:
            return Z_imp

    def get_sigma(self, sigma=None):
        if sigma is None:
            sigma = self.sigma
        sigma_rearranged = np.empty(sigma.shape)
        sigma_rearranged[np.ix_(self.ord_indices,self.ord_indices)] = sigma[:np.sum(self.ord_indices),:np.sum(self.ord_indices)]
        sigma_rearranged[np.ix_(self.cont_indices,self.cont_indices)] = sigma[np.sum(self.ord_indices):,np.sum(self.ord_indices):]
        sigma_rearranged[np.ix_(self.cont_indices,self.ord_indices)] = sigma[np.sum(self.ord_indices):,:np.sum(self.ord_indices)]
        sigma_rearranged[np.ix_(self.ord_indices,self.cont_indices)] =  sigma_rearranged[np.ix_(self.cont_indices,self.ord_indices)].T
        return sigma_rearranged

    def _init_sigma(self, sigma):
        sigma_new = np.empty(sigma.shape)
        sigma_new[:np.sum(self.ord_indices),:np.sum(self.ord_indices)] = sigma[np.ix_(self.ord_indices,self.ord_indices)]
        sigma_new[np.sum(self.ord_indices):,np.sum(self.ord_indices):] = sigma[np.ix_(self.cont_indices,self.cont_indices)]
        sigma_new[np.sum(self.ord_indices):,:np.sum(self.ord_indices)] = sigma[np.ix_(self.cont_indices,self.ord_indices)] 
        sigma_new[:np.sum(self.ord_indices),np.sum(self.ord_indices):] = sigma[np.ix_(self.ord_indices,self.cont_indices)] 
        self.sigma = sigma_new

    def change_point_test(self, x_batch, decay_coef, nsample=100, max_workers=4):
        n,p = x_batch.shape
        statistics = np.zeros((nsample,3))
        sigma_old = self.get_sigma()
        _, sigma_new = self.partial_fit_and_predict(x_batch, decay_coef=decay_coef, max_workers=max_workers, marginal_update=True, sigma_update=False, sigma_out=True)
        s = self.get_matrix_diff(sigma_old, sigma_new)
        # generate incomplete mixed data samples
        for i in range(nsample):
            np.random.seed(i)
            z = np.random.multivariate_normal(np.zeros(p), sigma_old, n)
            # mask
            x = np.empty(x_batch.shape)
            x[:,self.cont_indices] = self.transform_function.partial_evaluate_cont_observed(z)
            x[:,self.ord_indices] = self.transform_function.partial_evaluate_ord_observed(z)
            loc = np.isnan(x_batch)
            x[loc] = np.nan
            _, sigma = self.partial_fit_and_predict(x, decay_coef=decay_coef, max_workers=max_workers, marginal_update=False, sigma_update=False, sigma_out=True)
            statistics[i,:] = self.get_matrix_diff(sigma_old, sigma)
        # compute test statistics
        pval = np.zeros(3)
        for j in range(3):
            pval[j] = np.sum(s[j]<statistics[:,j])/(nsample+1)
        self._init_sigma(sigma_new)
        return pval, s

        # compute test statistics
    def get_matrix_diff(self, sigma_old, sigma_new, type = 'F'):
        '''
        Return the correlation change tracking statistics, as some matrix norm of normalized matrix difference.
        Support three norms currently: 'F' for Frobenius norm, 'S' for spectral norm and 'N' for nuclear norm. User-defined norm can also be used.
        '''
        p = sigma_old.shape[0]
        u, s, vh = np.linalg.svd(sigma_old)
        factor = (u * np.sqrt(1/s) ) @ vh
        diff = factor @ sigma_new @ factor
        if type == 'F':
            return np.linalg.norm(diff-np.identity(p))
        else:
            _, s, _ = np.linalg.svd(diff)
            if type == 'S':
                return max(abs(s-1))
            if type == 'N':
                return np.sum(abs(s-1))

class OnlineExpectationMaximization_ExtendGC(ExpectationMaximization):
    def __init__(self, cont_indices, cat_indices, ord_indices, window_size=200, cat_range=None, sigma_init=None):
        self.transform_function = OnlineTransformFunction(cont_indices, cat_indices, ord_indices, window_size=window_size)
        self.window_size = window_size
        # 索引数据
        self.cont_indices = cont_indices # bool类型,反应不同类型的特征对应索引
        self.ord_indices  = ord_indices
        self.cat_indices  = cat_indices

        #
        self.cat_range = cat_range

        # we assume boolean array of indices
        p = len(cont_indices)
        # By default, sigma corresponds to the correlation matrix of the permuted dataset (ordinals appear first, then continuous)
        # if sigma_init is not None:
        #     self.sigma = sigma_init
        # else:
        #     self.sigma = np.identity(p)
        if sigma_init == None:
            self.sigma = None       # extendGC的初始化在后面
        # mu, extendGC only
        self.mu = None

        # track what iteration the algorithm is on for use in weighting samples
        self.iteration = 1
        self.maxit = 50
        self.eps = 0.01
        self.runiter = 0
        self.trunc_method = 'Iterative'
        self.n_sample = 5000
        self.n_update = 1


    def _preprocess_extendGC(self, X_batch):
        '''
        数据预处理
        '''
        # if seed != None:
        #     set_seed(seed)
        n = len(X_batch)
        p = len(X_batch[0])
        dim = (n, p)
        old = False
        # logical values to int
        cat_index = self.cat_indices.copy()

        # 检测cat_index是否满足输入格式要求, 并转换为cat_index_int
        if isinstance(cat_index[0], np.bool_):  #
            if len(cat_index) != p:  #
                print('invalid cat_index')
                exit(-1)
            if isinstance(cat_index, np.ndarray) == False:
                cat_index = np.array(cat_index)
            cat_index_int = np.where(cat_index == True)[0].tolist()
        elif min(cat_index) < 0 or max(cat_index) > p - 1:
            print('invalid cat_index')
            exit(-1)
        else:
            cat_index_int = cat_index


        # relabel, 类型值[1,3,5] -> [1,2,3]
        cat_labels = dict()
        for j in cat_index_int:
            # relabel = cat_to_integers(X_batch[:, j])
            # cat_labels[j] = relabel['xlevels']
            cat_labels[j] = np.arange(self.cat_range[j][0], self.cat_range[j][1]+1)
            # for i in range(len(relabel['x'])):
            #     X_batch[:, j][i] = relabel['x'][i]

        # int to logical value
        cat_index = index_int_to_logi(cat_index_int.copy(), p)


        X_cat = X_batch[:, cat_index]
        X_noncat = X_batch[:, ~cat_index]

        # Do not allow empty row
        df = pd.DataFrame(X_batch)
        if not all(df.apply(lambda x: np.sum(~np.isnan(x)), axis=1)):
            print("remove empty row")
            exit(-1)

        # Do not allow column with only one level
        # unique_res = df.apply(lambda x: len(np.unique(x[~np.isnan(x)])) <= 1, axis=0)
        # if any(unique_res):
        #     print(f'remove column with only 0 or 1 unique value, {np.where(unique_res)[0]}')
        #     exit(-1)

        cat_level = np.zeros((p,))
        cat_freq = get_cat_index_freq_with_Laplacian(X_cat, cat_range=self.cat_range)
        # if (old):
        #     cat_level[cat_index] = cat_freq['nlevel'] - 1
        # else:
        cat_level[cat_index] = cat_freq['nlevel']  # ***!
        cat_level[~cat_index] = np.nan
        cat_index_list = create_cat_index_list(cat_level)  # create the index of values and keys, such like {'1':[0,1,2], '3':[3,4,5,6]}

        cat_index_all = reduct_to_vector(cat_index_list)
        d_cat = len(cat_index_all)
        d = len(X_noncat[0]) + d_cat      # length of the vector which considering one-hot encoding.
        self.d = d

        # TEST POINT 1 !!!
        # the index of categorical values
        dcat_index = np.zeros((d,)).astype(bool)
        dcat_index[cat_index_all] = True

        #
        dord_index = np.zeros((d,)).astype(bool)
        df = pd.DataFrame(X_noncat)

        # ord in noncat
        # nlevels = 20
        # ord_in_noncat = df.apply(lambda x: len(np.unique(x)) <= nlevels, axis=0).to_numpy()
        ord_in_noncat = [np.where(~self.cat_indices)[0].tolist().index(ii) for ii in np.where(self.ord_indices)[0].tolist() ]
        ord_in_noncat = index_int_to_logi(ord_in_noncat, sum(~self.cat_indices))
        dord_index[~dcat_index] = ord_in_noncat

        # discrete values
        d_index = dcat_index | dord_index
        dord = np.sum(d_index)
        d_index_int, dcat_index_index = np.where(d_index == True)[0].tolist(), np.where(dcat_index == True)[0].tolist()
        cat_in_d = index_int_to_logi(index=[d_index_int.index(ii) for ii in dcat_index_index], l=dord)

        c_index = ~d_index
        cat_input = dict(X_cat=X_cat,
                         cat_index_list=cat_index_list,
                         cat_index_all=cat_index_all, old=old)

        # the input of EM algorithm
        self.cat_input = cat_input

        # index of categorical values (bool and int style)
        self.cat_index_int = cat_index_int  # int type
        self.cat_index = cat_index          # bool type

        # vectors of X_noncat or X_cat

        self.cat_labels = cat_labels

        # index of categorical values in vectors contains one-hot vectors.
        self.dcat_index = dcat_index
        self.cat_freq = cat_freq


        self.dord = dord
        self.cat_in_d =cat_in_d
        self.X_noncat = X_noncat
        self.X_cat = X_cat

        self.c_index = c_index
        self.d_index = d_index
        self.cat_index_all = cat_index_all
        self.cat_index_list = cat_index_list

        self.cat_index = cat_index
        self.cat_index_int = cat_index_int
        self.ord_in_noncat = ord_in_noncat
        self.dcat_index = dcat_index
        self.dord_index = dord_index

    def partial_fit_and_predict(self, X_batch, max_workers=4, num_ord_updates=2, decay_coef=0.5,
                                sigma_update=True, marginal_update = True, sigma_out=False):
        """
        Updates the fit of the copula using the data in X_batch and returns the
        imputed values and the new correlation for the copula

        Args:
            X_batch (matrix): data matrix with entries to use to update copula and be imputed
            max_workers (positive int): the maximum number of workers for parallelism
            num_ord_updates (positive int): the number of times to re-estimate the latent ordinals per batch
            decay_coef (float in (0,1)): tunes how much to weight new covariance estimates
        Returns:
            X_imp (matrix): X_batch with missing values imputed
        """

        if marginal_update:
            self.transform_function.partial_fit(X_batch) # saving the most n_windows recent rows of data, for marginal updating

        # preprocessing
        self._preprocess_extendGC(X_batch)

        # imputing features
        res = self._fit_covariance_extendGC(X_batch, max_workers, num_ord_updates, decay_coef, sigma_update, sigma_out)
        return res




    def _fit_covariance(self, X_batch, max_workers=4, num_ord_updates=2, decay_coef=0.5, update=True, sigma_out=False, seed=1):
        """
        Updates the covariance matrix of the gaussian copula using the data
        in X_batch and returns the imputed latent values corresponding to
        entries of X_batch and the new sigma

        Args:
            X_batch (matrix): data matrix with which to update copula and with entries to be imputed
            max_workers: the maximum number of workers for parallelism
            num_ord_updates: the number of times to restimate the latent ordinals per batch
            decay_coef (float in (0,1)): tunes how much to weight new covariance estimates
        Returns:
            sigma (matrix): an updated estimate of the covariance of the copula
            Z_imp (matrix): estimates of latent values in X_batch
        """

        old = False
        verbose = False

        n = len(X_batch)

        # 估计均值
        ### 2.1 求无序值的均值, argmax分布
        # categorical dimensions
        # 在线模式下, 上一个windows的mu, 作为下一次mu预测的inits
        if (any(self.dcat_index)):
            if (self.mu == None):
                # estimating means of categorical variables
                mu_est = get_cat_mu(self.cat_freq['freq'], old=old, verbose=verbose, inits=self.mu)  # *?
                self.mu = mu_est['mu']
        else:
            self.mu = None

        Lower = np.zeros((n, self.dord))
        Lower[:] = np.nan
        Upper = Lower.copy()

        # 2.2 获取无序值的边界
        bounds = get_cat_bounds(self.X_cat, self.mu, self.cat_index_list, check=True, old=old)  # ***
        Lower[:, self.cat_in_d] = bounds['lower']
        Upper[:, self.cat_in_d] = bounds['upper']


        ### 2.3 求整形的upper和lower
        bounds = range_transform(self.X_noncat[:, self.ord_in_noncat], type='ordinal')
        if bounds is not None:
            Lower[:, ~self.cat_in_d] = bounds['Lower']
            Upper[:, ~self.cat_in_d] = bounds['Upper']

        # 2.3 初始化Z
        Z = initZ(Lower, Upper, X_batch, self.cat_index, self.ord_in_noncat, self.cat_in_d, self.c_index,
                  self.dord_index, self.dcat_index, self.cat_index_list, Z_cont=None, m=1, method='univariate_mean', old=old)

        batch_size, p = Z.shape

        # 2.4 方差更新
        # track previous sigma for the purpose of early stopping
        prev_sigma = self.sigma
        Z_imp = np.zeros((batch_size, p))
        C = np.zeros((p, p))

        if (prev_sigma is None):
            print("prev_sigma is none")
            fit_em = em_mixedgc(Z, Lower, Upper, window_size=self.window_size,
                                d_index=self.d_index, dcat_index=self.dcat_index,
                                cat_input=self.cat_input,
                                maxit=self.maxit, eps=self.eps, runiter=self.runiter, verbose=verbose,
                                trunc_method=self.trunc_method, n_sample=self.n_sample, n_update=self.n_update, )  # ***!

            Zimp = fit_em['Zimp']
            sigma = fit_em['corr']
            loglik = fit_em['loglik']
            Z = fit_em['Z']
        else:
            # out = latent_operation('fillup',
            #                        Z, Lower, Upper,
            #                        d_index=self.d_index, dcat_index=self.dcat_index,
            #                        cat_input=self.cat_input,
            #                        corr=prev_sigma,
            #                        n_update=self.n_update, n_sample=self.n_sample, trunc_method=self.trunc_method)  # ***!
            print("prev_sigma is not none")
            out = latent_operation('em',
                                   Z, Lower, Upper,
                                   d_index=self.d_index, dcat_index=self.dcat_index,
                                   cat_input=self.cat_input,
                                   corr=prev_sigma,
                                   n_update=self.n_update, n_sample=self.n_sample,
                                   trunc_method=self.trunc_method)  # ***!
            Zimp = out['Zimp']
            C = out['C']
            sigma = np.cov(Z_imp, rowvar=False) + C
            loglik = out['loglik']
            Z = out['Z']

        # corr = corr/batch_size
        # sigma = np.cov(Z_imp, rowvar=False) + C
        sigma = self._project_to_correlation(sigma)

        if update:
            self.sigma = sigma*decay_coef + (1 - decay_coef)*prev_sigma
            prev_sigma = self.sigma
            self.iteration += 1
        if sigma_out:
            if update:
                sigma = self.get_sigma()
            else:
                sigma = self.get_sigma(sigma*decay_coef + (1 - decay_coef)*prev_sigma)
            return Z_imp, sigma
        else:
            return Z_imp


    def _fit_covariance_extendGC(self, X_batch, max_workers=4, num_ord_updates=2, decay_coef=0.5, update=True, sigma_out=False, seed=1):
        """
        Updates the covariance matrix of the gaussian copula using the data
        in X_batch and returns the imputed latent values corresponding to
        entries of X_batch and the new sigma

        Args:
            X_batch (matrix): data matrix with which to update copula and with entries to be imputed
            max_workers: the maximum number of workers for parallelism
            num_ord_updates: the number of times to restimate the latent ordinals per batch
            decay_coef (float in (0,1)): tunes how much to weight new covariance estimates
        Returns:
            sigma (matrix): an updated estimate of the covariance of the copula
            Z_imp (matrix): estimates of latent values in X_batch
        """

        old = False
        verbose = False

        n = len(X_batch)

        # 估计均值
        ### 2.1 Estimating the mu with MC method
        if (any(self.dcat_index)):
            if (self.mu is None):
                # estimating means of categorical variables
                mu_est = get_cat_mu(self.cat_freq['freq'], old=old, verbose=verbose, inits=None)  # *?
            else:
                mu_est = get_cat_mu(self.cat_freq['freq'], old=old, verbose=verbose, inits=self.mu.copy())
            self.mu = mu_est['mu']
        else:
            self.mu = None

        Lower = np.zeros((n, self.dord))
        Lower[:] = np.nan
        Upper = Lower.copy()

        # 2.2 bounds of nonoimal
        bounds = get_cat_bounds(self.X_cat, self.mu, self.cat_index_list, check=True, old=old, cat_range=self.cat_range)  # ***
        Lower[:, self.cat_in_d] = bounds['lower']
        Upper[:, self.cat_in_d] = bounds['upper']


        ### 2.3 get the lower and upper of ordinal features
        bounds = range_transform_windows(self.X_noncat[:, self.ord_in_noncat], type='ordinal', windows=self.transform_function.window[:, self.ord_indices])
        if bounds is not None:
            Lower[:, ~self.cat_in_d] = bounds['Lower']
            Upper[:, ~self.cat_in_d] = bounds['Upper']

        # 2.4 initalize Z
        Z = initZ_windows(Lower, Upper, X_batch, self.cat_index, self.ord_in_noncat, self.cat_in_d, self.c_index,
                  self.dord_index, self.dcat_index, self.cat_index_list, self.transform_function.window[:,self.cont_indices],
                  Z_cont=None, m=1, method='univariate_mean', old=old)

        if self.sigma is None:
            self.sigma = np.identity(len(Z[0]))     # extendGC的方差

        batch_size, p = Z.shape

        # 2.4 方差更新
        # track previous sigma for the purpose of early stopping
        prev_sigma = self.sigma
        Z_imp = np.zeros((batch_size, p))
        # fitting /  start={'R':prev_sigma.copy()} -> start=None
        # fit_em = em_mixedgc(Z, Lower, Upper, window_size=self.window_size,
        #                     d_index=self.d_index, dcat_index=self.dcat_index,
        #                     cat_input=self.cat_input, start=None,
        #                     maxit=self.maxit, eps=self.eps, runiter=self.runiter, verbose=verbose,
        #                     trunc_method=self.trunc_method, n_sample=self.n_sample, n_update=self.n_update, )  # ***!
        #
        # Zimp = fit_em['Zimp']
        # sigma = fit_em['corr']
        # loglik = fit_em['loglik']
        # Z = fit_em['Z']
        # if (prev_sigma is None):
        #     fit_em = em_mixedgc(Z, Lower, Upper, window_size=self.window_size,
        #                         d_index=self.d_index, dcat_index=self.dcat_index,
        #                         cat_input=self.cat_input,
        #                         maxit=self.maxit, eps=self.eps, runiter=self.runiter, verbose=verbose,
        #                         trunc_method=self.trunc_method, n_sample=self.n_sample,
        #                         n_update=self.n_update, )  # ***!
        #
        #     Zimp = fit_em['Zimp']
        #     sigma = fit_em['corr']
        #     loglik = fit_em['loglik']
        #     print(f"prev_sigma is none, loglik = {loglik}")
        #     Z = fit_em['Z']
        # else:
        #     out = latent_operation('em',
        #                            Z, Lower, Upper,
        #                            d_index=self.d_index, dcat_index=self.dcat_index,
        #                            cat_input=self.cat_input,
        #                            corr=prev_sigma, window_size=self.window_size,
        #                            n_update=self.n_update, n_sample=self.n_sample,
        #                            trunc_method=self.trunc_method)  # ***!
        #     Zimp = out['Zimp']
        #     C = out['C']
        #     sigma = np.cov(Z_imp, rowvar=False) + C
        #     loglik = out['loglik']
        #     Z = out['Z']
        #     print(f"prev_sigma is not none, loglik = {loglik}")

        fit_em =  em_fix(Z, Z_lower=Lower, Z_upper=Upper,
                   d_index=self.d_index, dcat_index=self.dcat_index, cat_input=self.cat_input,
                   corr=prev_sigma, window_size=self.window_size, trunc_method=self.trunc_method, n_sample=self.n_sample,
                   n_update=1, scale_to_corr=True, verbose=False, corr_min_eigen=0.01)

        sigma =  fit_em['corr']       #dict(corr=R, loglik=loglik, Z=Z, Zimp=est_iter['Zimp'])
        loglik = fit_em['loglik']
        Zimp = fit_em['Zimp']

        #C = C/batch_size
        #sigma = np.cov(Z_imp, rowvar=False) + C
        # sigma = self._project_to_correlation(sigma)

        # transform the non-categorical variables to the observed vectors
        ## imp1, one-hot coding to normal coding (int)
        ## imp2, one-hot coding to latent coding (float)
        Z_imp1, Z_imp2 = np.zeros((batch_size, len(X_batch[0]))), np.zeros((batch_size, len(X_batch[0]))) # 两种映射形式
        Z_imp1[:], Z_imp2[:] = np.nan, np.nan
        tmp_index = list(range(len(Zimp[0])))
        list(map(tmp_index.remove, self.cat_index_all))
        Z_imp1[:, ~self.cat_index] = Zimp[:, tmp_index]
        Z_imp2[:, ~self.cat_index] = Zimp[:, tmp_index]

        # 将无序值的one-hot编码映射为序数值
        def one_hot_to_val1(Z_cat, X_cat, cat_index_list, cat_range=None, old=False):
            if (len(Z_cat[0]) != sum(map(len, cat_index_list.values()))):
                print('something wrong')
                exit(-1)
            cat_index_list = adjust_index_list(cat_index_list)  # {1:[5,6,7], 2:[8,9,10]} -> {1:[0,1,2], 2:[3,4,5]}
            cat_range_values = list(cat_range.values())
            Ximp_cat = X_cat.copy()
            for j, (k, v) in enumerate(cat_index_list.items()):
                start_from = cat_range_values[j][0]
                index_m = np.isnan(X_cat[:, j])
                if any(index_m):
                    index_cat = list(v)  # index_cat = cat_index_list[j]
                    zmis = Z_cat[np.ix_(index_m, index_cat)]
                    if start_from != 0: #range of value starting from what 0 or 1 ?
                        Ximp_cat[index_m, j] = np.apply_along_axis(nominal_z_to_x_col, axis=1,
                                                                   arr=zmis)  # apply(zmis, 1, nominal_z_to_x_col)
                    else:
                        Ximp_cat[index_m, j] = np.apply_along_axis(nominal_z_to_x_col_from_zero, axis=1,
                                                                       arr=zmis)  # apply(zmis, 1, nominal_z_to_x_col)
            return Ximp_cat

        # 将无序特征转换为latent值
        def select_latent(z, x, start_from_zero=False, index=None):
            if np.isnan(x):
                argmax = np.argmax(z)  # 对齐
                return z[argmax]
            else:
                if not start_from_zero:
                    return z[int(x - 1)]
                else:
                    return z[int(x)]

        # 将无序值的one-hot编码变为latent值
        def one_hot_to_val2(Z_cat, X_cat, cat_index_list, cat_range=None, old=False):
            cat_range_values = list(cat_range.values())
            if (len(Z_cat[0]) != sum(map(len, cat_index_list.values()))):
                print('something wrong')
                exit(-1)
            cat_index_list = adjust_index_list(cat_index_list)
            Ximp_cat = X_cat.copy()
            for j, (k, v) in enumerate(cat_index_list.items()):
                # index_m = np.isnan(X_cat[:, j])
                start_from = cat_range_values[j][0]
                index_m = np.arange(len(X_cat))
                index_cat = list(v)  # index_cat = cat_index_list[j]
                z_tmp = Z_cat[np.ix_(index_m, index_cat)]

                # Ximp_cat[:, j] = np.apply_along_axis(nominal_z_to_x_col, axis=1,
                #                                            arr=z_tmp)  # apply(zmis, 1, nominal_z_to_x_col)
                for i in range(len(z_tmp)):
                    Ximp_cat[i, j] = select_latent(z_tmp[i], Ximp_cat[i, j], start_from_zero=(start_from==0), index=j)
            return Ximp_cat

        Z_imp1[:, self.cat_index] = one_hot_to_val1(Z_cat=Zimp[:,self.cat_index_all], X_cat=self.X_cat, cat_index_list=self.cat_index_list, cat_range=self.cat_range)
        Z_imp2[:, self.cat_index] = one_hot_to_val2(Z_cat=Zimp[:,self.cat_index_all], X_cat=self.X_cat, cat_index_list=self.cat_index_list, cat_range=self.cat_range)

        # maping the latent vector Z to the observed vector X
        Ximp = latent_to_observed_windows(Zimp, X_batch, self.mu, self.cat_labels,
                                  cat_index=self.cat_index, ord_in_noncat=self.ord_in_noncat,
                                  cat_index_all=self.cat_index_all, cat_index_list=self.cat_index_list,
                                          windows=self.transform_function.window[:, ~self.cat_indices], old=old, cat_range=self.cat_range)

        if update:
            if prev_sigma is not None:
                self.sigma = sigma*decay_coef + (1 - decay_coef)*prev_sigma
            else:
                self.sigma = sigma
            prev_sigma = self.sigma
            self.iteration += 1
        if sigma_out:
            # if update:
            #     sigma = self.get_sigma()
            # else:
            #     sigma = self.get_sigma(sigma*decay_coef + (1 - decay_coef)*prev_sigma)
            sigma = self.sigma
            return Zimp, Z_imp1, Z_imp2, Ximp, sigma
        else:
            return Zimp, Z_imp1, Z_imp2, Ximp

    def get_sigma(self, sigma=None):
        if sigma is None:
            sigma = self.sigma
        sigma_rearranged = np.empty(sigma.shape)
        sigma_rearranged[np.ix_(self.ord_indices,self.ord_indices)] = sigma[:np.sum(self.ord_indices),:np.sum(self.ord_indices)]
        sigma_rearranged[np.ix_(self.cont_indices,self.cont_indices)] = sigma[np.sum(self.ord_indices):,np.sum(self.ord_indices):]
        sigma_rearranged[np.ix_(self.cont_indices,self.ord_indices)] = sigma[np.sum(self.ord_indices):,:np.sum(self.ord_indices)]
        sigma_rearranged[np.ix_(self.ord_indices,self.cont_indices)] =  sigma_rearranged[np.ix_(self.cont_indices,self.ord_indices)].T
        return sigma_rearranged

    def _init_sigma(self, sigma):
        sigma_new = np.empty(sigma.shape)
        sigma_new[:np.sum(self.ord_indices),:np.sum(self.ord_indices)] = sigma[np.ix_(self.ord_indices,self.ord_indices)]
        sigma_new[np.sum(self.ord_indices):,np.sum(self.ord_indices):] = sigma[np.ix_(self.cont_indices,self.cont_indices)]
        sigma_new[np.sum(self.ord_indices):,:np.sum(self.ord_indices)] = sigma[np.ix_(self.cont_indices,self.ord_indices)]
        sigma_new[:np.sum(self.ord_indices),np.sum(self.ord_indices):] = sigma[np.ix_(self.ord_indices,self.cont_indices)]
        self.sigma = sigma_new

    def change_point_test(self, x_batch, decay_coef, nsample=100, max_workers=4):
        n,p = x_batch.shape
        statistics = np.zeros((nsample,3))
        sigma_old = self.get_sigma()
        _, sigma_new = self.partial_fit_and_predict(x_batch, decay_coef=decay_coef, max_workers=max_workers, marginal_update=True, sigma_update=False, sigma_out=True)
        s = self.get_matrix_diff(sigma_old, sigma_new)
        # generate incomplete mixed data samples
        for i in range(nsample):
            np.random.seed(i)
            z = np.random.multivariate_normal(np.zeros(p), sigma_old, n)
            # mask
            x = np.empty(x_batch.shape)
            x[:,self.cont_indices] = self.transform_function.partial_evaluate_cont_observed(z)
            x[:,self.ord_indices] = self.transform_function.partial_evaluate_ord_observed(z)
            loc = np.isnan(x_batch)
            x[loc] = np.nan
            _, sigma = self.partial_fit_and_predict(x, decay_coef=decay_coef, max_workers=max_workers, marginal_update=False, sigma_update=False, sigma_out=True)
            statistics[i,:] = self.get_matrix_diff(sigma_old, sigma)
        # compute test statistics
        pval = np.zeros(3)
        for j in range(3):
            pval[j] = np.sum(s[j]<statistics[:,j])/(nsample+1)
        self._init_sigma(sigma_new)
        return pval, s

        # compute test statistics

    def get_matrix_diff(self, sigma_old, sigma_new, type = 'F'):
        '''
        Return the correlation change tracking statistics, as some matrix norm of normalized matrix difference.
        Support three norms currently: 'F' for Frobenius norm, 'S' for spectral norm and 'N' for nuclear norm. User-defined norm can also be used.
        '''
        p = sigma_old.shape[0]
        u, s, vh = np.linalg.svd(sigma_old)
        factor = (u * np.sqrt(1/s) ) @ vh
        diff = factor @ sigma_new @ factor
        if type == 'F':
            return np.linalg.norm(diff-np.identity(p))
        else:
            _, s, _ = np.linalg.svd(diff)
            if type == 'S':
                return max(abs(s-1))
            if type == 'N':
                return np.sum(abs(s-1))

