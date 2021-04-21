import os
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

def create_save_features(dname):

    '''
    Constructs the features per filter per time-series as well as the event
    target vaiable and saves both into separate files inside the time-series
    folders.
    '''

    ts = [i+'/' for i in sorted(os.listdir(dname)) if os.path.isdir(os.path.join(dname, i))]

    P = len(ts)
    K = 6

    for p in range(P):
        print(ts[p])
        # find max final time
        t_max = 0
        for k in range(1,K+1):
            if os.path.isfile(dname + ts[p] + '/f_' + str(k) + '.dat'):
                A = np.loadtxt(dname + ts[p] + '/f_' + str(k) + '.dat')
                maxA = np.max(A[:,0])

                if maxA > t_max: # check forthe max value
                    t_max = maxA

    # make dataset
        for k in range(1,K+1):
            if os.path.isfile(dname + ts[p] + '/f_' + str(k) + '.dat'): # if not already computed
                # call the function that creates the features and the target variable
                dataset_t, dataset_X, dataset_y = make_dataset(dname, ts[p]+'/', 'f_'+str(k)+'.dat', t_max)
                M, L = dataset_X.shape

                # save the features
                fid = open(dname + ts[p] + '/features_' + str(k) + '.dat', 'w')
                for ii in range(M): # for each row
                    fid.write('%.4f,' % (dataset_t[ii])) # save time-points
                    for jj in range(L):
                        if jj != L-1:
                            fid.write('%.4f,' % (dataset_X[ii,jj]))
                        else:
                            fid.write('%.4f\n' % (dataset_X[ii,jj]))
                fid.close()

                # save the target variable
                if k==1:
                    fid = open(dname + ts[p] + 'target_variable.dat', 'w')
                    for ii in range(M):
                        fid.write('%.4f,%.4f\n' % (dataset_t[ii], dataset_y[ii]))
                    fid.close()
    
    return 0


def make_dataset(dname, ts_nm, filt_nm, t_max):

    '''
    Create the (t,X,y) dataset (t:time-points, X:features, y: target variable)
    using the time-series given by ts_nm and the filters given by filt_nm.
    It is somewhat slow because it creates the basis splines repeatedly.
    The maximum length of the time-series is restricted to 100K days.
    '''

    if not os.path.isdir('./bspline/'):
        os.mkdir('./bspline/')

    P = 1#len(ts_nm)
    K = 1#len(filt_nm)

    # Hyperparameters (Moved to a json file?)
    Ts = 1 # starting time (horizon of interest)
    Tf = 60 # final time (horizon of interest). Horizon [Ts, Tf] consitutes the next 60 days
    Tw = 150 # window length (historical data). Look 150 days back.
    step = 5 # how often to save the features (in days)

    # basis functions (b-splines)
    ti = [1, Tw/4, Tw/2, 3*Tw/4, Tw] # user-defined knots
    tii = [ti[0]] + [ti[0]] + [ti[0]] + ti + [ti[-1]] + [ti[-1]] + [ti[-1]] # augmentation size depends on the order of splines
    L = len(ti)+2

    # initialize
    X = np.zeros((int(np.ceil(100000*P/step)), K*L)) # feature matrix. Maximum number of days in a time-series is 100K.
    y = np.zeros((X.shape[0], 1)) #target variable (binary)
    t = np.zeros_like(y) # time-points of each feature vector/target

    cnt = 0
    flag = 1
    dis_charged = np.ceil(30/step) # After an event happens, we do not save features for the next 30 days
    dis_cnt = dis_charged

    # run the dataset (ie, feature and target variables)
    #for p in range(P):
        # load all filters
    #t_cur = []
    #x_cur = []
    #    for k in range(K):
    A = np.loadtxt(dname + ts_nm + filt_nm)
    t_cur = np.round(A[:,0]) # time-points
    x_cur = A[:,1] # filter

        # load peaks
        #peaks = np.loadtxt(dname + ts_nm[p-1] + 'event_flags_yp.dat') # mine
    peaks = np.round(np.loadtxt(dname + ts_nm + 'event_flags.dat', usecols=0)) # Giorgos
    #if peaks.size != 0:
    #    peaks = np.round(peaks[:,0]) # keep only the time of the event

    for i in range(Tw,int(np.round(t_max))+1,step):
        #for k in range(K):
        t_hist, hist_idx, _ = np.intersect1d(t_cur, np.arange(i-Tw+1, i+1), return_indices=True) # find the measurements inside the current historical window.
        #                                                                          Each filter has measurements at different time-points (must be integers).
        phi, _, _ = bspline_basismatrix(4, tii, t_hist-(i-Tw+1)) # phi: basis functions

        x_tmp = x_cur
        x_tmp = x_tmp[hist_idx] #time-series values to be used for the estimation of the features

        # basis functions coefficient estimation (l2 regularization has been added)
        c = np.dot(np.linalg.inv(np.dot(phi.T, phi) + 0.001*np.identity(L)), np.dot(phi.T, x_tmp))

        # plt.plot(t_hist,x_tmp, '.') # check reconstruction accuracy
        # plt.plot(t_hist,phi*c,'o')
        # plt.show(block=False); plt.pause(3); plt.close()

        # feature vector
        X[cnt, np.arange(L)] = c

        # time-point
        t[cnt] = i

        # target variable
        if np.intersect1d(np.arange(i+Ts, i+Tf+1), peaks): # If an event is inside the horizon window (peaks must contain integer values)
            y[cnt] = 1

        # update counters
        if dis_cnt == dis_charged:
            cnt += 1
        else:
            dis_cnt -= 1

        # trigger discharged counter
        if cnt>2 and y[cnt-1]==0 and y[cnt-2]==1 and dis_cnt==dis_charged and flag:
            dis_cnt -= 1

        #reset flag
        if cnt>2 and y[cnt-1]==0 and y[cnt-2]==0:
            flag = 1

        #reset discharged counter
        if dis_cnt <0:
            flag = 0
            cnt -= 1
            dis_cnt = dis_charged

    # keep only the data with non-zero values
    dataset_t = t[np.arange(0,cnt)]
    dataset_X = X[np.arange(0,cnt), :]
    dataset_y = y[np.arange(0,cnt)]

    # return data
    return dataset_t, dataset_X, dataset_y

def bspline_basismatrix(n, t, x=None):
    '''
    B-spline basis function value matrix B(n) for x.

    Input arguments:
    n:
        B-spline order (2 for linear, 3 for quadratic, etc.)
    t:
        knot vector
    x (optional, default=None):
        an m-dimensional vector of values where the basis function is to be
        evaluated

    Output arguments:
    B:
        a matrix of m rows and numel(t)-n columns

    Copyright 2010 Levente Hunyadi
    '''

    if isinstance(x, np.ndarray):
        B =np.zeros((len(x),len(t)-n))
        Bdot = np.zeros_like(B)
        for j in range(0, len(t)-n):
            B_tmp, Bdot_tmp, _ = bspline_basis(j,n,t,x)
            B[:,j] = B_tmp
            Bdot[:,j] = Bdot_tmp
    else:
        b, x = bspline_basis(0,n,t)
        B = np.zeros((len(x), len(t)-n))
        B[:,0] = b
        for j in range(1, len(t)-n):
            B[:,j], _, _ = bspline_basis(j,n,t,x)

    return B, Bdot, x

def bspline_basis(j,n,t,x=None):
    '''
    B-spline basis function value B(j,n) at x.

    Input arguments:
    j:
        interval index, 0 =< j < numel(t)-n
    n:
        B-spline order (2 for linear, 3 for quadratic, etc.)
    t:
        knot vector
    x (optional):
        value where the basis function is to be evaluated

    Output arguments:
    y:
        B-spline basis function value, nonzero for a knot span of n

    Copyright 2010 Levente Hunyadi
    '''

    if not isinstance(x, np.ndarray):
        x = np.linspace(t[n], t[-1-n], 100)

    y, ydot = bspline_basis_recurrence(j, n, t, x)

    return y, ydot, x


def bspline_basis_recurrence(j, n, t, x):

    y = np.zeros_like(x)
    ydot = np.zeros_like(y)
    if n > 1:
        b, _, _ = bspline_basis(j,n-1,t,x)
        dn = x - t[j]
        dd = t[j+n-1] - t[j]
        if dd != 0: # indeterminate forms 0/0 are deemed to be zero
            y = y + b*(dn/dd)
            ydot = ydot + b*((n-1)/dd)
        b, _, _ = bspline_basis(j+1,n-1,t,x)
        dn = t[j+n] - x
        dd = t[j+n] - t[j+1]
        if dd != 0:
            y = y + b*(dn/dd)
            ydot = ydot + b*((n-1)/dd)
    elif t[j+1] < t[-1]: # treat last element of knot vector as a special case
        y[ (t[j] <= x) & (x < t[j+1]) ] = 1
    else:
        y[t[j] <= x] = 1

    return y, ydot

def train_mode(dname, filters, fname):
    '''
    Train the predictive model and save it
    '''

    # get the training time-series
    ts = [i+'/' for i in sorted(os.listdir(dname)) if os.path.isdir(os.path.join(dname, i))]

    # create the dataset
    dataset_t, dataset_X, dataset_y = make_dataset_fast(dname, ts, filters)

    # training of the predictive model
    B_coef, B_bias, res_t, res_y_hat, res_auc, res_fpr, res_tpr = train_minotaur(dataset_t, dataset_X, dataset_y) # fixed

    print('AUC on Training Set: %.4f\n' % (res_auc,)) # fixed

    # save the coefficients
    fid = open(fname, 'w')
    for ii in range(len(B_coef)): # fix
        fid.write('%.7f %.7f\n' % (B_coef[ii],B_bias[0])) # fixed
    fid.close()
    
    return 0

def predict_mode(dname, filters, fname, fname_out):
    '''
    Make predictions for all time-series in dname using the filters and the
    predictive model savedn in fname. Save the results in fname_out file
    '''

    # get the time-series
    ts = [i+'/' for i in sorted(os.listdir(dname)) if os.path.isdir(os.path.join(dname, i))]

    # create the dataset
    dataset_t, dataset_X, dataset_y = make_dataset_fast(dname, ts, filters)
    
    # load model
    B_coef, B_bias = np.loadtxt(fname, unpack=True) # fixed
    B_coef = B_coef[np.newaxis, :]
    B_bias = B_bias[0, np.newaxis]

    # predictions on (new) data
    res_y_hat, res_auc, res_fpr, res_tpr, res_y_bin, res_y_cat = predict_x_event(dataset_t, dataset_X, dataset_y, B_coef, B_bias, fname_out) # fixed

    print('AUC on Testing SetL %.4f\n' % (res_auc,)) # fixed

    return 0

def make_dataset_fast(dname, ts_nm, filt_nm):
    '''
    Create the (t,X,y) dataset (t:time-points, X:features, y: target variable)
    using the time-series given by ts_nm and the filters given by filt_nm.
    It is fast because it uses precomputed features using the create_save_features.m
    function.
    '''

    P = len(ts_nm)
    K = len(filt_nm)

    # initialize
    t = np.asarray([])
    X = np.asarray([])
    y = np.asarray([])

    # create the dataset from the saved files (ie, time-points, feature and target variables)

    for p in range(P): # for each time-series in ts_nm
        X_ = None
        for k in range(K): # for each filter in filt_nm
            filt_cur = filt_nm[k]
            X_tmp = np.loadtxt(dname + ts_nm[p] + 'features' + filt_cur[1:], delimiter=',')
            t_tmp = X_tmp[:,0] # time-points
            X_tmp = X_tmp[:,1:] # features
            

            # enforce consistency on matrices size ( Not anymore necessary but
            # keep it for robustness)
            if k>0:
                M1, L1 = X_.shape
                M2, L2 = X_tmp.shape
                if M1 >= M2:
                    X_ = np.concatenate([X_, np.concatenate([X_tmp, np.zeros((M1-M2,L2))], axis=0)], axis=1)
                else:
                    X_ = np.concatenate([np.concatenate([X_, np.zeros((M2-M1, L1))], axis=0), X_tmp], axis=1)
            else:
                X_ = X_tmp

        if X.any():
            X = np.concatenate([X,X_], axis=0)
        else:
            X = X_

        # time-points
        M3 = len(t_tmp)
        if M3 < max(M1,M2):
            t = np.concatenate([t, np.concatenate([t_tmp, np.zeros((max(M1,M2)-M3,1))], axis=1)], axis=1)
        else:
            t = np.concatenate([t, t_tmp[:max(M1,M2)]], axis=0)

        # target variable
        y_ = np.loadtxt(dname + ts_nm[p] + 'target_variable.dat', delimiter=',')
        y_ = y_[:,1] # discard time-points
        M3 = len(y_)
        if M3 < max(M1,M2):
            t = np.concatenate([y, np.concatenate([y_, np.zeros((max(M1,M2)-M3,1))], axis=1)], axis=1)
        else:
            y = np.concatenate([y, y_[:max(M1,M2)]], axis=0)
    dataset_t = t
    dataset_X = X
    dataset_y = y

    return dataset_t, dataset_X, dataset_y

def train_minotaur(dataset_t, dataset_X, dataset_y):
    '''
    Train a logistic regression model (generalized linear model) using the
    data inside the dataset structure.
    '''

    X = dataset_X
    y = dataset_y

    res_t = dataset_t

    # estimation of the logistic regression model, if not provided as an argument
    y_ = y
    y_[y==0] = 0
    
    # do a grid search for best parameters estimator
    #param_grid = [ {'C': [1, 10, 50, 100, 500, 1000, 5000], 'solver': ['lbfgs'], 'multi_class': ['multinomial']} ]
    #model = GridSearchCV(LogisticRegression(max_iter=3000), param_grid, scoring='roc_auc').fit(X, y_)
    #print(model.best_params_)
    #model = model.best_estimator_

    #model = LogisticRegression(C=100,solver='liblinear',intercept_scaling=72,multi_class='ovr',max_iter=2000).fit(X, y_)
    model = LogisticRegression(C=100, max_iter=10000, solver='lbfgs', multi_class='multinomial', tol=1e-9).fit(X, y_)
    
    #B_coef = np.asarray([-4.3729446,5.1902388,-11.4489957,19.6962230,-25.1415279,27.4224261,-50.3893813,2.6322106,1.5769867,-5.0720556,10.5173248,0.0028168,-50.0765307,86.7147143,-0.4655087,0.4072983,0.1696314,-0.9254681,-2.1243917,-1.4939199,-2.4948536])
    B_coef = model.coef_[0]
    #B_bias = np.asarray([-5.5368275])
    B_bias = model.intercept_

    # reconstruction
    y_hat = model.predict_proba(X)
    y_hat = y_hat[:,1]
    res_y_hat = y_hat

    z1, z2, _ = roc_curve(y, y_hat, pos_label=1)
    auc = roc_auc_score(y, y_hat)

    res_auc = auc
    res_fpr = z1 # false positive rate (false alarm)
    res_tpr = z2 # true positive rate (hit)

    return B_coef, B_bias, res_t, res_y_hat, res_auc, res_fpr, res_tpr
   
@ignore_warnings(category=ConvergenceWarning)
def predict_x_event(dataset_t, dataset_X, dataset_y, B_coef, B_bias, fname):
    '''
    Performs prediction for the data inside the dataset structure and
    computes the AUC performance of the predictive model.
    '''

    X = dataset_X
    y = dataset_y
    t = dataset_t

    thres = 0.99 # might change
    lvls = [0.2, 0.8, 0.95]

    # load model (new)
    model = LogisticRegression(max_iter=1).fit(X, y)
    #model.classes_ = [1, 2]
    model.coef_ = B_coef
    model.intercept_ = B_bias


    y_hat = model.predict_proba(X)
    y_hat = y_hat[:,1]
    res_y_hat = y_hat

    z1, z2, _ = roc_curve(y, y_hat, pos_label=1)
    auc = roc_auc_score(y, y_hat)

    res_auc = auc
    res_fpr = z1 # false positive rate (false alarm)
    res_tpr = z2 # true positive rate (hit)

    res_y_bin = y_hat < thres

    y_cat = np.zeros(len(y_hat))
    y_cat[ (y_hat > lvls[0]) & (y_hat < lvls[1]) ] = 1
    y_cat[ (y_hat > lvls[1]) & (y_hat < lvls[2]) ] = 2
    y_cat[ y_hat > lvls[2] ] = 3
    res_y_cat = y_cat

    # save
    fid = open(fname, 'w')
    for ii in range(len(y)):
        fid.write('%.4f, %.4f, %.4f, %.4f, %.4f\n' % (t[ii], y[ii], y_hat[ii], res_y_bin[ii], y_cat[ii]))
    fid.close()

    return res_y_hat, res_auc, res_fpr, res_tpr, res_y_bin, res_y_cat 
