import numpy as np
from functions import create_save_features, train_mode, predict_mode

dname_train = 'data/output_RXJ1131_train/' # folder with time-series folders
dname_test  = 'data/output_RXJ1131_test/'

# create bspline features (just call it once. It is a bit time-consuming)
#create_save_features(dname_train)
#create_save_features(dname_test)

# select which filters to use
all_filters = ['f_1.dat','f_2.dat','f_3.dat','f_4.dat','f_5.dat','f_6.dat'] # all six filters
filters = all_filters #['f_1.dat','f_3.dat','f_6.dat'] # chosen subset of filters

# training of the predictive model
model_fname = dname_train + 'mnr_coef.dat'
train_mode(dname_train, filters, model_fname)

print("predict on training data...")
output_fname = dname_train + 'results_trained_on_RXJ1131.dat'
res_train = predict_mode(dname_train, filters, model_fname, output_fname)

print("predict on test data...")
output_fname = dname_test + 'results_trained_on_RXJ1131.dat'
res_test = predict_mode(dname_test, filters, model_fname, output_fname)

#recon_08_08 = np.loadtxt(output_fname, delimiter=',')
#idx = np.where(np.diff(recon_08_08[:,0])<0)
