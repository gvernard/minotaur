import sys
import numpy as np
from functions import parseFilters, create_save_features, train_mode, predict_mode

train_dir = sys.argv[1] # 'data/output_RXJ1131_train/'
filters = parseFilters(sys.argv[2])

# create bspline features (just call it once. It is a bit time-consuming)
#create_save_features(dname_train)

# training of the predictive model
model_fname = 'mnr_coef.dat'
train_mode(train_dir, filters, model_fname)

print("predict on training data...")
output_fname = 'predict_on_training_data.dat'
res_train = predict_mode(train_dir, filters, model_fname, output_fname)
