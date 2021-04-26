import sys
import numpy as np
from functions import parseFilters, create_save_features, train_mode, predict_mode

test_dir = sys.argv[1] # 'data/output_RXJ1131_train/'
filters = parseFilters(sys.argv[2])
model_fname = sys.argv[3]

# create bspline features (just call it once. It is a bit time-consuming)
#create_save_features(dname_train)

print("predict on test data...")
output_fname = 'predictions.dat'
res_test = predict_mode(test_dir, filters, model_fname, output_fname)
