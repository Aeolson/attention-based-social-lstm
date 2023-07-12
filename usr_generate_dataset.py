import os
import pickle
from utils import *

if __name__ == '__main__':
    
    dir_pkl = './data_pkl/' + args['use_dataset'] + '/'
    if not os.path.exists(dir_pkl):
        os.makedirs(dir_pkl)
    
    print("load train dataset .............................")
    if not os.path.exists(dir_pkl+'train.pkl'):
        train_set = Dataset(data_name=args['use_dataset'], data_type='train', batch_size=500)
        train_set.run()
        with open(dir_pkl+'train.pkl', 'wb') as f:
            pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(dir_pkl+'train.pkl', 'rb') as f:
            train_set = pickle.load(f)

    print("load valid dataset .............................")
    if not os.path.exists(dir_pkl+'valid.pkl'):
        valid_set = Dataset(data_name=args['use_dataset'], data_type='valid', batch_size=500)
        valid_set.run()
        with open(dir_pkl+'valid.pkl', 'wb') as f:
            pickle.dump(valid_set, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(dir_pkl+'valid.pkl', 'rb') as f:
            valid_set = pickle.load(f)

    print("load test dataset .............................")
    if not os.path.exists(dir_pkl+'test.pkl'):
        test_set = Dataset(data_name=args['use_dataset'], data_type='test', batch_size=500)
        test_set.run()
        with open(dir_pkl+'test.pkl', 'wb') as f:
            pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(dir_pkl+'test.pkl', 'rb') as f:
            test_set = pickle.load(f)

    print("load test long trajectories ...................")
    if not os.path.exists(dir_pkl+'test_long.pkl'):
        test_trj = Dataset(data_name=args['use_dataset'], data_type='test', batch_size=0)
        test_trj.run()
        with open(dir_pkl+'test_long.pkl', 'wb') as f:
            pickle.dump(test_trj, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(dir_pkl+'test_long.pkl', 'rb') as f:
            test_trj = pickle.load(f)
