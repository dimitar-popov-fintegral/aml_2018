import os
from subprocess import *
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

import research.svm_data as sdt
import data as dt


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
SVM_LIB_PATH = os.path.abspath(os.path.join(PROJECT_ROOT,'..','libsvm-3.23'))

def run(train_pathname, test_pathname, balance):

    # svm, grid, and gnuplot executable files
    svmscale_exe = os.path.join(SVM_LIB_PATH, "svm-scale")
    svmtrain_exe = os.path.join(SVM_LIB_PATH, "svm-train")
    svmpredict_exe = os.path.join(SVM_LIB_PATH, "svm-predict")
    check_py = os.path.join(SVM_LIB_PATH, 'tools', "checkdata.py")
    grid_py = os.path.join(PROJECT_ROOT, 'research', "grid.py")

    assert os.path.exists(svmscale_exe),"svm-scale executable not found"
    assert os.path.exists(svmtrain_exe),"svm-train executable not found"
    assert os.path.exists(svmpredict_exe),"svm-predict executable not found"
    assert os.path.exists(grid_py),"grid.py not found"
    assert os.path.exists(train_pathname),"training file not found"

    file_name = train_pathname
    scaled_file = file_name + ".scale"
    model_file = file_name + ".model"
    range_file = file_name + ".range"

    file_name = test_pathname
    assert os.path.exists(test_pathname),"testing file not found"
    scaled_test_file = file_name + ".scale"
    predict_test_file = file_name + ".predict"

    print('Check input data...')
    cmd = '{0} "{1}"'.format(os.path.abspath(check_py), train_pathname)
    print("run [%s]" % cmd)
    print(Popen(cmd, shell = True, stdout = PIPE).communicate()[0].decode())
    cmd = '{0} "{1}"'.format(check_py, test_pathname)
    print("run [%s]" % cmd)
    print(Popen(cmd, shell = True, stdout = PIPE).communicate()[0].decode())


    cmd = '{0} -s "{1}" "{2}" > "{3}"'.format(svmscale_exe, range_file, train_pathname, scaled_file)
    print('Scaling training data...')
    print("run [%s]" % cmd)
    Popen(cmd, shell = True, stdout = PIPE).communicate()

    weights = ["-w%s %s" % (k,v) for (k,v) in enumerate(balance)]
    cmd = '{0} -svmtrain "{1}" "{2}"'.format(grid_py, svmtrain_exe, scaled_file)
    print('Cross validation...')
    print("run [%s]" % cmd)
    f = Popen(cmd, shell = True, stdout = PIPE).stdout

    line = ''
    while True:
        last_line = line
        line = f.readline()
        if not line: break
    c,g,rate = map(float,last_line.split())

    print('Best c={0}, g={1} CV rate={2}'.format(c,g,rate))

    cmd = '{0} -c {1} -g {2} {3} "{4}" "{5}"'.format(svmtrain_exe,c,g," ".join(weights),scaled_file,model_file)
    print('Training...')
    print("run [%s]" % cmd)
    Popen(cmd, shell = True, stdout = PIPE).communicate()

    print('Output model: {0}'.format(model_file))

    cmd = '{0} -r "{1}" "{2}" > "{3}"'.format(svmscale_exe, range_file, test_pathname, scaled_test_file)
    print('Scaling testing data...')
    print("run [%s]" % cmd)
    Popen(cmd, shell = True, stdout = PIPE).communicate()

    cmd = '{0} "{1}" "{2}" "{3}"'.format(svmpredict_exe, scaled_test_file, model_file, predict_test_file)
    print('Testing...')
    print("run [%s]" % cmd)
    Popen(cmd, shell = True).communicate()

    print('Output prediction: {0}'.format(predict_test_file))

    return  predict_test_file


#######################################################################
if __name__ == "__main__":

    train_file = os.path.join(dt.output_dir(), 'svm_train')
    validate_file = os.path.join(dt.output_dir(), 'svm_validate')
    all_train_file = os.path.join(dt.output_dir(), 'svm_all_train')
    test_file = os.path.join(dt.output_dir(), 'svm_test')
    result_file = os.path.join(dt.output_dir(), 'svm_result')

    y_ = pd.read_csv(os.path.join(dt.data_dir(), 'task2', 'y_train.csv'), header=0, index_col=0)#.iloc[:100,:50]
    x_ = pd.read_csv(os.path.join(dt.data_dir(), 'task2', 'X_train.csv'), header=0, index_col=0)#.iloc[:100,:50]
    x_test = pd.read_csv(os.path.join(dt.data_dir(), 'task2', 'X_test.csv'), header=0, index_col=0)#.iloc[:100,:50]
    assert all(y_.index == x_.index)
    assert all(x_test.columns == x_.columns)

    ## train / validate data
    index_val = dt.create_validation_set(y_, imbalance=True, enforce_imbalance_ratio=False)

    y_train = y_.reindex(y_.index.difference(index_val))
    x_train = x_.reindex(x_.index.difference(index_val))
    assert all(y_train.index == x_train.index)
    sdt.write_libsvm_input(y_train, x_train, train_file)

    y_val = y_.reindex(index_val)
    x_val = x_.reindex(index_val)
    assert all(y_val.index == x_val.index)
    assert all(x_val.columns == x_train.columns)
    sdt.write_libsvm_input(y_val, x_val, validate_file)

    ## all train / test data
    sdt.write_libsvm_input(y_, x_, all_train_file)
    sdt.write_libsvm_input(pd.Series(index=x_test.index, data=0), x_test, test_file)

    validate_predict_file = run(train_file, validate_file, balance=dt.BALANCE)

    y_val_hat = pd.read_csv(validate_predict_file, header=None, sep=' ')[0]
    score = balanced_accuracy_score(y_true=y_val.values.flatten(), y_pred=y_val_hat.values.flatten())
    print("Balanced score: %s" % score)

    test_predict_file = run(all_train_file, test_file, balance=dt.BALANCE)
    test_predict = pd.read_csv(test_predict_file, header=None, sep=' ')[0]
    test_predict.name = 'y'
    test_predict.index.name = 'id'
    pd.DataFrame(test_predict).to_csv(result_file)






