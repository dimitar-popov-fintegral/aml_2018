import os
from subprocess import *
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
import research.svm_data as sdt
import data as dt



PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
SVM_LIB_PATH = os.path.abspath(os.path.join(PROJECT_ROOT,'..','libsvm-3.23'))

def _format_range_arg(input_range):
    return ",".join(["%s" % el for el in input_range])

def train_rbf(train_pathname, test_pathname, balance, c_range, g_range):

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

    cmd = '%s -log2c %s -log2g %s -svmtrain "%s" "%s"' % (grid_py,
                                                          _format_range_arg(c_range),
                                                          _format_range_arg(g_range),
                                                          svmtrain_exe,
                                                          scaled_file)
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

    weights = ["-w%s %s" % (k,v) for (k,v) in enumerate(balance)]
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


def run_test(train_pathname, test_pathname, balance):

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

    weights =" ".join(["-w%s %s" % (k,v) for (k,v) in enumerate(balance)])

    c = 1.8
    g = -9.8
    cmd = '%s -c %s -g %s %s "%s" "%s"' % (svmtrain_exe,
                                           2**c,
                                           2**g,
                                           weights,
                                           scaled_file,model_file)
    """

    cmd = '%s -t 0 %s "%s" "%s"' % (svmtrain_exe,
                                    weights,
                                    scaled_file,
                                    model_file)
    """


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


    X, y, X_test = sdt.read_data(clean=False, num_pca=None, num_el=None)


    index_val = dt.create_validation_set(y, imbalance=True, enforce_imbalance_ratio=False)

    y_train = y.reindex(y.index.difference(index_val))
    X_train = X.reindex(X.index.difference(index_val))
    assert all(y_train.index == X_train.index)

    y_val = y.reindex(index_val)
    X_val = X.reindex(index_val)
    assert all(y_val.index == X_val.index)
    assert all(X_val.columns == X_train.columns)

    sdt.write_libsvm_input(y_train, X_train, train_file)
    sdt.write_libsvm_input(y_val, X_val, validate_file)
    sdt.write_libsvm_input(y, X, all_train_file)
    sdt.write_libsvm_input(pd.Series(index=X_test.index, data=0), X_test, test_file)


    balance=[4.1, 0.81, 3.97]


    validate_predict_file = run_test(train_file, validate_file, balance=balance)
    y_val_hat = pd.read_csv(validate_predict_file, header=None, sep=' ')[0]
    score = balanced_accuracy_score(y_true=y_val.values.flatten(), y_pred=y_val_hat.values.flatten())
    print("Balanced score (test kernel): %s" % score)

    ##IF metric not satisfactory terminate

    test_predict_file = run_test(all_train_file, test_file, balance=balance)

    test_predict = pd.read_csv(test_predict_file, header=None, sep=' ')[0]
    test_predict.name = 'y'
    test_predict.index.name = 'id'
    pd.DataFrame(test_predict).to_csv(result_file)



































    """

    c_range=(1,3,0.2)
    g_range=(-9,-13,-0.2)

    validate_predict_file = train_rbf(train_file, validate_file,
                                      c_range=c_range,
                                      g_range=g_range,
                                      balance=balance)

    y_val_hat = pd.read_csv(validate_predict_file, header=None, sep=' ')[0]
    score = balanced_accuracy_score(y_true=y_val.values.flatten(), y_pred=y_val_hat.values.flatten())
    print("Balanced score: %s" % score)


    test_predict_file = train_rbf(all_train_file, test_file,
                                  c_range=c_range,
                                  g_range=g_range,
                                  balance=balance)

    test_predict = pd.read_csv(test_predict_file, header=None, sep=' ')[0]
    test_predict.name = 'y'
    test_predict.index.name = 'id'
    pd.DataFrame(test_predict).to_csv(result_file)
    """

