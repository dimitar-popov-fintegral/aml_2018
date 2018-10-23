import os
import pandas as pd
import data as dt


def write_libsvm_input(y, X, set_name):

    assert all(y.index == X.index)

    with open(os.path.join(dt.output_dir(), 'task2_libsvm_%s' % set_name), 'w') as f:
        for row_n in range(len(y.index)):
            row = "%s " % int(y.iloc[row_n])
            for col_n in range(len(X.columns)):
                row += "%s:%s " % (col_n+1, X.iloc[row_n, col_n])
            row += "\n"
            f.write(row)



if __name__ == "__main__":


    y_ = pd.read_csv(os.path.join(dt.data_dir(), 'task2', 'y_train.csv'), header=0, index_col=0)
    x_ = pd.read_csv(os.path.join(dt.data_dir(), 'task2', 'X_train.csv'), header=0, index_col=0)
    assert all(y_.index == x_.index)

    index_val = dt.create_validation_set(y_)

    y_train = y_.reindex(y_.index.difference(index_val))
    x_train = x_.reindex(x_.index.difference(index_val))
    assert all(y_train.index == x_train.index)
    write_libsvm_input(y_train, x_train, 'train')

    y_val = y_.reindex(index_val)
    x_val = x_.reindex(index_val)
    assert all(y_val.index == x_val.index)
    assert all(x_val.columns == x_train.columns)
    write_libsvm_input(y_val, x_val, 'validate')


    x_test = pd.read_csv(os.path.join(dt.data_dir(), 'task2', 'X_test.csv'), header=0, index_col=0)
    assert all(x_test.columns == x_train.columns)
    write_libsvm_input(pd.Series(index=x_test.index, data=0), x_test, 'test')


    predict_ = pd.read_csv(os.path.join(dt.output_dir(), 'task2_test.predict'), header=None)
    predict_.columns = ['y']
    predict_.index.name = 'id'
    predict_.to_csv(os.path.join(dt.output_dir(), 'task2_svm_predict'))


    print("DONE")




