import os
import pandas as pd
import data as dt


#######################################################################
def write_libsvm_input(y, X, file_name):

    assert all(y.index == X.index)

    with open(os.path.join(file_name), 'w') as f:
        for row_n in range(len(y.index)):
            row = "%s " % int(y.iloc[row_n])
            for col_n in range(len(X.columns)):
                row += "%s:%s " % (col_n+1, X.iloc[row_n, col_n])
            row += "\n"
            f.write(row)


