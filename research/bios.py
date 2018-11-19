import os
import numpy
import pandas as pd
import data as dt
import biosppy


#######################################################################
def to_template(x, sample_frequency):

    try:
        templates = biosppy.signals.ecg.ecg(x, sampling_rate=sample_frequency, show=False)['templates']
        x_hat = pd.DataFrame(templates).mean()
        x_hat = pd.Series((x_hat - x_hat.mean()) / x_hat.std())
        return x_hat

    except Exception as e:
        print("error processing ecg: %s" % e)
        return pd.Series(index=range(180), data=0)


#######################################################################
def templates(X, sample_frequency=300):

    result = pd.DataFrame(index=range(180))
    for i in range(len(X)):
        print("processing serie: %d" % i)
        tmpl = to_template(X.iloc[i,:].dropna(), sample_frequency)
        tmpl.name=i
        result = result.join(tmpl)

    result = result.transpose()
    result.index.name='id'

    return result


#######################################################################
if __name__ == '__main__':

    import sys
    import logging

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    #nrows = 10
    nrows = 3030+443+1474+170

    y = pd.read_csv(os.path.join(dt.data_dir(), 'task3', 'y_train.csv'), header=0, index_col=0, nrows=nrows)
    _X = pd.read_csv(os.path.join(dt.data_dir(), 'task3', 'X_train.csv'), header=0, index_col=0, nrows=nrows)
    _X_test = pd.read_csv(os.path.join(dt.data_dir(), 'task3', 'X_test.csv'), header=0, index_col=0, nrows=nrows)


    if True:
        X_templates = templates(_X)
        X_test_templates = templates(_X_test)
        X_templates.to_csv(os.path.join(dt.data_dir(), 'task3', 'X_train_templates.csv'))
        X_test_templates.to_csv(os.path.join(dt.data_dir(), 'task3', 'X_test_templates.csv'))
        logger.info("Templates transform complete")



    print()