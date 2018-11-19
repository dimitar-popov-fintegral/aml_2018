import numpy
import pandas
import logging

from statsmodels.tsa.arima_model import ARIMA


#######################################################################
def _coeff_index(order):
    index = list()
    for i in range(order[0]):
        index.append("ar%d" % i)
    for i in range(order[2]):
        index.append("ma%d" % i)
    return index


#######################################################################
def arima_fit(x, order, begin=100, end=3000):

    logger = logging.getLogger(__name__)

    try:
        series = x[begin:end]
        series = (series - series.mean())/series.std()
        model = ARIMA(series.values, order=order)
        model_fit = model.fit(disp=0)
        coeff = numpy.append(model_fit.arparams, model_fit.maparams)

    except Exception as e:
        logger.exception("failed to fit arima model, reported error: %s" % e)
        return pandas.Series(index=_coeff_index(order))

    return pandas.Series(index=_coeff_index(order), data=coeff)


#######################################################################
def features(X, order=(3,0,1)):

    logger = logging.getLogger(__name__)

    arimas = pandas.DataFrame(index=_coeff_index(order))
    for i in range(len(X)):
        logger.info("processing serie: %d" % i)
        af = arima_fit(X.iloc[i,:].dropna(), order)
        af.name=i
        arimas = arimas.join(af)

    result = arimas.transpose()
    result.index.name='id'

    return result

