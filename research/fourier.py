import os
import sys

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(THIS_DIR, '..'))

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

import logging
import numpy
import pandas
import re
import matplotlib.pyplot as plt 

from scipy.signal import butter, lfilter


#######################################################################
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


#######################################################################
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


#######################################################################
def prepare_frequency_data_set(X, sample_frequency, normalize=True, scale=True, low_cut=0.5, high_cut=15, pow2=None):

    ##
    logger = logging.getLogger(__name__)
    logger.debug('<-- Time -> Freq. [Enter] -->')

    ##
    X_padded, nearest_power = pad_time_trace_with_zeros(X=X.copy(), pow2=None)

    """
    Band-pass filter 
    """
    logger.debug('running band-pass filter')
    def bounded_functor(x):
        ts = butter_bandpass_filter(x, low_cut, high_cut, sample_frequency, order=5)
        return ts

    X_filtered= numpy.apply_along_axis(
        func1d=bounded_functor,
        axis=1,
        arr=X_padded.copy(),
    )

    del bounded_functor

    """
    Fourier transform 
    """
    logger.debug('running fourier transform')
    def bounded_functor(x):
        frequency, magnitude = fourier_transform(x, sample_frequency, normalize, scale)
        return magnitude
    
    X_frequency_transformed = numpy.apply_along_axis(
        func1d=bounded_functor, 
        axis=1, 
        arr=X_filtered, 
    )

    del bounded_functor

    """
    Restrict frequencies to window
    """
    N, K =  X_frequency_transformed.shape
    tick_size_hz = sample_frequency / K  
    high_freq_idx = numpy.int(numpy.floor(high_cut / tick_size_hz))
    low_freq_idx = numpy.int(numpy.floor(low_cut / tick_size_hz))
    assert high_freq_idx > low_freq_idx
    deg = numpy.ceil(numpy.log2(high_freq_idx - low_freq_idx))
    adj_high_freq_idx = low_freq_idx + numpy.int(2**(deg-1))
    X_frequency_trimmed = X_frequency_transformed[:, low_freq_idx:adj_high_freq_idx]

    return X_frequency_trimmed, nearest_power


#######################################################################
def fourier_transform(x, sample_frequency, normalize=True, scale=True):

    N = len(x)
    T = 1 / sample_frequency
    x_fft = numpy.fft.fft(x)
    x_frequency = numpy.linspace(0, 1 / T, N)

    ##
    frequency = x_frequency[:N//2]
    magnitude = numpy.abs(x_fft)[:N//2]

    if normalize:
        magnitude = magnitude * (1/N)

    if scale:
        magnitude = magnitude / numpy.max(magnitude)

    return frequency, magnitude


#######################################################################
def pad_time_trace_with_zeros(X, pow2=None):
    """
    Args:
        :param X: (pandas.DataFrame) time-trace information, columns:= time, index:= ID
        :param pow2: (int) override the nearest power of 2 parameter 
    """

    n,m = X.shape
    missing = X.isnull().sum(axis=1)
    length = numpy.repeat(m, n)
    nearest_pow2 = int(numpy.max(numpy.ceil(numpy.log2(length - missing))))

    if pow2:
        assert isinstance(pow2, int)
        assert pow2 >= nearest_pow2
        nearest_pow2 = pow2

    ##
    def surrogate_function(row, M = 2**nearest_pow2):
        temp = row.copy()
        values = temp[~numpy.isnan(temp)]
        zeros = numpy.zeros(M - len(values))
        return numpy.hstack([values, zeros])

    result = numpy.apply_along_axis(
        func1d=surrogate_function,
        arr=X.values,
        axis=1
    )

    return result, nearest_pow2


#######################################################################
def data_sample():

    X = numpy.apply_along_axis(
        func1d=lambda y: numpy.sin(numpy.arange(0,2*numpy.pi, 0.01)), 
        arr=numpy.array([[1]]*10), 
        axis=1
    )   

    X_freq = prepare_frequency_data_set(pandas.DataFrame(X), 100)

    return X_freq


#######################################################################
def sample(low_p=2, high_p=200, low_s=0.01, high_s=1, random_noise=False):

    ##
    rs = numpy.random.RandomState(12357)

    ##
    low_x_values = numpy.array([i for i in numpy.arange(0, low_p*numpy.pi, low_s)])
    high_x_values = numpy.array([i for i in numpy.arange(0, high_p*numpy.pi, high_s)])

    if random_noise:
        high_x_values = high_x_values * rs.randn(len(high_x_values))

    assert len(low_x_values) == len(high_x_values),\
        'values for (low, high) period and step do not produce conformable x-axis'

    low_freq_component = numpy.sin(low_x_values)
    random_high_freq_component = numpy.cos(high_x_values)
    time_trace =  low_freq_component + random_high_freq_component 
    sample_frequency = 200

    return plot_fourier_transform(time_trace, sample_frequency)


#######################################################################
def plot_fourier_transform(x, sample_frequency, normalize=True, scale=True):
    """
    Args:
        :param x: (numpy.array)
        :param sample_frequency: (float) sample frequency in Hertz [Hz]
    """

    low_thresh = 0.67
    high_thresh = 25
    N = len(x)
    time_trace = numpy.linspace(0, N / sample_frequency, N)
    frequency, magnitude = fourier_transform(x, sample_frequency, normalize=normalize, scale=scale)    

    """
    Unfiltered time-series
    """

    ##
    fig, ax = plt.subplots(4)

    ##
    axis=0
    ax[axis].plot(time_trace, x)
    ax[axis].set_xlabel('Time [s]')
    ax[axis].set_ylabel('potential')

    ##
    axis+=1
    ax[axis].bar(frequency, magnitude, width=1)
    ax[axis].set_xlabel('Frequency [Hz]')
    ax[axis].set_ylabel('Magnitude [|x|]')

    """
    Reepeat for filtered time-series
    """
    y = butter_bandpass_filter(x, low_thresh, high_thresh, sample_frequency, order=5)
    frequency, magnitude = fourier_transform(y, sample_frequency, normalize=normalize, scale=scale)
    ##
    axis+=1
    ax[axis].plot(time_trace, y)
    ax[axis].set_xlabel('Time [s]')
    ax[axis].set_ylabel('potential [|x|]')

    ##
    axis+=1
    ax[axis].bar(frequency, magnitude, width=1)
    ax[axis].set_xlabel('Frequency [Hz]')
    ax[axis].set_ylabel('Magnitude [|x|]')


    return fig


#######################################################################
if __name__ == '__main__':

    root = logging.getLogger(__name__)
    root.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    root.addHandler(ch) 
    logger = logging.getLogger(__name__)

    ##
    logging.info('<-- Fourier lib [sample] -->')
    data_sample()


