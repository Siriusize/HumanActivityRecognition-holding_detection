# presume windows as numpy vector
import numpy as np
from scipy import signal

def average(window):
    window = np.array(window)
    return np.mean(window)

def std(window):
    window = np.array(window)
    return np.std(window)

# mean absolute deviation
def mean_abs_dev(window):
    window = np.array(window)
    w_median = np.median(window)
    window = abs(window - w_median)
    return np.median(window)

def max(window):
    window = np.array(window)
    return np.max(window)

def min(window):
    window = np.array(window)
    return np.min(window)

# tranform time domain to freq domain
def fft(window):
    window = np.array(window)
    return np.fft.fft(window)

# frequency skewness of a sliding window
def fre_skewness(window):
    window = np.array(window)
    trans = fft(window)
    aver = average(trans)
    stad = std(trans)
    trans = pow((trans - aver)/stad, 3)
    return abs(np.mean(trans))

def fre_kurtosis(window):
    window = np.array(window)
    trans = fft(window)
    aver = average(trans)
    numerator = pow(trans - aver, 4)
    numerator = np.mean(numerator)
    denominator = pow(trans - aver, 2)
    denominator = pow(np.mean(denominator), 2)
    return abs(numerator / denominator)

def fre_max(window):
    window = np.array(window)
    trans = fft(window)
    return np.max(trans)

# average energy
def energy(window):
    window = np.array(window)
    window = pow(window, 2)
    return np.mean(window)

# signal magnitude area
def sma(win1, win2, win3):
    win1 = np.array(win1)
    win2 = np.array(win2)
    win3 = np.array(win3)

    win1 = np.sum(abs(win1))
    win2 = np.sum(abs(win2))
    win3 = np.sum(abs(win3))
    return (win1 + win2 + win3) / 3

def entropy(window):
    window = np.array(window)
    sum = np.sum(window)
    window = window / sum
    t = 0
    for item in window:
        t = t + item * np.log2(item)
    return t / 3

def interquantile(window):
    window = np.array(window)
    a = np.percentile(window, 75)
    b = np.percentile(window, 25)
    return b - a

# autoregression model using least square
def ar_coef(window, order = 3):
    window = np.array(window)
    k = len(window)
    t = order
    x = list()
    y = list()
    for i in range(k-t-1):
        linex = window[i: i+t]
        x.append(linex)
        liney = [window[i+t]]
        y.append(liney)
    x = np.array(x)
    y = np.array(y)
    theta = np.dot(np.transpose(x), x)
    theta = np.linalg.inv(theta)
    theta = np.dot(theta, np.transpose(x))
    theta = np.dot(theta, y)
    return theta.reshape(-1)

# pearson correlation
def correlation(win1, win2):
    win1 = np.array(win1)
    win2 = np.array(win2)
    return np.corrcoef(win1, win2).sum().sum()

# frequency signal weighted average
def fswa(window):
    window = np.array(window)
    trans = np.fft.fft(window)
    deno = np.sum(trans)
    for i in range(len(trans)):
        trans[i] = trans[i] * (i+1)
    return abs(np.sum(trans) / deno)

# spectral energy
def fre_energy(window, a, b):
    window = np.array(window)
    trans = np.fft.fft(window)
    result = 0
    for i in range(a-1, b-1, 1):
        result = result + pow(trans[i], 2)
    return result / (a+b+1)


# input the 1-D sequence
def bw_filter(x, freq=10):
    x = np.array(x)
    sos = signal.butter(4, freq, 'lp', fs=50, output='sos')
    filtered = signal.sosfilt(sos, x)
    #filtered = list(filtered)
    return filtered


# list, list, integer
def nms(index, score, size):
    score = list(score)
    index = list(index)
    result = []
    while len(score) != 0:
        tar = np.argmax(np.array(score))
        result.append(index[tar])
        del score[tar]
        del index[tar]
        dust = []
        for i, item in enumerate(index):
            if abs(tar-item) <= size:
                dust.append(i)
        for i in dust:
            del score[i]
            del index[i]
    return result

