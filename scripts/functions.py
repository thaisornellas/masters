import numpy as np
import math

def mean(x, type = 'arithmetic'): # calculate the mean of a set of data x 
    if type == 'arithmetic':
        mean = sum(x) / len(x)
    
    elif type == 'bins':
        data = []
        weight = []
        for i in x:
            if i not in data:
                data.append(i)
                weight.append(x.count(i))
        w_s = sum([w * d for w, d in zip(weight, data)])
        mean = w_s / sum(weight)

    elif type == 'geometric':
        mean = (np.prod(x)) ** (1 / len(x))

    elif type == 'harmonic':
        sum_x1 = 0.0
        k = 0
        while k < len(x):
            sum_x1 = sum_x1 + x[k]**(-1)
            k = k + 1
        mean = len(x) / sum_x1
    
    elif type == 'rms':
        sum_x2 = 0.0
        j = 0
        while j < len(x):
            sum_x2 = sum_x2 + x[j]**2
            j = j + 1
        mean = (sum_x2 / len(x))**(1/2)

    return mean

def cov_matrix(x, y): #calculate the covariance matrix between data x and y

    xx = [x_i * x_i for x_i, x_i in zip(x, x)]
    xy = [x_i * y_j for x_i, y_j in zip(x, y)]
    yy = [y_i * y_j for y_i, y_j in zip(y, y)]

    cov_xx = mean(xx) - mean(x) * mean(x)
    cov_xy = mean(xy) - mean(x) * mean(y)
    cov_yx = mean(xy) - mean(y) * mean(x)
    cov_yy = mean(yy) - mean(y) * mean(y)

    matrix = np.array([[cov_xx, cov_xy], [cov_yx, cov_yy]])
    return matrix

def standard_deviation(x, y = 'null'): # estimator of the std when the true mean is known
    sum_std = 0
    if y == 'null':
        for i in range(0, len(x)):
            dif = (x[i] - mean(x))**2
            sum_std = sum_std + dif
        sd = math.sqrt((1 / len(x)) * sum_std)

    else:
        for i in range(0, len(x)):
            dif = (x[i] - y)**2
            sum_std = sum_std + dif
        sd = math.sqrt((1 / len(x)) * sum_std)

    return sd

def s(x): # estimator of the std given no prior knowledge of the true mean
    sum_s = 0
    for i in range(0, len(x)):
        dif = (x[i] - mean(x))**2
        sum_s = sum_s + dif
    return 1 / math.sqrt(len(x) - 1) * np.sum(math.sqrt(sum_s))

def std_of_mean(resolution, x): #std of the mean when the true mean is known
    return resolution / math.sqrt(len(x))

def std_of_std(std, x): #std of std when the true mean is known
    return std / math.sqrt(2*len(x))

def std_of_std2(std, x): #std of std with no knowledge of the true mean
    return std / math.sqrt(2*(len(x) - 1))

def clt(data_array, n_samples, n_times): # pick n_samples from the data_array n_times
    sample_mean = []
    for i in range(n_times):
        sample = np.random.choice(data_array, size=n_samples, replace=False)
        mean_sample = np.sum(sample)
        sample_mean.append(mean_sample)
    return sample_mean

def ls_simple(x, y):
    return mean([x_i * y_i for x_i, y_i in zip(x,y)]) / mean([x_i * x_i for x_i in x])

def ls_simple_var(x, sigma):
    return sigma**2 / (len(x) * mean([x_i * x_i for x_i in x]))

def chi2(y, fx, sigma):
    return np.sum([((y_i - fx_i) / sigma)**2 for y_i, fx_i in zip(y, fx)])
