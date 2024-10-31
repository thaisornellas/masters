import numpy as np
import math

def mean(x, type = 'arithmetic'):
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

def cov_matrix(x, y):

    xx = [x_i * x_i for x_i, x_i in zip(x, x)]
    xy = [x_i * y_j for x_i, y_j in zip(x, y)]
    yy = [y_i * y_j for y_i, y_j in zip(y, y)]

    cov_xx = mean(xx) - mean(x) * mean(x)
    cov_xy = mean(xy) - mean(x) * mean(y)
    cov_yx = mean(xy) - mean(y) * mean(x)
    cov_yy = mean(yy) - mean(y) * mean(y)

    matrix = np.array([[cov_xx, cov_xy], [cov_yx, cov_yy]])
    return matrix

def standard_deviation(x):
    sum = 0
    for i in range(0, len(x)):
        dif = (x[i] - mean(x))**2
        sum = sum + dif
        
    sd = math.sqrt((1 / len(x)) * sum)
    return sd
