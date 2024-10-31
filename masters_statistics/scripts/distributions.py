import math
import numpy as np

def binomial_dist(r, p, n):
    return p**r*(1 - p)**(n-r)*math.factorial(n) / (math.factorial(r) * math.factorial(n - r))

def poisson_dist(r, lam):
    return math.exp(-lam) * lam**r / math.factorial(r)

def gaussian_dist(x, mu, sig):
    return 1 / (sig * math.sqrt(2 * np.pi)) * math.exp(- (x - mu)**2 / (2 * sig**2))