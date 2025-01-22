import math
import numpy as np
import matplotlib.pyplot as plt

def binomial_dist(r, p, n):
    return p**r*(1 - p)**(n-r)*math.factorial(n) / (math.factorial(r) * math.factorial(n - r))

def poisson_dist(r, lam):
    return math.exp(-lam) * lam**r / math.factorial(r)

def normal_dist(x, mu, sig):
    return 1 / (sig * math.sqrt(2 * np.pi)) * math.exp(- (x - mu)**2 / (2 * sig**2))

####### PLOT

def plot_binomial(r_vals, p, n):
    bin_dist = [binomial_dist(r, p, n) for r in r_vals]
    plt.bar(r_vals, bin_dist, color='red')
    plt.title(f"Binomial Distribution (p = {p}, n = {n})")
    plt.xlabel("Number of success (r)")
    plt.ylabel("P(r; p, n)")

def plot_poisson(r_vals, lam):
    poi_dist = [poisson_dist(r, lam) for r in r_vals]
    plt.bar(r_vals, poi_dist, color='lightblue')
    plt.title(f"Poisson Distribution (λ = {lam})")
    plt.xlabel("Number of events (k)")
    plt.ylabel("P(k, λ)")

def plot_normal(x_vals, mu, sig):
    gauss_dist = [normal_dist(x, mu, sig) for x in x_vals]
    plt.plot(x_vals, gauss_dist, color='blue')
    plt.title(f"Gaussian Distribution (μ = {mu}, σ = √{mu})")
    plt.xlabel("x")
    plt.ylabel("P(x; μ, σ)")