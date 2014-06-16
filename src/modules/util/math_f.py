import numpy as np
import matplotlib.pyplot as plt


### Gaussian PDF
def p_gaussian(x, mu, sigma):
    return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2))


def pdf_gaussian(xs, mu, sigma):
    return [p_gaussian(x, mu, sigma) for x in xs]


### Gaussian Kernel
def kernel_gaussian(x, h):
    return np.exp(-x**2 / (2*h**2))


def sum_gaussian(data, linspa, h):
    ret = np.array([])
    for x in linspa:
        ret = np.append(ret, sum([kernel_gaussian(x - xi, h) for xi in data]) / (len(data) * h))
    return ret


def plot_gaussian(data, sigma, h):
    ### Kernel Density Estimator
    positives = data[data[:, 1] == 1]
    negatives = data[data[:, 1] == -1]
    # plot individual kernels
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
    ax[0].hist(positives[:, 0], normed=True, alpha=0.5, label='Histogram of positives')
    ax[1].hist(negatives[:, 0], normed=True, alpha=0.5, label='Histogram of negatives')
    first = True
    for xx in positives:
        x = xx[0]
        kernel_xs = np.linspace(x - 5*sigma, x + 5*sigma)
        kernel_ys = np.array(pdf_gaussian(kernel_xs, x, sigma)) * 0.4  # scaling just so it looks better
        if first:
            ax[0].plot(kernel_xs, kernel_ys, 'b--', linewidth=1, label=' positive Kernels')
            first = False
        else:
            ax[0].plot(kernel_xs, kernel_ys, 'b--', linewidth=1)
    first = True
    for xx in negatives:
        x = xx[0]
        kernel_xs = np.linspace(x - 5*sigma, x + 5*sigma)
        kernel_ys = np.array(pdf_gaussian(kernel_xs, x, sigma)) * 0.4 # scaling just so it looks better
        if first:
            ax[1].plot(kernel_xs, kernel_ys, 'r--', linewidth=1, label=' negative Kernels')
            first = False
        else:
            ax[1].plot(kernel_xs, kernel_ys, 'r--', linewidth=1)
    # plot kernel density estimator (sum of individual kernels)
    min_neg = min(negatives[:, 0]) - 20
    max_neg = max(negatives[:, 0]) + 20
    min_pos = min(positives[:, 0]) - 20
    max_pos = max(positives[:, 0]) + 20
    xs_n = np.linspace(min_neg, max_neg, 100)
    xs_p = np.linspace(min_pos, max_pos, 100)
    kde_n = sum_gaussian(negatives[:, 0], xs_n, h)
    kde_p = sum_gaussian(positives[:, 0], xs_p, h)
    # plot Kernel density estimation
    ax[0].plot(xs_p, kde_p, linewidth=3, color='blue', label='positive KDE')
    ax[1].plot(xs_n, kde_n, linewidth=3, color='red', label='negative KDE')
    ax[0].legend()
    ax[1].legend()
    plt.show()