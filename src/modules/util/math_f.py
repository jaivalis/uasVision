import numpy as np
import matplotlib.pyplot as plt


def plot_histograms(pos_labeled_data, neg_labeled_data):
    height = pos_labeled_data[:, 0]
    # hist, bins = np.histogram(height, 50)
    # center = (min(pos_labeled_data[:, 1]) + max(pos_labeled_data[:, 1])) / 2
    plt.hist(height, bins=10)
    plt.show()
    # plt.bar(center, hist, align='center', width=99)
    # plt.show()


def append_gaussian(gmm, gaussian, lin_space):
    ret = None
    if gmm is None:
        ret = gaussian
    else:
        for x in lin_space:
            ret = np.append(ret, gmm[x] + gaussian[x])
        pass
    return ret


### Gaussian PDF
def p_gaussian(x, mu, sigma):
    return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu)**2 / (2 * sigma**2))


def pdf_gaussian(xs, mu, sigma):
    return [p_gaussian(x, mu, sigma) for x in xs]


### Gaussian Kernel
def kernel_gaussian(x, h):
    return np.exp(-x**2 / (2*h**2))


def sum_of_gaussians(data, lin_space, h):
    """
    :param data: Data points only, no annotations (splitting must be done before)
    :param lin_space: Usually np.linspace(min_neg, max_neg, 100)
    :param h: Kernel width
    :return: The function expressed as a sum of Gaussians for the given linear space
    """
    ret = np.array([])
    for x in lin_space:
        ret = np.append(ret, sum([kernel_gaussian(x - xi, h) for xi in data]) / (len(data) * h))
    return ret


def plot_gaussians(data, sigma, h):
    """
    :param data: As acquired from classifier, annotated
    :param sigma: Standard deviation
    :param h: Kernel width
    :return: void
    """
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
        kernel_xs = np.linspace(x - 2*sigma, x + 2*sigma)
        kernel_ys = np.array(pdf_gaussian(kernel_xs, x, sigma)) * 0.4  # scaling just so it looks better
        if first:
            ax[0].plot(kernel_xs, kernel_ys, 'b--', linewidth=1, label=' positive Kernels')
            first = False
        else:
            ax[0].plot(kernel_xs, kernel_ys, 'b--', linewidth=1)
    first = True
    for xx in negatives:
        x = xx[0]
        kernel_xs = np.linspace(x - 2*sigma, x + 2*sigma)
        kernel_ys = np.array(pdf_gaussian(kernel_xs, x, sigma)) * 0.4  # scaling just so it looks better
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
    kde_n = sum_of_gaussians(negatives[:, 0], xs_n, h)
    kde_p = sum_of_gaussians(positives[:, 0], xs_p, h)

    # plot Kernel density estimation
    ax[0].plot(xs_p, kde_p, linewidth=3, color='blue', label='positive KDE')
    ax[1].plot(xs_n, kde_n, linewidth=3, color='red', label='negative KDE')
    ax[0].legend()
    ax[1].legend()
    plt.show()