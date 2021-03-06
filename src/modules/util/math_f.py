import numpy as np
import matplotlib.pyplot as plt


def plot_histograms(pos_labeled_data, neg_labeled_data):
    plt.hist(pos_labeled_data[:, 0], bins=50, alpha=.5, color='blue')
    plt.hist(neg_labeled_data[:, 0], bins=50, alpha=.5, color='red')
    plt.show()


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


def get_two_kdes(positives, negatives, h):
    margin = max(max(negatives)-min(negatives), max(positives)-min(positives))
    min_neg = min(negatives) - margin
    max_neg = max(negatives) + margin
    min_pos = min(positives) - margin
    max_pos = max(positives) + margin
    xs_n = np.linspace(min_neg, max_neg, 1000)
    xs_p = np.linspace(min_pos, max_pos, 1000)
    kde_n = sum_of_gaussians(negatives, xs_n, h)
    kde_p = sum_of_gaussians(positives, xs_p, h)
    return kde_n, kde_p, xs_n, xs_p


def plot_gaussians(neg_ratios, pos_ratios, sigma, h):
    """
    :param neg_ratios: As acquired from classifier, annotated
    :param pos_ratios: As acquired from classifier, annotated
    :param sigma: Standard deviation
    :param h: Kernel width
    :return: void
    """
    ### Kernel Density Estimator
    positives = pos_ratios
    negatives = neg_ratios
    # plot individual kernels
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
    ax[0].hist(positives, normed=True, alpha=.5, label='Histogram of positives', color='blue')
    ax[1].hist(negatives, normed=True, alpha=.5, label='Histogram of negatives', color='red')
    first = True
    for xx in positives:
        x = xx
        kernel_xs = np.linspace(x - 2*sigma, x + 2*sigma)
        kernel_ys = np.array(pdf_gaussian(kernel_xs, x, sigma))
        if first:
            ax[0].plot(kernel_xs, kernel_ys, 'b--', linewidth=1, label=' positive Kernels')
            first = False
        else:
            ax[0].plot(kernel_xs, kernel_ys, 'b--', linewidth=1)
    first = True
    for xx in negatives:
        x = xx
        kernel_xs = np.linspace(x - 2*sigma, x + 2*sigma)
        kernel_ys = np.array(pdf_gaussian(kernel_xs, x, sigma))
        if first:
            ax[1].plot(kernel_xs, kernel_ys, 'r--', linewidth=1, label=' negative Kernels')
            first = False
        else:
            ax[1].plot(kernel_xs, kernel_ys, 'r--', linewidth=1)

    kde_n, kde_p, xs_n, xs_p = get_two_kdes(positives, negatives, h)

    # plot Kernel density estimation
    ax[0].plot(xs_p, kde_p, linewidth=3, color='blue', label='positive KDE')
    ax[1].plot(xs_n, kde_n, linewidth=3, color='red', label='negative KDE')
    ax[0].legend()
    ax[1].legend()
    plt.show()


def plot_ratios(r, theta_a, theta_b):
    candi = []
    ratioi = []
    for cand, ratio in r:
        candi = np.append(candi, cand)
        ratioi = np.append(ratioi, ratio)

    if -5 < theta_a < 5:  # theta_a sanity check
        plt.axvline(x=theta_a, linewidth=1, ls='dashed', color='black', label='theta a')
    if -5 < theta_b < 5:  # theta_a sanity check
        plt.axvline(x=theta_b, linewidth=1, ls='dashed', color='black', label='theta b')

    plt.plot(candi, ratioi, linewidth=3, alpha=0.5, color='blue', label='Likelihood ratio')
    plt.legend()
    plt.show()


def plot_wc(wc):
    pos = wc.annotated_responses[wc.annotated_responses[:, 1] == +1]
    neg = wc.annotated_responses[wc.annotated_responses[:, 1] == -1]
    # smaller_pos = pos[pos[:, 0] < wc.threshold]
    # smaller_pos_w = np.sum(smaller_pos[:, 2])
    bigger_pos = pos[pos[:, 0] > wc.threshold]
    bigger_pos_w = np.sum(bigger_pos[:, 2])
    smaller_neg = neg[neg[:, 0] < wc.threshold]
    smaller_neg_w = np.sum(smaller_neg[:, 2])
    bigger_neg = neg[neg[:, 0] > wc.threshold]
    bigger_neg_w = np.sum(bigger_neg[:, 2])
    weights_pos = pos[:, 2]
    weights_neg = neg[:, 2]
    y_p = [0] * len(pos)
    y_n = [0] * len(neg)
    p = plt.scatter(pos[:, 0], y_p, weights_pos * len(y_p + y_n) * 40, 'b', label='')
    n = plt.scatter(neg[:, 0], y_n, weights_neg * len(y_p + y_n) * 40, 'r')
    plt.errorbar(wc.threshold, 0, yerr=0.5, linestyle="dashed", marker="None", color="green")
    legend_str1 = "%.2f < threshold < %.2f" % (smaller_neg_w, bigger_pos_w)
    legend_str2 = "%.2f < threshold < %.2f" % (smaller_neg_w, bigger_neg_w)
    plt.legend([p, n], [legend_str1, legend_str2])
    plt.show()