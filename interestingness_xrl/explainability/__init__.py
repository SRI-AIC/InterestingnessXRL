__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

import numpy as np
from itertools import groupby
from operator import itemgetter


def get_variation_ratio(dist):
    """
    Gets the statistical dispersion of a given nominal distribution in [0,1]. The larger the ratio (-> 1), the more
    differentiated or dispersed the data are. The smaller the ratio (0 <-), the more concentrated the distribution is.
    See: https://en.wikipedia.org/wiki/Variation_ratio
    :param array-like dist: the nominal distribution.
    :return float: a measure of the dispersion of the data in [0,1].
    """
    total = np.sum(dist)
    mode = float(np.max(dist))
    return 1. - mode / total


def get_distribution_evenness(dist):
    """
    Gets the evenness of a given nominal distribution as the normalized true diversity in [0,1]. It has into account the
    number of different categories one would expect to find in the distribution, i.e. it handles 0 entries.
    The larger the ratio (-> 1), the more differentiated or even the data are.
    The smaller the ratio (0 <-), the more concentrated the distribution is, i.e., the more uneven the data are.
    See: https://en.wikipedia.org/wiki/Species_evenness
    See: https://en.wikipedia.org/wiki/Diversity_index#Shannon_index
    :param array-like dist: the nominal distribution.
    :return float: a measure of the evenness of the data in [0,1].
    """
    num_expected_elems = len(dist)
    nonzero = np.nonzero(dist)[0]
    if len(nonzero) == 0:
        return 1. / np.log(num_expected_elems)
    if len(nonzero) == 1:
        return 0.
    dist = np.array(dist)[nonzero]
    total = float(np.sum(dist))
    dist = np.true_divide(dist, total)
    in_dist = [p * np.log(p) for p in dist]
    return - np.sum(in_dist) / np.log(num_expected_elems)


def get_outliers_double_mads(data, thresh=3.5):
    """
    Identifies outliers in a given data set according to the data-points' "median absolute deviation" (MAD), i.e.,
    measures the distance of all points from the median in terms of median distance.
    From answer at: https://stackoverflow.com/a/29222992
    :param np.array data: the data from which to extract the outliers.
    :param float thresh: the z-score threshold above which a data-point is considered an outlier.
    :return np.ndarray: an array containing the indexes of the data that are considered outliers.
    """
    # warning: this function does not check for NAs nor does it address
    # issues when more than 50% of your data have identical values
    m = np.median(data)
    abs_dev = np.abs(data - m)
    left_mad = np.median(abs_dev[data <= m])
    right_mad = np.median(abs_dev[data >= m])
    y_mad = left_mad * np.ones(len(data))
    y_mad[data > m] = right_mad
    modified_z_score = 0.6745 * abs_dev / y_mad
    modified_z_score[data == m] = 0
    return np.where(modified_z_score > thresh)[0].tolist()


def get_outliers_dist_mean(data, std_devs=2., above=True, below=True):
    """
    Identifies outliers according to distance of a number of standard deviations to the mean.
    :param np.array data: the data from which to extract the outliers.
    :param float std_devs: the number of standard deviations above/below which a point is considered an outlier.
    :return: np.ndarray: an array containing the indexes of the data that are considered outliers.
    """
    mean = np.mean(data)
    std = np.std(data)
    outliers = [False] * len(data)
    if above:
        outliers |= data >= mean + std_devs * std
    if below:
        outliers |= data <= mean - std_devs * std
    return np.where(outliers)[0].tolist()


def get_jensen_shannon_divergence(dist1, dist2):
    """
    Computes the Jensen-Shannon divergence between two probability distributions. Higher values (close to 1) mean that
    the distributions are very dissimilar while low values (close to 0) denote a low divergence, similar distributions.
    See: https://stackoverflow.com/a/40545237
    See: https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    Ref: Lin J. 1991. "Divergence Measures Based on the Shannon Entropy".
        IEEE Transactions on Information Theory. (33) 1: 145-151.
    Input must be two probability distributions of equal length that sum to 1.
    :param ndarray dist1: the first probability distribution.
    :param ndarray dist2: the second probability distribution.
    :return float: the divergence between the two distributions in [0,1].
    """

    def _kl_div(a, b):
        return np.sum(a * np.log2(a / b))

    m = 0.5 * (dist1 + dist2)
    return 0.5 * (_kl_div(dist1, m) + _kl_div(dist2, m))


def get_pairwise_jensen_shannon_divergence(dist1, dist2):
    """
    Computes the pairwise Jensen-Shannon divergence between two probability distributions. This corresponds to the
    un-summed JSD, i.e., the divergence according to each component of the given distributions. Summing up the returned
    array yields the true JSD between the two distributions.
    See: https://stackoverflow.com/a/40545237
    See: https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence
    Input must be two probability distributions of equal length that sum to 1.
    :param ndarray dist1: the first probability distribution.
    :param ndarray dist2: the second probability distribution.
    :return ndarray: the divergence between each component of the two distributions in [0,1].
    """

    def _kl_div(a, b):
        return a * np.log2(a / b)

    m = 0.5 * (dist1 + dist2)
    return 0.5 * (_kl_div(dist1, m) + _kl_div(dist2, m))


def group_by_key(data):
    """
    Groups the given list of tuples according to the values in the first element of each tuple. The result is a list of
    (key, [(values1), (values2), ...]) where the data is indexed by the key.
    See: https://stackoverflow.com/a/5695349
    :param list data: the data to be grouped containing tuples indexed by key.
    :return list: a list where the values are grouped/indexed by the corresponding key.
    """
    data.sort()
    groups = groupby(data, itemgetter(0))
    return [(key, [tup[1:] for tup in tups]) for (key, tups) in groups]


def get_diff_means(mean1, std1, n1, mean2, std2, n2):
    """
    Gets the difference of the given sample means (mean1 - mean2).
    See: https://stattrek.com/sampling/difference-in-means.aspx
    :param float mean1: the first mean value.
    :param float std1: the first mean's standard deviation.
    :param int n1: the first mean's count.
    :param float mean2: the first mean value.
    :param float std2: the first mean's standard deviation.
    :param int n2: the first mean's count.
    :return float, float, int: the differences of the mean, standard deviation and number of elements.
    """
    return \
        mean1 - mean2, \
        np.sqrt((std1 * std1) / n1 + (std2 * std2) / n2).item(), \
        n1 - n2
