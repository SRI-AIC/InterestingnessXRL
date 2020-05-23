__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

import csv
import numpy as np

CSV_DELIMITER = ','


def discretize_features(feats, feats_max, feats_min, feats_nbins):
    """
    Discretizes a given set of continuous features into a unique number identifier.
    :param array_like feats: a list containing the continuous features to be discretized.
    :param array_like feats_max: a list with the maximum values that each feature can have.
    :param array_like feats_min: a list with the minimum values that each feature can have.
    :param array_like feats_nbins: a list with the number of (discrete) bins that each feature can have.
    :return int: an index denoting the unique combination of the given features after discretization.
    """
    # discretizes each feature
    obs_vec = np.zeros(len(feats), np.uint32)
    for i in range(len(feats)):
        obs_vec[i] = min(int(np.floor((feats[i] - feats_min[i]) / (feats_max[i] - feats_min[i]) * feats_nbins[i])),
                         feats_nbins[i] - 1)

    # gets discretized index
    return get_discretized_index(obs_vec, feats_nbins)


def get_discretized_index(obs_vec, feats_nbins):
    """
    Gets a number/index denoting the unique combination of the given discretized features.
    :param array_like obs_vec: a list containing the discretized features.
    :param array_like feats_nbins: a list with the number of bins for each feature, i.e., the maximal features values
    that each feature can have.
    :return int: an index denoting the unique combination of the given features.
    """
    idx = 0
    for i in range(len(obs_vec)):
        if obs_vec[i] == 0:
            continue
        stride = 1
        for j in range(i + 1, len(obs_vec)):
            stride *= feats_nbins[j]
        idx += obs_vec[i] * stride
    return idx


def get_features_from_index(idx, feats_nbins):
    """
    Gets a number/index denoting the unique combination of the given discretized features.
    :param int idx: an index denoting the unique combination of the features.
    :param array_like feats_nbins: a list with the number of bins for each feature, i.e., the maximal features values
    that each feature can have.
    :return array_like: a list containing the discretized features.
    """
    obs_vec = np.zeros(len(feats_nbins), np.uint32)
    for i in range(len(feats_nbins)):
        stride = 1
        for j in range(i + 1, len(feats_nbins)):
            stride *= feats_nbins[j]
        feat = idx // stride
        obs_vec[i] = feat
        idx -= feat * stride
    return obs_vec


def write_3d_table_csv(table, csv_file_path, delimiter=CSV_DELIMITER, col_names=None):
    """
    Writes the given N-dimensional array (N>2) into a CSV file. The output format is index0, index1, index2, ..., value.
    for non-zero entries in the array.
    :param array_like table: the data to be written to the CSV file.
    :param str csv_file_path: the path to the CSV file in which to write the data.
    :param str delimiter: the delimiter for the fields in the CSV file.
    :param array_like col_names: a list containing the names of each column/index of the data.
    :return:
    """
    # selects indexes for non-zero values and only writes those to file
    nz_idxs = np.transpose(np.nonzero(table))
    with open(csv_file_path, 'w') as csv_file:
        writer = csv.writer(csv_file, delimiter=delimiter)
        if col_names is not None:
            writer.writerow(col_names)
        for idxs in nz_idxs:
            row = idxs.tolist()
            row.append(table[idxs[0]][idxs[1]][idxs[2]])
            writer.writerow(row)


def read_3d_table_csv(table, csv_file_path, delimiter=CSV_DELIMITER, dtype=float, has_header=False):
    """
    Populates the given N-dimensional array (N>2) array with data loaded from a CSV file, where the data is in the
    format is index0, index1, index2, ..., value.
    :param array_like table: the table on which to load the data.
    :param str csv_file_path: the path to the CSV file from which to load the data.
    :param str delimiter: the delimiter for the fields in the CSV file.
    :param dtype: the type of the elements stored in the data file.
    :param bool has_header: whether the first line of the CSV file contains the column names.
    :return:
    """
    with open(csv_file_path) as csv_file:
        reader = csv.reader(csv_file, delimiter=delimiter)
        if has_header:
            next(reader)
        for row in reader:
            table[int(row[0])][int(row[1])][int(row[2])] = dtype(row[3])


def write_table_csv(table, csv_file_path, delimiter=CSV_DELIMITER, fmt='%s', col_names=None):
    """
    Writes the given array into a CSV file.
    :param array_like table: the data to be written to the CSV file.
    :param str csv_file_path: the path to the CSV file in which to write the data.
    :param str delimiter: the delimiter for the fields in the CSV file.
    :param str fmt: the string used to format the output of the elements in the data.
    :param array_like col_names: a list containing the names of each column of the data.
    :return:
    """
    header = '' if col_names is None else delimiter.join(col_names)
    np.savetxt(csv_file_path, table, delimiter=delimiter, fmt=fmt, header=header, comments='')


def read_table_csv(csv_file_path, delimiter=CSV_DELIMITER, dtype=float, has_header=False):
    """
    Loads an array from a CSV file.
    :param str csv_file_path: the path to the CSV file from which to load the data.
    :param str delimiter: the delimiter for the fields in the CSV file.
    :param object dtype: the type of the elements stored in the data file.
    :param bool has_header: whether the first line of the CSV file contains the column names.
    :return np.ndarray: the numpy array loaded from the CSV file.
    """
    return np.loadtxt(csv_file_path, delimiter=delimiter, dtype=dtype, skiprows=1 if has_header else 0)


def convert_table_to_array(table, def_entry=None):
    max_len = max([len(x) for x in table])
    return np.array([x + [def_entry] * (max_len - len(x)) for x in table])


def convert_array_to_table(array, def_entry=None):
    table = []
    for i in range(len(array)):
        idx = index(array[i], def_entry)[0]
        table.append(array[i][:idx].tolist())
    return table


def index(array, item):
    for idx, val in np.ndenumerate(array):
        if val == item:
            return idx
    return array.shape


def get_combined_mean(means, stds, counts):
    """
    Calculates the combined mean and standard deviation of the means and stds of multiple groups.
    Given statistics should be index-aligned for each group.
    See: https://www.statstodo.com/CombineMeansSDs_Pgm.php
    :param np.ndarray means: the mean values of the groups.
    :param np.ndarray stds: the standard deviation values of the groups.
    :param np.ndarray counts: the sample size of the groups.
    :rtype: (float, float, int)
    :return: a tuple containing the combined population mean, standard deviation and total count.
    """
    tn = counts.sum()
    if len(means) < 2 or tn < 2:
        return means[0], stds[0], counts[0]

    sum_x = means * counts
    sum_x2 = stds * stds * (counts - 1) + ((sum_x * sum_x) / counts)

    tx = sum_x.sum()
    txx = sum_x2.sum()
    c_m = tx / tn
    c_sd = np.sqrt((txx - (tx * tx) / tn) / (tn - 1))
    return c_m, c_sd, tn
