__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

import copy
from collections import defaultdict

"""
All code adapted from: https://github.com/sequenceanalysis/sequenceanalysis.github.io/blob/master/notebooks/part2.ipynb
"""


def prefix_span(dataset, min_support, min_length=1, max_length=9999999):
    """
    The PrefixSpan algorithm. Computes the frequent sequences in a sequence dataset for a given min_support
    See: https://github.com/sequenceanalysis/sequenceanalysis.github.io/blob/master/notebooks/part2.ipynb
    :param list dataset: a list of sequences, for which the frequent (sub-)sequences are computed
    :param int min_support: the minimum support that makes a sequence frequent.
    :param int min_length: the minimum sequence length.
    :param int max_length: the maximum sequence length.
    :return list: a list of tuples (s, c), where s is a frequent sequence, and c is the count for that sequence.
    """
    result = []
    item_counts = generate_item_supports(dataset)
    for item, count in item_counts:
        if count >= min_support:
            new_prefix = [(item,)]
            if min_length <= 1:
                result.append((new_prefix, count))
            if max_length > 1:
                result.extend(_prefix_span_internal(
                    project_database(dataset, (item,), False), min_support, min_length, max_length, new_prefix))
    return result


def project_sequence(sequence, prefix, new_event):
    """
    Projects a sequence according to a given prefix, as done in PrefixSpan.
    See: https://github.com/sequenceanalysis/sequenceanalysis.github.io/blob/master/notebooks/part2.ipynb
    :param list sequence: the sequence the projection is built from.
    :param tuple prefix: the prefix that is searched for in the sequence.
    :param bool new_event: if set to True, the first itemset is ignored.
    :return: If the sequence does not contain the prefix, then None.
    Otherwise, a new sequence starting from the position of the prefix, including the itemset that includes the prefix.
    """
    result = None
    for i, itemset in enumerate(sequence):
        if result is None:
            if (not new_event) or i > 0:
                if all(x in itemset for x in prefix):
                    result = [tuple(itemset)]
        else:
            result.append(tuple(itemset))
    return result


def project_database(dataset, prefix, new_event):
    """
    Projects a dataset according to a given prefix, as done in PrefixSpan.
    See: https://github.com/sequenceanalysis/sequenceanalysis.github.io/blob/master/notebooks/part2.ipynb
    :param list dataset: the dataset the projection is built from.
    :param tuple prefix: the prefix that is searched for in the sequence.
    :param bool new_event: if set to True, the first itemset is ignored.
    :return list: a (potentially empty) list of sequences.
    """
    projected_db = []
    for sequence in dataset:
        seq_projected = project_sequence(sequence, prefix, new_event)
        if seq_projected is not None:
            projected_db.append(seq_projected)
    return projected_db


def generate_items(dataset):
    """
    See: https://github.com/sequenceanalysis/sequenceanalysis.github.io/blob/master/notebooks/part2.ipynb
    Generates a list of all items that are contained in a dataset
    """
    return sorted(set([item for sublist1 in dataset for sublist2 in sublist1 for item in sublist2]))


def generate_item_supports(dataset, ignore_first_event=False, prefix=()):
    """
    See: https://github.com/sequenceanalysis/sequenceanalysis.github.io/blob/master/notebooks/part2.ipynb
    Computes a defaultdict that maps each item in the dataset to its support
    """
    result = defaultdict(int)
    for sequence in dataset:
        if ignore_first_event:
            sequence = sequence[1:]
        co_occurring_items = set()
        for itemset in sequence:
            if all(x in itemset for x in prefix):
                for item in itemset:
                    if item not in prefix:
                        co_occurring_items.add(item)
        for item in co_occurring_items:
            result[item] += 1
    return sorted(result.items())


def _prefix_span_internal(dataset, min_support, min_length, max_length, prev_prefixes=[]):
    result = []

    # Add a new item to the last element (==same time)
    item_count_same_event = generate_item_supports(dataset, False, prefix=prev_prefixes[-1])
    for item, count in item_count_same_event:
        if (count >= min_support) and item > prev_prefixes[-1][-1]:
            new_prefix = copy.deepcopy(prev_prefixes)
            new_prefix[-1] += (item,)
            prefix_len = len(new_prefix)
            if prefix_len >= min_length:
                result.append((new_prefix, count))
            if prefix_len < max_length:
                result.extend(_prefix_span_internal(
                    project_database(dataset, new_prefix[-1], False), min_support, min_length, max_length, new_prefix))

    # Add a new event to the prefix
    item_count_subsequent_events = generate_item_supports(dataset, True)
    for item, count in item_count_subsequent_events:
        if count >= min_support:
            new_prefix = copy.deepcopy(prev_prefixes)
            new_prefix.append((item,))
            prefix_len = len(new_prefix)
            if prefix_len >= min_length:
                result.append((new_prefix, count))
            if prefix_len < max_length:
                result.extend(_prefix_span_internal(
                    project_database(dataset, (item,), True), min_support, min_length, max_length, new_prefix))
    return result


def filter_closed(result):
    """
    See: https://github.com/sequenceanalysis/sequenceanalysis.github.io/blob/master/notebooks/part2.ipynb
    Given a list of all frequent sequences and their counts, compute the set of closed frequent sequence (as a list)
    This is only a very simplistic (naive) implementation for demonstration purposes!
    """
    for super_sequence, count_seq in list(result):
        for sub_sequence, count_sub_seq in list(result):
            if sub_sequence != super_sequence and count_seq == count_sub_seq and \
                    is_sub_sequence(super_sequence, sub_sequence):
                result.remove((sub_sequence, count_sub_seq))


def filter_maximal(result):
    """
    See: https://github.com/sequenceanalysis/sequenceanalysis.github.io/blob/master/notebooks/part2.ipynb
    Given a list of all frequent sequences and their counts, compute the set of maximal frequent sequence (as a list)
    This is only a very naive implementation for demonstration purposes!
    """
    for super_sequence, count_seq in list(result):
        for sub_sequence, count_sub_seq in list(result):
            if sub_sequence != super_sequence and is_sub_sequence(super_sequence, sub_sequence):
                result.remove((sub_sequence, count_sub_seq))


def is_sub_sequence(main_sequence, sub_sequence):
    """
    See: https://github.com/sequenceanalysis/sequenceanalysis.github.io/blob/master/notebooks/part2.ipynb
    This is a simple recursive method that checks if subsequence is a sub_sequence of main_sequence
    """
    sub_sequence_clone = list(sub_sequence)  # clone the sequence, because we will alter it
    return _is_sub_sequence_recursive(main_sequence, sub_sequence_clone)  # start recursion


def _is_sub_sequence_recursive(main_sequence, sub_sequence_clone, start=0):
    """
    See: https://github.com/sequenceanalysis/sequenceanalysis.github.io/blob/master/notebooks/part2.ipynb
    Function for the recursive call of is_sub_sequence, not intended for external calls
    """
    # Check if empty: End of recursion, all itemsets have been found
    if not sub_sequence_clone:
        return True
    # retrieves element of the subsequence and removes is from subsequence
    first_elem = set(sub_sequence_clone.pop(0))
    # Search for the first itemset...
    for i in range(start, len(main_sequence)):
        if set(main_sequence[i]).issuperset(first_elem):
            # and recurse
            return _is_sub_sequence_recursive(main_sequence, sub_sequence_clone, i + 1)
    return False
