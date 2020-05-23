__author__ = 'Pedro Sequeira'
__email__ = 'pedrodbs@gmail.com'

from pyfpgrowth.pyfpgrowth import FPTree, FPNode
from itertools import combinations


def filter_closed(freq_sets, counts):
    """
    Given a list of all frequent item-sets and their counts, compute the set of closed frequent sets.
    This is only a very simplistic (naive) implementation for demonstration purposes!
    """
    for super_set_lst in list(freq_sets.keys()):
        super_set = set(super_set_lst)
        for sub_set_lst in list(freq_sets.keys()):
            sub_set = set(sub_set_lst)
            if super_set != sub_set and counts[super_set_lst] == counts[sub_set_lst] and sub_set.issubset(super_set):
                del freq_sets[sub_set_lst]


def filter_maximal(freq_sets):
    """
    Given a list of all frequent item-sets and their counts, compute the set of maximal frequent sets.
    This is only a very naive implementation for demonstration purposes!
    """
    for super_set_lst in list(freq_sets.keys()):
        super_set = set(super_set_lst)
        for sub_set_lst in list(freq_sets.keys()):
            sub_set = set(sub_set_lst)
            if super_set != sub_set and sub_set.issubset(super_set):
                del freq_sets[sub_set_lst]


def generate_association_rules(patterns, counts, threshold):
    """
    Gets a list of association rules in the form (left, right, count, confidence) given a set of frequent itemsets.
    :param dict patterns: the frequent itemsets and corresponding index from which to derive the association rules.
    :param dict counts: the frequency counts associated with each pattern and sub-pattern.
    :param float threshold: the minimum confidence for a rule to be considered pertinent.
    :return list: a list of association rules in the form (left, right, count, confidence).
    """
    rules = []
    for itemset in patterns.keys():
        upper_support = counts[itemset]

        for i in range(1, len(itemset)):
            for antecedent in combinations(itemset, i):
                antecedent = tuple(sorted(antecedent))
                consequent = tuple(sorted(set(itemset) - set(antecedent)))

                if antecedent in counts:
                    lower_support = counts[antecedent]
                    confidence = float(upper_support) / lower_support

                    if confidence >= threshold:
                        rules.append((antecedent, consequent, counts[itemset], confidence))

    return rules


def find_patterns_above(tree, jacc_threshold):
    """
    Finds all interesting patterns (itemsets) in the given tree with a Jaccard index above or equal to a threshold.
    :param FPTree tree: the tree containing the prefix counts.
    :param float jacc_threshold: the value of the index above which an itemset is a pattern.
    :return dict, dict, dict: the patterns found, the non-patterns found (and tested) and the itemset counts (tested).
    """
    counts = {}
    patterns = {}
    no_patterns = {}
    _find_patterns_above_recursive(tree, jacc_threshold, tree.root, (), counts, patterns, no_patterns)
    return patterns, no_patterns, counts


def _find_patterns_above_recursive(tree, jacc_threshold, node, itemset, counts, patterns, no_patterns):
    for child in node.children:

        # creates new pattern by adding item (correct order is maintained)
        new_itemset = itemset + (child.value,)

        # gets Jaccard index (checks if already calculated)
        if new_itemset in patterns:
            keep_search = True
        elif new_itemset in no_patterns:
            keep_search = False
        else:
            # calculates index and adds to corresponding table
            jacc = get_jaccard(tree, new_itemset, counts)
            if jacc >= jacc_threshold:
                patterns[new_itemset] = jacc
                keep_search = True
            else:
                no_patterns[new_itemset] = jacc
                keep_search = False

        # checks keep searching with new pattern
        if keep_search:
            _find_patterns_above_recursive(tree, jacc_threshold, child, new_itemset, counts, patterns, no_patterns)

        # in any case keep looking with original pattern
        _find_patterns_above_recursive(tree, jacc_threshold, child, itemset, counts, patterns, no_patterns)


def get_pattern_tree(patterns):
    root = FPNode(None, None, None)
    for pattern in patterns:
        _insert_pattern_tree(root, pattern)
    return root


def _insert_pattern_tree(node, itemset):
    if len(itemset) == 0:
        return node

    item = itemset[0]
    child = node.get_child(item)
    if child is None:
        child = node.add_child(item)
    return _insert_pattern_tree(child, itemset[1:])


def _get_pattern_itemset(node):
    itemset = []
    while node.parent is not None:
        itemset.append(node.value)
        node = node.parent
    itemset.reverse()
    return tuple(itemset)


def find_patterns_below(tree, jacc_threshold):
    """
    Finds all interesting patterns (itemsets) in the given tree with a Jaccard index below the given threshold. The
    patterns are minimal, i.e., all super-patterns are also patterns.
    :param FPTree tree: the tree containing the prefix counts.
    :param float jacc_threshold: the value of the index below which an itemset is a pattern.
    :return list: the patterns found.
    """
    # first find all patterns above the threshold
    patterns_above, _, _ = find_patterns_above(tree, jacc_threshold)

    # creates a pattern tree from the found patterns
    root = get_pattern_tree(patterns_above)

    # searches for all minimal patterns using a parallel search tree
    all_items = list(tree.frequent.keys())
    all_items.sort()
    all_items.sort(key=lambda x: tree.frequent[x], reverse=True)
    search_root = FPNode(None, None, None)
    patterns_below = []
    for item in all_items:
        _find_patterns_below_recursive(root, search_root, item, tuple(), patterns_below)

    return patterns_below


def _find_patterns_below_recursive(node, search_node, item, itemset, patterns):
    child = node.get_child(item)
    if child is None:

        patterns.append(itemset + (item,))

    else:

        for search_child in search_node.children:
            child = node.get_child(search_child.value)
            _find_patterns_below_recursive(child, search_child, item, itemset + (child.value,), patterns)

        search_node.add_child(item)


def get_jaccard(tree, itemset, counts):
    """
    Gets the Jaccard index of the given itemset.
    :param FPTree tree: the tree containing the prefix counts.
    :param Iterable itemset: the itemset that we want to get the Jaccard index.
    :param dict counts: a dictionary (possibly empty) containing the counts of the sub-itemsets.
    :return float: the Jaccard index of the given itemset.
    """
    if len(itemset) == 0:
        return 0.

    # gets all sub-combinations
    sub_itemsets = []
    for i in range(1, len(itemset) + 1):
        sub_itemsets.extend(combinations(itemset, i))

    # gets denominator's sum
    sum_count = 0
    for sub_itemset in sub_itemsets:
        sub_itemset = tuple(sub_itemset)
        if sub_itemset in counts:
            count = counts[sub_itemset]
        else:
            # gets count from tree
            count = get_count(tree, sub_itemset)
            counts[sub_itemset] = count

        # adds count to sum (if even itemset, then count is negated)
        sum_count += - count if len(sub_itemset) % 2 == 0 else count

    # returns the Jaccard index
    return float(counts[tuple(itemset)]) / sum_count


def get_count(tree, itemset):
    """
    Gets the number of times that the itemset was added into the tree.
    :param FPTree tree: the tree containing the prefix counts.
    :param Iterable itemset: the itemset that we want to get the count.
    :return int: the itemset count.
    """
    if len(itemset) == 0:
        return 0

    # gets next item and goes through all its node links
    next_item = itemset[0]
    count = 0
    node = tree.headers[next_item]
    while node is not None:
        count += _get_count_recursive(tree, node, itemset[1:])
        node = node.link
    return count


def _get_count_recursive(tree, node, itemset):
    if len(itemset) == 0:
        return node.count

    count = 0
    next_item = itemset[0]
    next_item_count = tree.frequent[next_item]
    for child in node.children:

        # if child node corresponds to item, remove it from sequence and keep searching
        if child.value == next_item:
            count += _get_count_recursive(tree, child, itemset[1:])

        # else if count of child node's item is less, it means there's no node down corresponding to the item
        elif tree.frequent[child.value] < next_item_count:
            continue

        # otherwise keep looking for the item along the child's path
        else:
            count += _get_count_recursive(tree, child, itemset)

    return count


def print_tree(tree):
    """
    Utility method that prints the tree to the console in a horizontal layout.
    :param FPTree tree: the tree to be printed.
    :return:
    """
    _print_tree_recursive(tree.root)


def _print_tree_recursive(node, depth=0):
    print('{}{}:{}'.format('\t' * depth, node.value, node.count))
    for child in node.children:
        _print_tree_recursive(child, depth + 1)
