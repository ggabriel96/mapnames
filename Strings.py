import numpy as np


def diff_view(str1, str2):
    """ Calculate the lengths of the longest common prefix
    and suffix between str1 and str2.

    Let str1 = axb of length m and str2 = ayb of length n,
    then this function finds and returns i and j such that:
    str1[0:i] = str2[0:i] = a and str1[m-j:] = str2[n-j:] = b.
    In the case that a or b does not exist (no common prefix
    or suffix), then i or j are 0.

    :param str1: the first string
    :param str2: the second string
    :return: common prefix and suffix lengths (i.e. i and j; see description)
    """
    m, n = len(str1), len(str2)
    len_limit = min(m, n)
    prefix_len, suffix_len = 0, 0
    while prefix_len < len_limit and str1[prefix_len] == str2[prefix_len]:
        prefix_len += 1
    # was using negative indexing,
    # I just think this way is better understandable
    while suffix_len < len_limit \
            and len_limit - suffix_len > prefix_len \
            and str1[m - 1 - suffix_len] == str2[n - 1 - suffix_len]:
        suffix_len += 1
    return prefix_len, suffix_len


def trim_both_equal(str1, str2):
    """ Removes common prefix and suffix of both str1 and str2.

    :param str1: the first string
    :param str2: the second string
    :return: str1 and str2 with their common prefix and suffix removed
    """
    begin, end = diff_view(str1, str2)
    if end == 0:
        return str1[begin:], str2[begin:]
    return str1[begin:-end], str2[begin:-end]


def wagner_fischer(str1, str2, trim=False):
    if trim:
        a, b = trim_both_equal(str1, str2)
    else:
        a, b = str1, str2
    m, n = len(a), len(b)
    D = np.empty((m + 1, n + 1), dtype=np.int64)
    # Positions of D represent the edit distance from s1 to s2 where row
    # i and column j means the edit distance from s1[0..i] to s2[0..j]. So:
    # D[0][0] = 0 because there is no operation to do on empty strings
    # D[i][0] = D[i - 1][0] + 1 (cost of deletion) for i from
    # 1 to len(str1) 'cause the only thing to do is delete all
    # characters of str1 until ''. Same for D[0][j].
    D[:, 0] = np.arange(m + 1)
    D[0, :] = np.arange(n + 1)
    for i in np.arange(1, m + 1):
        for j in np.arange(1, n + 1):
            # Change operation
            cost1 = D[i - 1][j - 1] + int(a[i - 1] != b[j - 1])
            # Minimum of deletion and insertion operations
            # Deletion means cost of transforming str1[0..i-1]
            # to str2[0..j] and deleting str1[i]
            # Insertion means cost of transforming str1[0..i]
            # to str2[0..j-1] and inserting str2[j]
            cost2 = np.minimum(D[i - 1][j] + 1, D[i][j - 1] + 1)
            D[i][j] = np.minimum(cost1, cost2)
    # [-1][-1] is the last column of the last row, which holds the edit
    # distance of the whole str1 and str2 strings
    return D[-1][-1]
