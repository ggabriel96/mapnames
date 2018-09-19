import operator as op
from collections import Counter

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


def print_trace(str1, str2, trace):
    """ Prints the sequence of edit operations performed by wagner_fischer()
    to transform str1 into str2.

    Indicates deletion with '-', insertion with '+' and change with '.'.

    :param str1: a string
    :param str2: the string that str1 was compared to
    :param trace: the trace obtained from wf_trace()
    """
    print(f'str1: {str1}')
    print('      ', end='')
    for op in trace[::-1]:
        if op == -1:
            print('-', end='')
        elif op == +1:
            print('+', end='')
        else:
            print(f'.', end='')
    print(f'\nstr2: {str2}')


def wf_trace(D, str1, str2):
    """ Gets the list of operations performed by wagner_fischer().

    :param D: the memo matrix D used in wagner_fischer()
    :param str1: a string
    :param str2: the string that str1 was compared to
    :return: a list of -1, 0, +1 indicating insert, change, delete (from end to
             beginning of a and str2)
    """
    trace = []
    i, j = len(str1), len(str2)
    while i > 0 and j > 0:
        if D[i][j] == D[i - 1][j] + 1:
            trace.append(-1)
            i -= 1
        elif D[i][j] == D[i][j - 1] + 1:
            trace.append(+1)
            j -= 1
        else:
            trace.append(0)
            i -= 1
            j -= 1
    return trace


def wagner_fischer(str1, str2, trim=False, with_trace=False):
    """ Calculates the edit distance from str1 to str2.

    :param str1: a string
    :param str2: a string to compare str1 to
    :param trim: if should remove equal prefix and suffix between str1 and str2
    :param with_trace: if should also return the sequence of performed
                       operations
    :return: the edit distance and the sequence of performed operations
             if with_trace = True
    """
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
            change_cost = D[i - 1][j - 1] + int(a[i - 1] != b[j - 1])
            # Minimum of deletion and insertion operations
            # Deletion means cost of transforming str1[0..i-1]
            # to str2[0..j] and deleting str1[i]
            # Insertion means cost of transforming str1[0..i]
            # to str2[0..j-1] and inserting str2[j]
            delete_cost = D[i - 1][j] + 1
            insert_cost = D[i][j - 1] + 1
            d_or_i_cost = np.minimum(delete_cost, insert_cost)
            D[i][j] = np.minimum(change_cost, d_or_i_cost)
    # [-1][-1] is the last column of the last row, which holds the edit
    # distance of the whole str1 and str2 strings
    trace = None
    if with_trace:
        trace = wf_trace(D, a, b)
    return D[-1][-1], trace


def edit_distance(str1, str2, trim=False, with_trace=False):
    dist, _ = wagner_fischer(str1, str2, trim, with_trace)
    return dist


def qprofile(string, q):
    n = len(string)
    qgrams = [string[i:i + q] for i in range(n - q + 1)]
    return Counter(qgrams)


def qdistance(str1, str2, q):
    profile1 = qprofile(str1, q)
    profile2 = qprofile(str2, q)
    profile1.subtract(profile2)
    return sum((abs(count) for count in profile1.values()))


class QProfile(Counter):
    def __init__(self, param, q=2):
        """
        If a QProfile is passed in through param, a new object is constructed
        that is a copy of it. In this case, the q parameter is ignored.

        :param param: string or QProfile
        :param q: q-value for this profile
        """
        if isinstance(param, str):
            self.q = q
            n = len(param)
            qgrams = [param[i:i + q] for i in range(n - q + 1)]
            super().__init__(qgrams)
        elif isinstance(param, QProfile):
            self.q = param.q
            super().__init__(param)
        else:
            raise TypeError(f'object of type {type(param)} not supported')

    def __sub__(self, other):
        """ Returns a new QProfile that is the subtraction of self and other """
        if not isinstance(other, QProfile):
            return NotImplemented
        result = QProfile(self)
        result.subtract(other)
        return result

    def distance_to(self, other):
        diff = self - other
        return sum((abs(count) for count in diff.values()))


class SuffixArray:
    def __init__(self, strings):
        """ Saves the strings and builds the Suffix Array.

        :param strings: list of strings to build the Suffix Array on
        """
        self.strings = strings
        self.suffixes = None
        self.build()

    def build(self):
        """ Builds the Suffix Array based on saved strings """
        self.suffixes = [(self.strings[i][j:], i, j)
                         for i in range(len(self.strings))
                         for j in range(len(self.strings[i]))]
        self.suffixes.sort()

    def binary_search(self, string, lower=True):
        """ Searches for lower or upper bound for string
        on self.suffixes with a binary search.

        Method based on stringMatching function from
        Competitive Programming 3 book (page 259).

        :param string: string to search for
        :param lower: True to return the lower bound or
                      False to return the upper bound
        :return: the lower or upper bound, according to lower
        """
        if lower:
            cmp = op.ge
        else:
            cmp = op.gt
        strlen = len(string)
        lo, hi = 0, len(self.suffixes) - 1
        while lo < hi:
            mid = int(lo + (hi - lo) / 2)
            suffix = self.suffixes[mid][0]
            # print(lo, mid, hi, suffix)
            if cmp(suffix[:strlen], string):
                hi = mid
            else:
                lo = mid + 1
        # special case: if searching for upper bound and
        # last suffix is not equal to the query string,
        # then decrease upper bound (should we?)
        if not lower:
            if self.suffixes[hi][0][:strlen] != string:
                hi -= 1
        return lo if lower else hi

    def suffix_bounds(self, string):
        """ Searches for both lower and upper bounds
        for string on self.suffixes.

        :param string: string to search for
        :return: tuple containing lower and upper bounds of self.suffixes
        """
        lower_bound = self.binary_search(string)
        upper_bound = self.binary_search(string, lower=False)
        if lower_bound > upper_bound:
            lower_bound, upper_bound = upper_bound, lower_bound
        return lower_bound, upper_bound

    def indices_between_bounds(self, string):
        """ Given the lower and upper bounds on self.suffixes,
        returns a list of indices of self.strings which point
        to the strings that own the suffixes between the bounds.

        (What a good explanation, huh)

        :param string: string to search for
        :return: list of indices pointing to filtered strings of self.strings
        """
        lower_bound, upper_bound = self.suffix_bounds(string)
        return [self.suffixes[i][1]
                for i in range(lower_bound, upper_bound + 1)]

    def __call__(self, string):
        return self.indices_between_bounds(string)
