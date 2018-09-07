import operator as op


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
