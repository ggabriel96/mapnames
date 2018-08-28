import operator as op
import statistics as st

import numpy as np

from Strings import wagner_fischer


class Vertex:
    def __init__(self, label, idx=None):
        self.label = label
        self.idx = idx
        self.prefs = None
        self.ratings = None

    def set_prefs(self, others, h_fn, sort=False, ref_only=False):
        """ Sets the preference list of this vertex according to the results of
        h_fn.

        :param others: set of other vertices to compute preference list against
        :param h_fn: must be a function accepting two vertices and returning
        an integer that describes how closely related these two vertices are.
        In this case, the function will be called with self and another
        vertex in others. The integers will be used to sort the preference
        list. They need not be the final vertex position.
        :param sort: if the preference list should be sorted at the end
        :param ref_only: if the preference list should be left with only
        references to vertices instead of (vertex reference, h_fn result)
        """
        self.ratings = [(other, h_fn(self, other)) for other in others]
        if sort:
            self.ratings.sort(key=op.itemgetter(1))
        if ref_only:
            self.restore_prefs()

    def restore_prefs(self):
        self.prefs = [other for (other, _) in self.ratings]

    def __str__(self):
        return self.label

    def __repr__(self):
        return self.__str__()


class BipartiteMatcher:
    def __init__(self, left, right, filter_class=None):
        """
        :param left: left set of elements
        :param right: right set of elements
        :param filter_class: a callable Suffix Tree-like class to build
        filters for left and right sets. Upon called with a string, must
        return a list of indexes of candidates to compare to that string.
        """
        self.left = np.array([Vertex(left[l], l) for l in range(len(left))])
        self.right = np.array([Vertex(right[r], r) for r in range(len(right))])
        self.filter_on_left = None
        self.filter_on_right = None
        self.prefs_min = None
        self.prefs_max = None
        self.prefs_mean = None
        self.prefs_std = None
        self.prefs_qtiles = None
        if filter_class is not None:
            # filter_on_left is used to filter left-side candidates,
            # thus should be queried with a right vertex (and vice-versa)
            self.filter_on_left = filter_class(left)
            self.filter_on_right = filter_class(right)

    def stable_match(self):
        """ Irving weakly-stable marriage algorithm.

        (an extension to Gale-Shapley's).
        :return: a dict mapping the vertices of U to V.
        """
        husband = {}
        matching = {}
        # preferences list will get screwed...
        free_men = set(self.left)
        while len(free_men) > 0:
            m = free_men.pop()
            w = m.prefs[0]

            # if some man h is engaged to w,
            # set him free
            h = husband.get(w)
            if h is not None:
                del matching[h]
                free_men.add(h)

            # m engages w
            matching[m] = w
            husband[w] = m

            # for each successor m' of m on w's preferences,
            # remove w from m's preferences so that no man
            # less desirable than m will propose w
            succ_index = w.prefs.index(m) + 1
            for i in range(succ_index, len(w.prefs)):
                successor = w.prefs[i]
                successor.prefs.remove(w)
            # and delete all m' from w so we won't attempt
            # to remove w from their list more than once
            del w.prefs[succ_index:]
        return matching

    def set_prefs(self, h_fn):
        """ Sets the preference list for all vertices in the left set against
        all in the right, and all in the right set against all in the left.

        :param h_fn: see Vertex.set_prefs
        """
        prefs_len = []
        for these, those, filter_on_them in \
                zip([self.left, self.right],
                    [self.right, self.left],
                    [self.filter_on_right, self.filter_on_left]):
            for this in these:
                if filter_on_them is not None:
                    filtrd_idxs = filter_on_them(this.label)
                    them = those[filtrd_idxs]
                    prefs_len.append(len(them))
                else:
                    them = those
                this.set_prefs(them, h_fn)
        if self.filter_on_left is not None or self.filter_on_right is not None:
            self.prefs_min = min(prefs_len)
            self.prefs_max = max(prefs_len)
            self.prefs_mean = st.mean(prefs_len)
            self.prefs_std = st.pstdev(prefs_len, self.prefs_mean)
            self.prefs_qtiles = np.percentile(prefs_len, [25, 50, 75])

    def restore_prefs(self):
        for u in self.left:
            u.restore_prefs()
        for v in self.right:
            v.restore_prefs()


# Did not put on Vertex class because
# this is too non-standard for graphs
def vertex_diff(u, v):
    ed, _ = wagner_fischer(u.label, v.label, True)
    return ed
