import operator as op
import statistics as st

import numpy as np
from ortools.graph import pywrapgraph as ortg
from tqdm import tqdm

from Strings import qdistance


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
               self.stable_match won't work if a filter is provided.
        """
        self.n = len(left)

        self.flow = None
        self.solve_status = None
        self.source, self.sink = None, None
        self.l_idx_shift, self.r_idx_shift = 0, 0

        self.marriage = None

        self.prefs_min = None
        self.prefs_max = None
        self.prefs_mean = None
        self.prefs_std = None
        self.prefs_qtiles = None

        self.filter_on_left = None
        self.filter_on_right = None
        if filter_class is not None:
            self.source, self.sink = 0, 2 * self.n + 1
            self.l_idx_shift, self.r_idx_shift = 1, self.n + 1
            # filter_on_left is used to filter left-side candidates,
            # thus should be queried with a right vertex (and vice-versa)
            self.filter_on_left = filter_class(left)
            self.filter_on_right = filter_class(right)

        self.left = np.array([Vertex(left[l], l + self.l_idx_shift)
                              for l in range(len(left))])
        self.right = np.array([Vertex(right[r], r + self.r_idx_shift)
                               for r in range(len(right))])

    def min_cost_flow(self):
        """ Builds pywrapgraph.SimpleMinCostFlow and calls Solve() """
        self.flow = ortg.SimpleMinCostFlow()

        # arcs from source to all left v
        for l in self.left:
            self.flow.AddArcWithCapacityAndUnitCost(tail=self.source,
                                                    head=l.idx,
                                                    capacity=1, unit_cost=0)
        # arcs from all right v to sink
        for r in self.right:
            self.flow.AddArcWithCapacityAndUnitCost(tail=r.idx,
                                                    head=self.sink,
                                                    capacity=1, unit_cost=0)
        # arcs from preferences of left vertices
        for l in self.left:
            for r, cost in l.ratings:
                self.flow.AddArcWithCapacityAndUnitCost(tail=l.idx,
                                                        head=r.idx,
                                                        capacity=1,
                                                        unit_cost=int(cost))
        # arcs from preferences of right vertices
        for r in self.right:
            for l, cost in r.ratings:
                self.flow.AddArcWithCapacityAndUnitCost(tail=l.idx,
                                                        head=r.idx,
                                                        capacity=1,
                                                        unit_cost=int(cost))

        self.flow.SetNodeSupply(self.source, self.n)
        self.flow.SetNodeSupply(self.sink, -self.n)

        self.solve_status = self.flow.Solve()

    def endpoints(self, arc):
        """ Returns the normalized endpoints of arc.

        The normalized endpoints of an arc are the indices of self.left and
        self.right.

        :param arc: the query arc
        :return: (left, right) indices
        """
        l = self.flow.Tail(arc) - self.l_idx_shift
        r = self.flow.Head(arc) - self.r_idx_shift
        return l, r

    def stable_match(self):
        """ Run Irving's weakly-stable marriage algorithm.

        It is Irving's extension to Gale-Shapley's. Requires self to hold a
        complete bipartite graph (i.e., full preference lists).
        """
        husband = {}
        self.marriage = {}
        # preferences list will get screwed...
        free_men = set(self.left)
        while len(free_men) > 0:
            m = free_men.pop()
            w = m.prefs[0]

            # if some man h is engaged to w,
            # set him free
            h = husband.get(w)
            if h is not None:
                del self.marriage[h]
                free_men.add(h)

            # m engages w
            self.marriage[m] = w
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
            for this in tqdm(these):
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

    def accuracy(self, correct_mapping):
        if self.marriage is not None:
            return self.__accuracy_marriage(correct_mapping)
        return self.__accuracy_min_cost_flow(correct_mapping)

    def __accuracy_marriage(self, correct_mapping):
        errors = []
        equal_amount = 0
        for man, woman in self.marriage.items():
            is_equal = woman.label == correct_mapping[man.label]
            if is_equal:
                equal_amount += 1
            else:
                errors.append(man)
        return equal_amount / len(self.marriage), errors

    def __accuracy_min_cost_flow(self, correct_mapping):
        errors = []
        equal_amount = 0
        for arc in range(self.flow.NumArcs()):
            if self.flow.Tail(arc) != self.source \
                    and self.flow.Head(arc) != self.sink \
                    and self.flow.Flow(arc) > 0:
                l, r = self.endpoints(arc)
                is_equal = self.right[r].label == \
                           correct_mapping[self.left[l].label]
                if is_equal:
                    equal_amount += 1
                else:
                    errors.append(arc)
        return equal_amount / self.n, errors


# Did not put on Vertex class because
# this is too non-standard for graphs
def vertex_diff(u, v):
    dist = qdistance(u.label, v.label, 2)
    # dist, _ = wagner_fischer(u.label, v.label, True)
    return dist
