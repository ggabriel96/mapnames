import operator as op
import statistics as st
from abc import ABC, abstractmethod

import numpy as np
from ortools.graph import pywrapgraph as ortg
from tqdm import tqdm

import mapnames.string


def vertex_diff(u, v, string_metric, **kwargs):
    return string_metric(u.label, v.label, **kwargs)


class Vertex:
    def __init__(self, label, idx=None):
        self.label = label
        self.idx = idx
        self.prefs = None
        self.ratings = None
        self.__ratings = None

    def set_ratings(self, others, h_fn, sort=False, also_prefs=False):
        """ Sets the rating list of this vertex according to the results of
        h_fn.

        As setting the ratings might be computationally expensive, this method
        stores the result in an internal attribute that is exposed through
        the public self.ratings one. It is safe to modify self.ratings as it is
        recoverable through this internal backup with self.restore_ratings().

        :param others: set of other vertices to compute rating list against
        :param h_fn: must be a function accepting two vertices and returning
                     an integer that describes how closely related these two
                     vertices are. The function will be called with self and
                     another vertex in others. The integers will be used to
                     sort the list, if asked. They need not be the final
                     vertex position.
        :param sort: if the rating list should be sorted at the end
        :param also_prefs: if the preference list should be set at the end (will
                           also sort the ratings before actually setting prefs).
        """
        self.__ratings = [(other, h_fn(self, other)) for other in others]
        if sort or also_prefs:
            self.__ratings.sort(key=op.itemgetter(1))
        if also_prefs:
            self.restore_prefs()
        self.restore_ratings()

    def restore_ratings(self):
        """ Set self.ratings based on an internal backup of it.

        For more information, see self.set_ratings().
        """
        self.ratings = self.__ratings.copy()

    def restore_prefs(self):
        """ Set self.prefs based on self.ratings.

        Will make self.prefs store only the reference to the vertices, making
        it a "real" preference list. Mainly for stable marriage or algorithms
        that alter the preference lists during their execution.
        """
        self.prefs = [other for (other, _) in self.__ratings]

    def __str__(self):
        return self.label

    def __repr__(self):
        return self.__str__()


################################################################################
# ABSTRACT BASE CLASSES
################################################################################

class BipartiteMatcher(ABC):
    def __init__(self):
        self.left = None
        self.right = None
        self.n = None

    @abstractmethod
    def match(self):
        """ Run the matching implemented """
        pass

    @abstractmethod
    def set_prefs(self, h_fn):
        """ Set the necessary preference lists according to h_fn.

        See Vertex.set_ratings() for more details regarding h_fn.
        """
        pass

    @abstractmethod
    def accuracy(self, correct_mapping, opt_dict_out=None):
        """ Computes the achieved accuracy according to correct_mapping.

        The accuracy must be between 0.0 and 1.0. Other optional return values
        may be returned through opt_dict_out, e.g. the list of mismatches (of
        which the elements may vary per implementation).

        :param correct_mapping: a dict holding the correct mapping
        :param opt_out: optional dictionary for additional output values
        """
        pass


class CompleteBipartiteMatcher(BipartiteMatcher, ABC):
    def set_prefs(self, h_fn):
        """ Sets the preference list for all vertices in the left set against
        all in the right, and all in the right set against all in the left.

        Shows a progress-bar for each of the two runs.

        :param h_fn: see Vertex.set_ratings()
        """
        for l in tqdm(self.left):
            l.set_ratings(self.right, h_fn, also_prefs=True)
        for r in tqdm(self.right):
            r.set_ratings(self.left, h_fn, also_prefs=True)

    def restore_prefs(self):
        for l in self.left:
            l.restore_prefs()
        for r in self.right:
            r.restore_prefs()


class IncompleteBipartiteMatcher(BipartiteMatcher, ABC):
    def __init__(self):
        super().__init__()

        # filter_on_left is used to filter left-side candidates,
        # thus should be queried with a right vertex (and vice-versa)
        self.filter_on_left = None
        self.filter_on_right = None

        # preference statistics computed on self.set_prefs
        # if a filter was provided
        self.prefs_min = None
        self.prefs_max = None
        self.prefs_mean = None
        self.prefs_std = None
        self.prefs_qtiles = None

    def set_prefs(self, h_fn, sort=False, also_prefs=False):
        """ Sets the preference list for all vertices in the left set against
        some in the right, and all in the right set against some in the left,
        where 'some' depends on the provided filter.

        If no filter was provided, 'some' will be 'all'. Shows a progress-bar
        for each of the two runs.

        For a description of the parameters, see Vertex.set_ratings().
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
                this.set_ratings(them, h_fn, sort, also_prefs)
        if self.filter_on_left is not None or self.filter_on_right is not None:
            self.prefs_min = min(prefs_len)
            self.prefs_max = max(prefs_len)
            self.prefs_mean = st.mean(prefs_len)
            self.prefs_std = st.pstdev(prefs_len, self.prefs_mean)
            self.prefs_qtiles = np.percentile(prefs_len, [25, 50, 75])


class IncompleteStableMarriage(IncompleteBipartiteMatcher, ABC):
    def __init__(self, left, right, filter_class=mapnames.string.SuffixArray):
        """
        :param left: left set of elements
        :param right: right set of elements
        :param filter_class: a callable class to build filters for left and
                             right sets (so the constructor will be called with
                             them). Upon called with a string, must return a
                             list of indexes of candidates to compare to that
                             string.
        """
        super().__init__()

        self.n = len(left)

        # holds the result of self.match()
        self.assignment = None

        if filter_class is not None:
            self.filter_on_left = filter_class(left)
            self.filter_on_right = filter_class(right)

        self.left = np.array([Vertex(left[l], l) for l in range(len(left))])
        self.right = np.array([Vertex(right[r], r) for r in range(len(right))])

    def accuracy(self, correct_mapping, opt_dict_out=None):
        """ Computes the achieved accuracy and the list of mismatches.

        The accuracy takes into account unmatched elements as wrong matchings.

        :param correct_mapping: a dict holding the correct mapping
        :param opt_dict_out: if not None, a list of mismatched and a list of
                             unmatched Vertex elements from self.left
        :return: the accuracy
        """
        errors = []
        for l, r in self.assignment.items():
            is_equal = r.label == correct_mapping[l.label]
            if not is_equal:
                errors.append(l)
        unmatched = self.get_unmatched()
        if opt_dict_out is not None:
            opt_dict_out['errors'] = errors
            opt_dict_out['unmatched'] = unmatched
        return (self.n - len(errors) - len(unmatched)) / self.n

    def get_unmatched(self):
        return [l for l in self.left if l not in self.assignment]


################################################################################
# IMPLEMENTATIONS
################################################################################

class StableMarriage(CompleteBipartiteMatcher):
    def __init__(self, left, right):
        """
        :param left: left set of elements
        :param right: right set of elements
        """
        super().__init__()

        self.n = len(left)

        # holds the result of self.match()
        self.assignment = None

        self.left = np.array([Vertex(l) for l in left])
        self.right = np.array([Vertex(r) for r in right])

    def match(self):
        """ Run Irving's weakly-stable marriage algorithm.

        It is Irving's extension to Gale-Shapley's. Must be called after a call
        to self.set_ratings(also_prefs=True) and will ruin the preference lists
        (they can be restored with self.restore_prefs()). Stores its result in
        self.assignment.
        """
        husband = {}
        self.assignment = {}
        # preference lists will get screwed...
        free_men = set(self.left)
        while len(free_men) > 0:
            m = free_men.pop()
            w = m.prefs[0]

            # if some man h is engaged to w,
            # set him free
            h = husband.get(w)
            if h is not None:
                del self.assignment[h]
                free_men.add(h)

            # m engages w
            self.assignment[m] = w
            husband[w] = m

            # for each successor m' of m in w's preferences,
            # remove w from the preferences of m' so that no man
            # less desirable than m will propose w
            succ_index = w.prefs.index(m) + 1
            for i in range(succ_index, len(w.prefs)):
                successor = w.prefs[i]
                successor.prefs.remove(w)
            # and delete all m' from w's preferences so we won't
            # attempt to remove w from their list more than once
            del w.prefs[succ_index:]

    def accuracy(self, correct_mapping, opt_dict_out=None):
        """ Computes the achieved accuracy and the list of mismatches.

        :param correct_mapping: a dict holding the correct mapping
        :param opt_dict_out: if not None, a list of mismatched Vertex elements
                             from self.left
        :return: the accuracy
        """
        errors = []
        for l, r in self.assignment.items():
            is_equal = r.label == correct_mapping[l.label]
            if not is_equal:
                errors.append(l)
        if opt_dict_out is not None:
            opt_dict_out['errors'] = errors
        return (self.n - len(errors)) / self.n


class SimpleMarriageAttempt(IncompleteStableMarriage):
    def match(self):
        """ Run a (somewhat bad) adaptation to Irving's weakly-stable marriage
        algorithm with support for incomplete preference lists.

        Must be called after a call to self.set_ratings() and will ruin the
        preference lists (they can be restored with self.restore_prefs()).
        Stores its result in self.assignment.
        """
        husbands = {}
        self.assignment = {}
        free_men = list(self.left)
        while free_men:
            man = free_men.pop(0)

            # man preferences might get emptied later on
            if not man.prefs:
                continue

            # pop to prevent infinite loop
            woman = man.prefs.pop(0)

            # if some man husband is engaged to woman,
            # check if man is better than him and, if so,
            # change the marriage
            husband = husbands.get(woman)
            if husband is not None:
                try:
                    man_idx = woman.prefs.index(man)
                    husband_idx = woman.prefs.index(husband)
                    # i < j in a preference list means that i-th
                    # person is more preferable than j-th person
                    should_change = man_idx < husband_idx
                except ValueError:
                    should_change = True

                if should_change:
                    del self.assignment[husband]
                    free_men.append(husband)
                else:
                    woman.prefs.remove(man)
                    continue

            # man engages woman
            self.assignment[man] = woman
            husbands[woman] = man

            # if man is in woman's preferences, then
            # for each successor man' of man in woman's preferences,
            # remove woman from the preferences of man' so that no man'
            # less desirable than man will propose woman
            try:
                succ_idx = woman.prefs.index(man) + 1
            except ValueError:
                continue
            for i in range(succ_idx, len(woman.prefs)):
                successor = woman.prefs[i]
                # no guarantee that preference list has woman,
                # because this inherits from FilteredBipartiteMatcher
                try:
                    successor.prefs.remove(woman)
                except ValueError:
                    pass
            # and delete all man' from woman's preferences so we won't
            # attempt to remove woman from their list more than once
            del woman.prefs[succ_idx:]

    def set_prefs(self, h_fn, sort=False, also_prefs=True):
        """ Auxiliary method to call super().set_prefs() with also_prefs=True
         by default """
        super().set_prefs(h_fn, sort, also_prefs)


class LeftGreedyMarriage(IncompleteStableMarriage):
    def match(self):
        """ Run a (somewhat bad) greedy and even more left-biased adaptation to
        Irving's weakly-stable marriage algorithm with support for incomplete
        preference lists.

        Must be called after a call to self.set_ratings() and will ruin the
        rating lists (they can be restored with self.restore_ratings()). Only
        takes into account the ratings of the left side. Stores its result in
        self.assignment.
        """
        husbands = {}
        self.assignment = {}
        free_men = list(self.left)
        while free_men:
            man = free_men.pop(0)

            if not man.ratings:
                continue

            # pop to prevent infinite loop
            woman, cost_man = man.ratings.pop(0)

            # if some man husband is engaged to woman,
            # check if man is better than him and, if so,
            # change the marriage
            husband_and_cost = husbands.get(woman)
            if husband_and_cost is not None:
                husband, cost_husband = husband_and_cost
                if cost_man < cost_husband:
                    del self.assignment[husband]
                    free_men.append(husband)
                else:
                    continue

            # man engages woman
            self.assignment[man] = woman
            husbands[woman] = man, cost_man

    def restore_ratings(self):
        for l in self.left:
            l.restore_ratings()
        for r in self.right:
            r.restore_ratings()


class MinCostFlow(IncompleteBipartiteMatcher):
    def __init__(self, left, right, filter_class=mapnames.string.SuffixArray):
        """
        :param left: left set of elements
        :param right: right set of elements
        :param filter_class: a callable class to build filters for left and
                             right sets (so the constructor will be called with
                             them). Upon called with a string, must return a
                             list of indexes of candidates to compare to that
                             string.
        """
        super().__init__()

        self.n = len(left)

        self.flow = ortg.SimpleMinCostFlow()
        self.solve_status = None
        self.source, self.sink = 0, 2 * self.n + 1
        # numbering shift required by Google's library
        self.l_idx_shift, self.r_idx_shift = 1, self.n + 1

        if filter_class is not None:
            self.filter_on_left = filter_class(left)
            self.filter_on_right = filter_class(right)

        self.left = np.array([Vertex(left[l], l + self.l_idx_shift)
                              for l in range(len(left))])
        self.right = np.array([Vertex(right[r], r + self.r_idx_shift)
                               for r in range(len(right))])

    def match(self):
        """ Calls self.flow.Solve() """
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

    def accuracy(self, correct_mapping, opt_dict_out=None):
        """ Computes the achieved accuracy and the list of mismatches.

        :param correct_mapping: a dict holding the correct mapping
        :param opt_dict_out: if not None, a list of mismatched arcs
        :return: the accuracy
        """
        if self.solve_status == ortg.SimpleMinCostFlow.OPTIMAL:
            errors = []
            for arc in range(self.flow.NumArcs()):
                if self.flow.Tail(arc) != self.source \
                        and self.flow.Head(arc) != self.sink \
                        and self.flow.Flow(arc) > 0:
                    l, r = self.endpoints(arc)
                    is_equal = self.right[r].label == \
                               correct_mapping[self.left[l].label]
                    if not is_equal:
                        errors.append(arc)
            if opt_dict_out is not None:
                opt_dict_out['errors'] = errors
            return (self.n - len(errors)) / self.n
        return 0.0
