import progressbar

"""
This module performs a crappy matching, inspired by GaleShapley's stable matching.

The crappy matching only takes in account the preferences of the X partition and ignores the
preferences of the Y partition.
The main idea of this matching is that it only needs O(n * d) in memory, where d is the
maximum degree of G.
"""
class G:
    """
    A graph, to store the preferences, before running crappy matching.
    """

    def __init__(self, n):
        """
        Creates an empty graph.

        Parameters
        ----------
        n : int
            The number of vertices. Vertices will be identified by i < n.
        """
        self.p = []
        for i in range(n):
            self.p.append({})

    def grade(self, ai, bi, v):
        """
        Upgrades the weight of (ai,bi) by v in self.

        Parameters
        ----------
        ai : int
            A vertex in the X partition.
        bi : int
            A vertex in the Y partition.
        v : float
            A score, added to the current score of (ai, bi).
            If there was no score for (ai, bi), the scores is set as v.
        """
        if not bi in self.p[ai]:
            self.p[ai][bi] = 0
        self.p[ai][bi] += v

    def makeprefs(self):
        """
        Compiles the lists of preferences from self.

        Returns
        -------
        list(list(int))
            A list, containing lists of integers.
            The outer list represents the vertices in X. The inner lists represent vertices in Y.
            The vertices in the inner lists are ordered according to the preferences of X.
        """

        ps = [[(b, a) for (a,b) in x.items()] for x in self.p]
        for p in ps:
            p.sort()
            p.reverse()
        ps = [[a for (b,a) in x] for x in ps]

        return ps

def GaleShapley(pref, ps):
    """
    Performs crappy matching.

    Parameters
    ----------
    ps : list(list(int))
        A list, containing lists of integers.
        The outer list represents the vertices in X. The inner lists represent vertices in Y.
        The vertices in the inner lists are ordered according to the preferences of X.

    Returns
    -------
    dict ( int -> int )
        Returns a dict with the matching. The keys are set from Y to X.
    """

    def get_pref(k, l):
        if k in pref and l in pref[k]:
            return pref[k][l]
        return 0

    xs = list(range(len(ps)))
    engaged = {}

    max_len = len(xs)

    with progressbar.ProgressBar(max_value=max_len) as bar:
        while xs:
            x = xs.pop(0)
            if ps[x]:
                y = ps[x][0]
                if y in engaged:
                    xn = engaged[y]
                    if get_pref(x, y) > get_pref(xn, y):
                        xs.append(xn)
                        engaged[y] = x
                        ps[x].pop(0)
                else:
                    engaged[y] = x
            bar.update(max_len-len(xs))

    ns = set(range(len(ps)))

    exs = ns - set(engaged.values())
    eys = ns - set(engaged.keys())

    for x,y in zip(exs, eys):
        engaged[y] = x

    return engaged

