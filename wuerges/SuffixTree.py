"""
This module can be used to create a Generalized Suffix Tree.
"""

class N:
    """
    N is a Node in a Suffix Tree.
    """
    def __init__(self, parent, word, p, c, wid):
        """
        Creates a Node to be added into a tree.

        By default, a node is created as a leaf node (not real).
        If it is a leaf, represents one suffix.
        If not, represents a character.

        Parameters
        ----------
        parent : N
            The node that will be the parent of self in the tree.
        word : str
            A reference to the string that has the suffix that will be represented by self.
        p : int
            The index of the first character of self.
        c : chr
            The character that can be used to travel from parent to self using child.
        wid : int
            An id for the word, that will be the same for all its suffixes.
        """
        self.p = p
        self.c = c
        self.word = word
        self.link = None
        self.parent = parent
        self.child = {}
        self.real = False
        self.wid = wid
        self.size = -1

    def materialize(self):
        """
        Materilizes self.

        Nodes are first created lazy, and represent a whole suffix.
        When a node is materialized, it will represent only one character.
        The remaining of the string will be represented by its child.
        """
        if self.real:
            return
        self.real = True
        if self.p < len(self.word):
            self.child[self.word[self.p]] = N(self, self.word, self.p+1,self.word[self.p], self.wid)

    def get_link(self):
        """
        Finds the suffix link of self.

        A suffix link of self is the node of the tree that represents the same spot
        in the tree that self represents, for the next suffix (the suffix that is one character shorter).
        """
        if self.link:
            return self.link
        if not self.parent:
            self.link = self
            return self
        if not self.parent.parent:
            self.link = self.parent
            return self.parent
        self.link = self.parent.get_link().child[self.c]
        self.link.materialize()
        return self.link

    def add_suffix_it(self, new_word, new_word_id):
        """
        Adds a word and all its suffixes to the tree.

        Parameters
        ----------
        new_word : str
            The string that will be added to the Tree.
        new_word_id : int
            An integer, that identifies the suffix of word.
            It will be the same for every leaf of the tree that represents a suffix of new_word.
        """
        i = 0
        x = self
        while i < len(new_word):
            x.materialize()
            ci = new_word[i]

            if ci in x.child:
                x = x.child[ci]
                i += 1
            else:
                x.child[ci] = N(x, new_word, i+1, ci, new_word_id)
                if x.parent:
                    x = x.get_link()
                else:
                    i += 1

    def search(self, term):
        """
        Performs a search in the tree.

        Parameters
        ----------
        term : str
            The term to be searched in the tree.

        Returns
        -------
        position, word, word_id, size, length : tuple

        position : int
            The index of the beginning of the term.
        word : str
            The string that was used to build the node.
        word_id : int
            The word_id set by add_suffix_it.
        size : int
            The number of leaves under this node.
            The number of matches of the string.
        length : int
            The lenght of the match.
        """
        i = 0
        x = self
        while i < len(term) and x.real:
            #print("searching ", term[:i+1])
            if term[i] in x.child:
                x = x.child[term[i]]
                i += 1
            else:
                break
        if not x.real:
            j = x.p
            #print("xxx", i, term[i:], x.word[x.p:])
            while i < len(term) and j < len(x.word) and term[i] == x.word[j]:
                i += 1
                j += 1
        return x.p, x.word, x.wid, x.size, i

    def calcSize(self):
        """
        Calculates the number of leaves under self and stores it in size.
        """
        if not self.real:
            self.size = 1
        else:
            r = 0
            for k,v in self.child.items():
                v.calcSize()
                r += v.size
            self.size = r

    def dfs(self, l):
        """
        Performs a dfs in self.
        Stores every (p, word, word_id) in l.

        Parameters
        ----------
        l : list
            A list that will hold the results of the dfs.
        """
        if not self.real:
            l.append((self.p, self.word, self.wid))
        else:
            for k,v in self.child.items():
                v.dfs(l)


def CreateTree(data, suf="#$!"):
    """
    Creates a Generalized Suffix Tree from a list of strings.

    Parameters
    ----------
    data : list(str)
        A list of strings.
    suf : str
        An unique terminator that will be appended to each string.

    Returns
    -------
    N
        A Generalized Suffix Tree, containing all the suffixes
        of every word in data.
        The index of the string of each word in data will
        be used as word_id within the tree.

    """
    t = N(None, "", 0, None, 0)
    t.real = True
    import progressbar

    i = 0
    for s in progressbar.progressbar(data):
        t.add_suffix_it(s + suf + str(i), i)
        i+= 1

    t.calcSize()
    return t


