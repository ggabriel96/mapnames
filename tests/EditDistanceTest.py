import string
import unittest as ut

from mapnames import string


class EditDistanceTest(ut.TestCase):

    def test_identity(self):
        for i in range(len(string.digits)):
            id = string.digits[:i]
            self.assertEqual(string.wagner_fischer(id, id), 0)

    def test_one_empty(self):
        for i in range(len(string.digits)):
            s = string.digits[:i]
            self.assertEqual(string.wagner_fischer('', s), i)
            self.assertEqual(string.wagner_fischer(s, ''), i)

    def test_ascending(self):
        for i in range(len(string.digits)):
            s1 = string.digits[:i]
            for j in range(len(string.digits)):
                s2 = string.digits[:j]
                self.assertEqual(string.wagner_fischer(s1, s2), abs(i - j))

    def test_misc(self):
        self.assertEqual(string.wagner_fischer('abc', 'b'), 2)
        self.assertEqual(string.wagner_fischer('abc', 'b'), 2)
        self.assertEqual(string.wagner_fischer('abc', 'ac'), 1)
        self.assertEqual(string.wagner_fischer('abc', 'bc'), 1)
        self.assertEqual(string.wagner_fischer('abcdefg', 'abce'), 3)
        # Different from last case because of symmetric 'd'
        self.assertEqual(string.wagner_fischer('abcdefg', 'gfedcba'), 6)
        # '0123456789' and '9876543210'
        self.assertEqual(
            string.wagner_fischer(string.digits, string.digits[::-1]),
            len(string.digits))


if __name__ == '__main__':
    ut.main()
