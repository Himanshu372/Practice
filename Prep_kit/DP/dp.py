
class MaxArraySumNonAdjacent():
    """

    """
    def __init__(self, array):
        self._array = array

    def max_array_sum_non_adjacent(self):
        incl = self._array[0]
        excl = 0

        for i in range(1, len(self._array)):
            new_excl = incl if incl > excl else excl
            incl = excl + self._array[i]
            excl = new_excl
        return max(incl, excl)

def abbreviation(a, b):
    if len(a) < len(b):
        return False
    if len(b) == 0:
        return a.islower()
    if a.upper() == b:
        return True
    if a[0].isupper():
        if a[0] == b[0]:
            return abbreviation(a[1:], b[1:])
        else:
            return False
    else:
        if a[0].upper() == b[0]:
            return abbreviation(a[1:], b[1:])
        else:
            return (abbreviation(a[1:], b[1:]) or abbreviation(a[1:], b))

class Minrewardsarray():

    def __init__(self, array):
        self._array = array
        self._rewards = [1 for _ in range(len(self._array))]

    def min_rewards(self):
        l = len(self._array)
        left, right, num = [[0 for _ in range(l)] for _ in range(3)]
        pointer = 1
        while pointer < l:
            if self._array[pointer] > self._array[pointer - 1]:
                left[pointer] = left[pointer - 1] + 1
            pointer += 1
        pointer = l - 2
        while pointer >= 0:
            if self._array[pointer] > self._array[pointer + 1]:
                right[pointer] = right[pointer + 1] + 1
            pointer -= 1
        pointer = 0
        while pointer < l:
            num[pointer] = 1 + max(left[pointer], right[pointer])
            pointer += 1
        return sum(num)

class LongestCommonPalindromicSubsequence:
    """
    A variation of LCS problem, here our aim is to find longest common palindromic subsequence
    Idea is to create a grid with same string in row and column, then compare substring of various lenghts
    For grid, all the diagnol elements will be filled as 1, as if the string just had a single string, lcps = 1
    The required algo is
    if string[i] == string[j]:
        grid[i][j] = 2 + diagonally opp lower element
    else:
        grid[i][j] = max(element to left, element below)
    return grid[0][lenght of string - 1]
    ** Important to note is the iteration loop
    """
    def __init__(self, s):
        self._string = s
        self._len = len(s)

    def lcps(self):
        grid = [[0 for _ in range(self._len)] for _ in range(self._len)]
        for diag in range(self._len):
            grid[diag][diag] = 1
        for curr_len in range(2, self._len + 1):
            for i in range(self._len - curr_len + 1):
                j = i + curr_len - 1
                if self._string[i] == self._string[j]:
                    grid[i][j] = 2 + grid[i + 1][j - 1]
                else:
                    grid[i][j] = max(grid[i][j - 1], grid[i + 1][j])
        return grid[0][self._len - 1]

class EditDistance:
    """
    Class implementating solution for minimum steps required to convert one string to another
    Distance known as Levenshtein distance.
    Solved by Recursion and DP.
    Main idea is
    if string[row] == string[col]:
        grid[row][col] = grid[row - 1][col - 1] (Taking the element diagonally above)
    else:
        grid[row][col] = 1 + min(grid[row - 1][col], grid[row][col - 1], grid[row - 1][col - 1])
    Same idea applies to recursion solution
    """
    def edit_distance_recursive(self, convert, convert_to):
        if len(convert) == 0 and len(convert_to) == 0:
            return len(convert)
        if len(convert) == 0:
            return len(convert_to)
        if len(convert_to) == 0:
            return len(convert)
        if convert[0] == convert_to[0]:
            return self.edit_distance_recursive(convert[1:], convert_to[1:])
        else:
            return 1 + min(self.edit_distance_recursive(convert[1:], convert_to),
                        self.edit_distance_recursive(convert, convert_to[1:]),
                        self.edit_distance_recursive(convert[1:], convert_to[1:]))

    def edit_distance_dp(self, convert, convert_to):
        m, n = len(convert), len(convert_to)
        grid = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
        grid[0] = [i for i in range(n + 1)]
        for j in range(m + 1):
            grid[j][0] = j
        for row in range(1, m + 1):
            for col in range(1, n + 1):
                if convert[row - 1] == convert_to[col - 1]:
                    grid[row][col] = grid[row - 1][col - 1]
                else:
                    grid[row][col] = 1 + min(grid[row - 1][col], grid[row][col - 1], grid[row - 1][col - 1])
        return grid[m][n]


class LongestRepeatingSubsequence:
    """
    A variation of LCS problem. Here the main idea is to find,
    compare a string with its substring and find any recurrence of the same substring.
    `If the character match, we pick element diagonally above and add 1
    otherwise we pick the max from the element at top or left`
    """
    def __init__(self, string):
        self._s = string
        self._len = len(string)
        self._grid = [[0 for _ in range(self._len + 1)] for _ in range(self._len + 1)]

    def lrs_recursive(self, m, n):
        if m == 0 or n == 0:
            return 0
        if self._s[m - 1] == self._s[n - 1] and m != n:
            return 1 + self.lrs_recursive(m - 1, n - 1)
        return max(self.lrs_recursive(m, n - 1), self.lrs_recursive(m - 1, n))

    def lrs_dp(self):
        for row in range(1, self._len + 1):
            for col in range(1, self._len + 1):
                if self._s[row - 1] == self._s[col - 1] and row != col:
                    self._grid[row][col] = 1 + self._grid[row - 1][col - 1]
                else:
                    self._grid[row][col] = max(self._grid[row - 1][col], self._grid[row][col - 1])
        return self._grid[self._len][self._len]


class DistinctSubsequences:
    """
    This class handles the problem of distinct subsequences.
    A variation to LCS, here
    1. if the elements match, i.e add up diagonally above element and element to left
        else take the element to the left
        if string[col - 1] == subseq[row - 1]:
            grid[row][col] = grid[row - 1][col - 1] + grid[row][col - 1]
        else:
            grid[row][col] = grid[row][col - 1]
    """
    def __init__(self, string, subseq):
        self._string = string
        self._subseq = subseq
        self._grid = [[0 for _ in range(len(self._string) + 1)] for _ in range(len(self._subseq) + 1)]
        self._mem = {}

    def _init_fill_grid(self):
        for i in range(len(self._string) + 1):
            self._grid[0][i] = 1

    def distinct_subseq_dp(self):
        self._init_fill_grid()
        for row in range(1, len(self._subseq) + 1):
            for col in range(1, len(self._string) + 1):
                if self._subseq[row - 1] == self._string[col - 1]:
                    self._grid[row][col] = self._grid[row][col - 1] + self._grid[row - 1][col - 1]
                else:
                    self._grid[row][col] = self._grid[row][col - 1]
        return self._grid[len(self._subseq)][len(self._string)]

class InterLeavingString:
    """
    Class deals with solution of checking if given 2 strings can make up the given 3rd string.


    """
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
        
    def is_interleave(self):
        if len(self.a) + len(self.b) != len(self.c):
            return 0
        return self.is_interleave_recursive(self.a, self.b, self.c, 0, 0, 0)

    def is_interleave_recursive(self, a, b, c, i, j, k):
        if i > len(a):
            return 0
        if j > len(b):
            return 0
        if k == len(c):
            return 1
        #
        if i == len(a):
            if b[j] == c[k]:
                return self.is_interleave_recursive(a, b, c, i, j + 1, k + 1)
            return 0
        if j == len(b):
            if a[i] == c[k]:
                return self.is_interleave_recursive(a, b, c, i + 1, j, k + 1)
            return 0
        if a[i] == b[j]:
            if a[i] == c[k]:
                return self.is_interleave_recursive(a, b, c, i + 1, j, k + 1) or \
                       self.is_interleave_recursive(a, b, c, i, j + 1, k + 1)
            return 0
        if a[i] == c[k]:
            return self.is_interleave_recursive(a, b, c, i + 1, j, k + 1)

        if b[j] == c[k]:
            return self.is_interleave_recursive(a, b, c, i, j + 1, k + 1)
        return 0

    def is_interleave_dp(self):
        grid = [[0 for _ in range(len(self.a) + 1)] for _ in range(len(self.b) + 1)]
        grid[0][0] = 1
        for col in range(1, len(self.a)):
            if self.a[col - 1] == self.c[col - 1]:
                grid[0][col] = grid[0][col - 1]
        for row in range(1, len(self.b)):
            if self.b[row - 1] == self.c[row - 1]:
                grid[row][0] = grid[row - 1][0]
        for row in range(1, len(self.a) + 1):
            for col in range(1, len(self.b) + 1):
                if self.b[row - 1] == self.c[(row + col) - 1]:
                    grid[row][col] = grid[row - 1][col]
                elif self.a[col - 1] == self.c[(row + col) - 1]:
                    grid[row][col] = grid[row][col - 1]
        return grid[len(self.a)][len(self.b)]


import re
def regex_match(a, b):
    if not any([a.find('?') > 0, a.find('*') > 0]) and not any([b.find('?') > 0, b.find('*') > 0]):
        if a != b:
            return 0
        else:
            return 1
    t = {'?': '.?', '*': '.*'}
    for key in t:
        b = b.replace(key, t[key])
    res = re.match(r'{}'.format(b), a)
    if res:
        return 1 if res.group() == a else 0
    else:
        return 0


class RegexMatch:
    """

    """
    def __init__(self, string, pattern):
        self.string = string
        self.pattern = pattern
        self.grid = [[False for _ in range(len(pattern) + 1)] for _ in range(len(string) + 1)]
        self.grid[0][0] = True

    def find_match_dp(self):
        for col in range(1, len(self.pattern)):
            self.grid[0][col + 1] = self.grid[0][col - 1] and self.pattern[col] == '*'
        for row in range(1, len(self.string) + 1):
            for col in range(1, len(self.pattern) + 1):
                if self.string[row - 1] == self.pattern[col - 1] or self.pattern[col - 1] == '.':
                    self.grid[row][col] = self.grid[row - 1][col - 1]
                elif self.pattern[col - 1] == '*':
                    if (col - 2) >= 0 and self.grid[row][col - 2] is True:
                        self.grid[row][col] = True
                    elif (col - 2) >= 0 and self.string[row - 1] == self.pattern[col - 2]:
                        self.grid[row][col] = self.grid[row - 1][col]
        return self.grid[len(self.string)][len(self.pattern)]

class CheckScrambledString:
    """
    Class represents solution of the problem to check if given other string(scrambled) is scrambled version of first one.
    For eg. string = 'great', scrambled = 'rgate' => Answer is yes
    String can be represented as binary tree, and by swapping its non leaf node, we can arrive at number of scrambled strings
            great
            / \
          gr   eat
         /\    /\
        g  r  e at
                /\
               a  t
        Swapping 'gr' and 'eat', we get to 'rgate'
    """
    def __init__(self, string, scrambled):
        self.string = string
        self.scrambled = scrambled
        self.mem = {}

    def check_scramble(self, s1=None, s2=None):
        s1, s2 = (self.string, self.scrambled) if (s1 is None and s2 is None) else (s1, s2)
        if len(s1) != len(s2):
            return 0
        if len(s1) <= 1:
            return 1 if s1 == s2 else 0
        if res := self.mem.get((s1, s2), 0):
            return res
        for i in range(1, len(s1)):
            if (self.check_scramble(s1[:i], s2[:i]) and self.check_scramble(s1[i:], s2[i:])) or \
                    (self.check_scramble(s1[:i], s2[-i:]) and self.check_scramble(s1[i:], s2[:-i])):
                self.mem[(s1, s2)] = 1
                return 1
        self.mem[(s1, s2)] = 0
        return 0



if __name__ == '__main__':
    # k = MaxArraySumNonAdjacent([-2, 1, 3, -4, 5])
    # print(k.max_array_sum_non_adjacent())
    # print(abbreviation('K', 'KXzQ'))
    # c = Minrewardsarray([2,4,2,6,1,7,8,9,2,1])
    # print(c.min_rewards())
    # k = LongestCommonPalindromicSubsequence('bebeeed')
    # print(k.lcps())
    # o = EditDistance()
    # print(o.edit_distance_dp('Anshuman', 'Antihuman'))
    # c = LongestRepeatingSubsequence('abba')
    # print(c.lrs_dp())
    # s = DistinctSubsequences('rabbbit', 'rabbit')
    # print(s.distinct_subseq_dp())
    # s = InterLeavingString('aab', 'axy', 'aaxaby')
    # print(s.is_interleave())
    # print(s.is_interleave_dp())
    s = CheckScrambledString('abb', 'bba')
    print(s.check_scramble())
