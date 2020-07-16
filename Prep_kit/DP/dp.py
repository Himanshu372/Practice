
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
    c = LongestRepeatingSubsequence('abba')
    print(c.lrs_dp())