
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





if __name__ == '__main__':
    # k = MaxArraySumNonAdjacent([-2, 1, 3, -4, 5])
    # print(k.max_array_sum_non_adjacent())
    # print(abbreviation('K', 'KXzQ'))
    c = Minrewardsarray([2,4,2,6,1,7,8,9,2,1])
    print(c.min_rewards())