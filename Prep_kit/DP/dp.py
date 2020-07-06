
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



if __name__ == '__main__':
    k = MaxArraySumNonAdjacent([-2, 1, 3, -4, 5])
    print(k.max_array_sum_non_adjacent())

