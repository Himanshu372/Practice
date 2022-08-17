def max_subarray_sum_kadane(array, brute_force):
    """
       Implementation of Kadane's algo for an array, idea is to find at each index,
        maximum sum of contiguous subarray, with adding the element at the index with the
        sum of trailing subarray
        :param array:
        :return:
        """
    if brute_force:
        max_sum = array[0]
        for i in range(len(array)):
            for j in range(i + 1, len(array)):
                current_array = array[i:j + 1]
                max_sum = max(max_sum, sum(current_array))
        return max_sum
    max_sum = current_sum = array[0]
    for i in range(1, len(array)):
        current_sum = max(array[i], current_sum + array[i])
        if max_sum < current_sum:
            max_sum = current_sum
    return max_sum


if __name__ == "__main__":
    print(
        f"{max_subarray_sum_kadane([-2, -3, 4, -1, -2, 1, 5, -3], True)} result of brute force"
    )
    print(
        f"{max_subarray_sum_kadane([-2, -3, 4, -1, -2, 1, 5, -3], False)} result of optimal algo"
    )
