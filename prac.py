import re
import sys
import json

def revrot(strng, sz):
    if sz <= 0 or len(strng) == 0:
        return ""
    chunks = len(strng)//sz
    ret_str = ""
    for chunk in range(chunks):
        substr = strng[chunk * sz:(chunk + 1) * sz]
        if find_cube(int(substr)):
            substr = substr[::-1]
        ret_str += substr
        
    return ret_str

def find_cube(k):
    sum = 0 
    while k != 0:
        n = k % 10 
        sum += pow(n, 3)
        k = k // 10
    return True if sum % 2 == 0 else False


def to_underscore(s):
    if str(s).isdigit():
        return '{}'.format(s)
    parts = [i for i in re.split('([A-Z])', s) if len(i) > 0]
    counter = 0
    s = []
    k = 0
    while(counter < len(parts)):
        if parts[counter].isupper():
            s.append(parts[counter].lower())
            k += 1
        else:
            s[k - 1] += parts[counter]
        counter += 1
    return '_'.join(s)


def parse_molecule(formula):
    l = re.findall(r'[A-Z][a-z]*', formula)
    d = {key: l.count(key) for key in l}
    for key in d.keys():
        adjoining = re.findall(r'{}([0-9])'.format(key), formula)
        square_brac = re.findall(r'\[.*{}.*\]([0-9])'.format(key), formula)
        circular_brac = re.findall(r'\(.*{}.*\)([0-9])'.format(key), formula)
        if adjoining:
            d[key] *= int(adjoining[0])
        if square_brac:
            d[key] *= int(square_brac[0])
        if circular_brac:
            d[key] *= int(circular_brac[0])
    return d


def return_angle(s):
    hour, minute = [int(i) for i in s.split(':')]
    hour = (hour-12) if hour > 12 else hour
    hour_hand_angle = hour * 30 if hour != 12 else 0
    minute_hand_angle = minute * 6 if minute != 60 else 0
    hour_hand_angle = (360 - hour_hand_angle) if hour_hand_angle > 180 else hour_hand_angle
    minute_hand_angle = (360 - minute_hand_angle) if minute_hand_angle > 180 else minute_hand_angle
    angle = abs(hour_hand_angle - minute_hand_angle)
    return angle

def get_max_profit(array, k):
    """

    :param array:
    :param k:
    :return:
    """
    total_profit = k * (max(array) - min(array))
    # for i in range(k):
    #     current_min = min(array)
    #     current_max = max(array)
    #     total_profit += current_max - current_min
    #     array.remove(current_min)
    #     array.remove(current_max)
    return total_profit

def longest_special_subseq(n,k,s):
    longest_len = 0
    #Code here
    a = [ord(i) for i in s]
    max_len = 0
    for i in range(1, len(a)):
        current_len = 0
        for j in range(i):
            if abs(a[j] - a[i]) <= k:
                current_len += 1
        if current_len > max_len:
            max_len = current_len
    return max_len


def Palindromic_Subsequence(s):
    # Your code goes here
    l = []
    if len(s) == 1:
        return s
    for i in range(2, len(s)):
        s = ''
        for j in range(i):
            temp_s = s[j:i]
            if temp_s == temp_s[::-1]:
                s += temp_s
        l.append(s)
    return sorted(l)[0] if len(l) > 0 else -1


def get_pairwise_product(l):
    min_prod = sys.maxsize
    for i in range(1, len(l)):
        for j in range(i):
            current_prod = l[j] * l[i]
            if current_prod < min_prod:
                min_prod = current_prod
    return min_prod


import ast

def get_inner(nested):
    for i in range(len(nested)):
        while isinstance(nested[i], list):
            get_inner(nested[i])
        




def evaluate_expression(cir):
    cir = cir.replace('!', '"!"')
    cir = cir.replace('&', '"&"')
    cir = cir.replace('|', '"|"')
    cir = ast.literal_eval(cir)
    for i in get_inner(cir):
        print(i)

import os
import re

def open_file(fs):
    with open(fs) as text_file:
        counter = 0
        obj_list = []
        for line in text_file:
            # next_line = next(text_file)
            # while True:
            #     if not re.match(r'.*/\w+:.*', next_line) and len(next_line) > 0:
            #         line = ' ' + next_line
            #     else:
            #         text_file.__next__ = next_line
            #         break
            if re.match(r'.*/\w+:.*', line) and counter < 8:
                obj_list.append(re.split(':', line.strip(), 1)[1])
                counter += 1
            else:
                if counter == 8:
                    print(obj_list)
                    obj_list = []
                    counter = 0
                else:
                    pass


def print_pattern(n):
    for i in range(n):
        for x in range(i):
            print(n - x, end='')
        for y in range((2*n - 1) - 2*i):
            print(n - i, end='')
        for z in range((i - 1), -1, -1):
            print(abs(z - n), end='')
        print('\n')
    for i in range(n, 1, -1):
        for x in range(i - 2):
            print(n - x, end='')
        for y in range((2*n - 1) - 2*(i - 2)):
            print(n - i + 2, end='')
        for z in range(i - 2, 0, -1):
            print(abs(z - n - 1), end='')
        print('\n')

def print_odd_even(n):
    for i in range(1, n + 1, 2):
        for j in range(i*n - n + 1, i*n + 1):
            print(j, end=" ")
        print('\n')
    if n % 2 != 0:
        k = n - 1
    else:
        k = n
    for i in range(k, 1, -2):
        for j in range(i*n - n + 1, i*n + 1):
            print(j, end=" ")
        print('\n')

def tasks(n, a, b):
    d = {k: v for k, v in zip(a, b)}
    cyclic_tasks = 0
    accounted = []
    for k, v in d.items():
        counter = 0
        temp_accounted = []
        while v in d.keys():
            counter += 1
            if (v, d[v]) not in accounted:
                temp_accounted.append((v, d[v]))
            if d[v] == k:
                if not ((k, v) in accounted or (v, k) in accounted):
                    cyclic_tasks += counter
                    accounted.extend(temp_accounted)
                break
            if counter == n:
                break
            v = d[v]
    return n - cyclic_tasks


def search(grid, r, c, row_len, col_len, word):
    dir = [[-1, 0], [1, 0], [0, -1], [0, 1], [1, 1], [1, -1], [-1, 1], [-1, -1]]
    if grid[r][c] != word[0]:
        return False
    for x, y in dir:
        row, col = r + x, c + y
        flag = True
        for ind in range(1, len(word)):
            if (0 <= row < row_len) and (0 <= col < col_len) and (grid[row][col] == word[ind]):
                row += x
                col += y
            else:
                flag = False
                break
        if flag:
            return ' '.join([word, str(r), str(c)])
    return False

def find_pattern(grid, word):
    row_len = len(grid)
    col_len = len(grid[0])
    for i in range(row_len):
        for j in range(col_len):
            result = search(grid, i, j, row_len, col_len, word)
            if result:
                print(result)
                return
            else:
                continue
    else:
        print(word, '-1', '-1')


class Node:
    def __init__(self, info):
        self.info = info
        self.left = None
        self.right = None
        self.level = None


class BinarySearchTree:
    def __init__(self):
        self.root = None

    def __repr__(self):
        self.inorder_traversal()

    def inorder_traversal(self, node=None):
        curr_node = node if node else self.root
        if curr_node.left:
            self.inorder_traversal(curr_node.left)
        print(str(curr_node.info))
        if curr_node.right:
            self.inorder_traversal(curr_node.right)



    def create(self, val):
        if self.root is None:
            self.root = Node(val)
        else:
            current = self.root

            while True:
                if val < current.info:
                    if current.left:
                        current = current.left
                    else:
                        current.left = Node(val)
                        break
                elif val > current.info:
                    if current.right:
                        current = current.right
                    else:
                        current.right = Node(val)
                        break
                else:
                    break




def find_alignment(val, node):
    if node.info > val:
        parent = node
        curr_node = node.left
        while curr_node and curr_node.info != val:
            parent = curr_node
            curr_node = curr_node.left



def lca(root, v1, v2):
   curr_node = root
   if curr_node is None:
       return None
   elif curr_node.info > v1 and curr_node.info > v2:
       return lca(node.left, v1, v2)
   elif curr_node.info < v1 and curr_node.info < v2:
       return lca(node.right, v1, v2)
   else:
       return node


def taxiDriver(p, d, t):
    amt = []
    for i in range(len(p)):
        drop = d[i]
        total = 0
        total += (d[i] - p[i]) + t[i]
        j = i
        while j < (len(p) - 1):
            j += 1
            if p[j] >= drop:
                total += (d[j] - p[j]) + t[j]
                drop = d[j]
            else:
                pass
        amt.append(total)
    return max(amt)


def taxiDriveralternate(p, d, t):
    amt = []
    r = {}
    for i in range(len(p) - 1, 0, -1):
        total = 0
        r['{}-{}'.format(str(p[i]), str(d[i]))] = (d[i] - p[i]) + t[i]
        j = i
        while j < (len(p) - 1):
            if d[j] <= p[j + 1]:
                if r.get('{}-{}'.format(str(p[j]), str(d[j])), None):
                    total += (d[j] - p[j]) + t[i]
                total += (d[j] - p[j]) + t[j]
                drop = d[j]
            else:
                pass
        amt.append(total)
    return max(amt)

def split_array(arr):
    l = {}
    r = {}
    count = 0
    l['sum'] = arr[0]
    r['sum'] = sum(arr[1:])
    count += 1 if (l['sum'] > r['sum']) else 0
    for i in range(1, len(arr) - 1):
        l['sum'] += arr[i]
        r['sum'] -= arr[i]
        count = (count + 1) if (l['sum'] > r['sum']) else count
    print(count)


def threepalinstring(word):
    r = []
    for i in range(2, len(word) // 2):
        for j in range(0, len(word) // 2, i):
            if word[j: j + i] == word[j: j + i: -1]:
                if len(r) > 3:
                    return r
                else:
                    r.append()
    else:
        return ['Impossible']


def checkBST(root):
    """
    Check if the given tree is BST
    """
    return check_binary_tree(root, -1, 10001)


def check_binary_tree(root, min_val, max_val):
    """
    Starting from root, idea is to check wheather left and right sub-tree return True
    """
    if root is None:
        return True
    data = root.data
    # Check this step used for recursion
    if min_val < data < max_val:
        return check_binary_tree(root.left, min_val, data) and check_binary_tree(root.right, data, max_val)
    else:
        return False

def decodeHuff(root, s):
    """
    Huffman encoding-decoding. Encoding involves assiging smallest code for character with highest count
    and larger code for character with smallest count in a string. For eg.
    'ABACA', encoding is '1001011'. For string, char_counts are 'A'=3,['B','c']=1. A tree is created for while encoding,
    starting with lowest count characters, ie 'B' and 'C'  
    """
    string = decode_decode_Huff(root, s)
    print(string)

def decode_decode_Huff(root, s):
    if len(s) == 0:
        return ''
    start = 0
    curr_node = root
    while True:
        if s[start] == '1':
            if is_leaf(curr_node.right):
                return curr_node.right.data + decode_decode_Huff(root, s[start + 1:])
            else:
                curr_node = curr_node.right
        elif s[start] == '0':
            if is_leaf(curr_node.left):
                return curr_node.left.data + decode_decode_Huff(root, s[start + 1:])
            else:
                curr_node = curr_node.left
        start += 1

def is_leaf(node):
    return not node.left and not node.right


def add_reverse(a, b):
    diff = abs(len(a) - len(b))
    if len(a) > len(b):
        b = [0] * diff + b
    else:
        a = [0] * diff + a
    a = a[::-1]
    b = b[::-1]
    carry = 0
    result = []
    for i, j in zip(a[:-1], b[:-1]):
        addition = i + j + carry
        carry = addition // 10
        addition = addition % 10
        result.append(addition)
    last_addition = a[-1] + b[-1] + carry
    last_addition = list(map(int, list(str(last_addition))))
    result = result + last_addition[::-1]
    print(result[::-1])

def add_reverse_simple(a, b):
    addition = int(''.join(map(str, a))) + int(''.join(map(str, b)))
    return list(map(int, list(str(addition))))


def getMinimumCost(k, c):
    sum = 0
    c = sorted(c, reverse=True)
    for _ in range(k):
        sum += c.pop(0)
    bought = 1
    counter = k
    while len(c) != 0:
        if counter == 0:
            bought += 1
            counter = k
        sum += c.pop(0) * (1 + bought)
        counter -= 1
    return sum


from collections import defaultdict

def reverseshufflemerge(S):
    S = S[::-1]
    count = defaultdict(int)
    for c in S:
        count[c] += 1
    need = {}
    for c in count:
        need[c] = count[c] / 2
    solution = []
    i = 0
    while len(solution) < len(S) / 2:
        min_char_at = -1
        while True:
            c = S[i]
            if need[c] > 0 and (min_char_at < 0 or c < S[min_char_at]):
                min_char_at = i
            count[c] -= 1
            if count[c] < need[c]:
                break
            i += 1
        for j in range(min_char_at+1, i+1):
            count[S[j]] += 1
        need[S[min_char_at]] -= 1
        solution.append(S[min_char_at])
        i = min_char_at + 1
    print(''.join(solution))

def commonChild(s1, s2):
    """
    Recursive non memoized solution for longest subsequence problem between 2 strings
    """
    if len(s1) == 0 or len(s2) == 0:
        return 0
    elif s1[-1] == s2[-1]:
        return 1 + commonChild(s1[:-1], s2[:-1])
    else:
        return max(commonChild(s1, s2[:-1]), commonChild(s1[:-1], s2))

def commonChild_memoized(s1, s2):
    """
    Memoized solution for longest subsequence problem
    1. We create a grid of zeros, using the (length + 1) of s1 and s2(first position for both taken by empty string)
    2. We iterate through each row,
        a. When both elements are equal, we add 1 to the element of the grid, which is diagonally above this
         particular element
        b. If not equal, we take max from previous row and column
    3. We fill the whole grid by above rule and the end return the last element of the grid, ie grid[-1][-1]

    ** To make solution more efficient in Python, rather than using grid
    """
    m, n = len(s1), len(s2)
    # grid = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    prev = [0 for _ in range(n + 1)]
    for row in range(1, n + 1):
        curr = [0 for _ in range(n + 1)]
        for col in range(1, m + 1):
            if s1[col - 1] == s2[row - 1]:
                curr[col] = 1 + prev[col - 1]
            else:
                curr[col] = max(prev[col], curr[col - 1])
        prev = curr
    return curr[-1]



class MinPathSum():

    def __init__(self, array):
        self._array = array
        self._len = len(array)

    def min_path_sum(self):
        """
        This function is used for solving problem of minimum path sum in a 2D array. Goal is to get to the bottom right
        corner, starting from top left with minimum sum of array elements. You can only move right or down at each step.
        [[1, 3, 2],
        [4, 3, 1],
        [5, 6, 1]]
        Ans is 8
        """
        row, col = len(self._array) - 1, len(self._array[0]) - 1
        result = 0
        while row > -1 and col > -1:
            result += self._array[row][col]
            if row == 0:
                col -= 1
            elif col == 0:
                row -= 1
            else:
                top, left = self._array[row - 1][col], self._array[row][col - 1]
                if top < left:
                    row -= 1
                else:
                    col -= 1
        return result

    def min_path_sum_dp(self):
        """
        The same above problem solved using dp array
        """
        rows, cols = len(self._array), len(self._array[0])
        dp = [[0 for col in range(cols)] for row in range(rows)]
        for row in range(rows):
            for col in range(cols):
                if row == 0 and col == 0:
                    dp[row][col] = self._array[row][col]
                else:
                    top = sys.maxsize if row == 0 else dp[row - 1][col]
                    left = sys.maxsize if col == 0 else dp[row][col - 1]
                    dp[row][col] = self._array[row][col] + min(top, left)
        return dp[rows - 1][cols - 1]


class CombinationSum():
    """
    This class is used as solution to the problem of finding the number of sub-arrays that sum up to given 'total',
    for eg. array = [2, 4, 6, 10] and total = 16
            All the subarrays that sum up to 16 are [6, 10] and [2, 4, 10]
    Solved using recursion and dp recursion.
    Idea is to build a recursive function where we, for eg we are considering element 10
    sum(all the subarrays that sum to 16 including 10, all the subarrays that sum to 16 without 10)
    We recursively go down until we reduce total to 0(dp[0] = 1) and calculate number required for each number up until
    we get to the required total.
    """
    def __init__(self, array, total):
        self._array = array
        self._total = total
        self._mem = {}

    def comb_sum(self, flag):
        return self.recursion(len(self._array) - 1, self._total) if flag else self.dp_recursion(len(self._array) - 1,
                                                                                                self._total)

    def recursion(self, i, total):
        if total == 0:
            return 1
        elif total < 0:
            return 0
        elif i < 0:
            return 0
        elif total < self._array[i]:
            return self.recursion(i - 1, total)
        else:
            return self.recursion(i - 1, total - self._array[i]) + self.recursion(i - 1, total)

    def dp_recursion(self, i, total):
        key = '{}:{}'.format(total, self._array[i])
        res = self._mem.get(key, None)
        if not res:
            if total == 0:
                return 1
            elif total < 0:
                return 0
            elif i < 0:
                return 0
            elif total < self._array[i]:
                res = self.dp_recursion(i - 1, total)
            else:
                res = self.dp_recursion(i - 1, total - self._array[i]) + self.dp_recursion(i - 1, total)
        return res


class MinCoins():
    """
    This class acts as solution for min coins problem. We need to find min coins that make up the given `target` from
    given denomination(array) of coins. Once again the idea is same, we calculate the min coins required for each number,
    until we get to required target.
    We start from dp[0] = 0,
    if coin of denomination 1 is present, dp[1] = 1
    if coin of denomination 2 is present, dp[2] = min(dp[2], dp[2 - 1] + 1(as we will be using coin))
    """
    def __init__(self, array, target):
        self._array = array
        self._target = target
        self._dp = [sys.maxsize for _ in range(self._target + 1)]
        self._dp[0] = 0

    def min_coins_dp(self):
        for i in range(1, self._target + 1):
            for num in self._array:
                if i - num < 0:
                    break
                else:
                    self._dp[i] = min(self._dp[i], self._dp[i - num] + 1)
        return -1 if self._dp[self._target] >= sys.maxsize else self._dp[self._target]


if __name__=='__main__':
    # print_pattern(4)
    # print_odd_even(3)
    # print(getMinimumCost(3, [2, 5, 6]))
    # reverseshufflemerge('abcdefgabcdefg')
    # print(commonChild_memoized('SHINCHAN', 'NOHARAAA'))
    # print(get_odd_subarrays([2, 2, 5, 6, 9, 11, 4, 2], 2))
    k = MinCoins([2], 3)
    print(k.min_coins_dp())