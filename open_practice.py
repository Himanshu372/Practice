from datetime import datetime, timedelta
from sys import stdin
# import pandas as pd
from copy import copy
from random import randint
from itertools import groupby
from collections import Counter
from fractions import Fraction
from itertools import combinations
from operator import itemgetter
import gzip
from subprocess import PIPE,Popen
import string
# import pandas as pd
import sys
import time
import re
from collections import defaultdict


def countingValleys(n, s):
    sea_level = False
    valley = 0
    height = 0
    for step in s:
        if step == 'U':
            height += 1
        elif step == 'D':
            height -= 1
        if height == 0:
            sea_level = True
        if height == 0 and sea_level == True and step == 'U':
            valley += 1
    return valley

# Implementating Linked List
class Node():
    """docstring for Node"""

    def __init__(self, data=None):
        self.data = data
        self.next = None

class linked_list():
    """docstring for Linked_list"""

    def __init__(self):
        self.head = Node()
        self.next = None

    def append(self, data):
        node = Node(data)
        curr = self.head
        while curr.next != None:
            curr = curr.next
        curr.next = node

    def length(self):
        curr = self.head
        lenght = 0
        while curr.next != None:
            curr = curr.next
            lenght += 1
        return lenght

    def display(self):
        curr = self.head
        l = []
        while curr.next != None:
            curr = curr.next
            l.append(curr.data)
        print(l)


    def get(self, index):
        curr = self.head
        count = 0
        while curr.next != None:
            curr = curr.next
            if index == count:
                return curr.data
            count += 1
        return print('Index out of range')

class bts_node(object):
    """docstring for bts_node"""
    def __init__(self, value = None):
        super(bts_node, self).__init__()
        self.value = value
        self.left = None
        self.right = None
        self.parent = None

class binary_tree(object):
    """docstring for binary_tree"""
    def __init__(self):
        super(binary_tree, self).__init__()
        self.root = None


    def insert(self, value):
        if self.root == None:
            self.root = bts_node(value)
        else:
            self._insert(value, self.root)

    def _insert(self, value, node):
        if value < node.value:
            if node.left != None:
                self._insert(value, node.left)
            else:
                node.left = bts_node(value)
                node.left.parent = node
        elif value > node.value:
            if node.right != None:
                self._insert(value, node.right)
            else:
                node.right = bts_node(value)
                node.right.parent = node
        else:
            print('Value already present in the tree!')

    def display(self):
        if self.root != None:
            k = self._display(self.root, 1)
            return k

    def _display(self, node, level, l = []):
        if node != None:
            self._display(node.left, level + 1, l)
            # print(node.value)
            # return (node.value, level)
            l.append((node.value, level))
            self._display(node.right, level + 1, l)
        return l

    def height(self):
        if self.root != None:
            return self._height(self.root)
        else:
            return 0

    def _height(self, node, cur_height = 0):
        if node == None: return cur_height
        left = self._height(node.left, cur_height + 1)
        right = self._height(node.right, cur_height + 1)
        return max(left, right)

    def search(self, value):
        if self.root != None:
            return self._search(value, self.root)
        else:
            return False
    def _search(self, value, node):
        if node.value == value:
            return True
        elif value < node.value and node.left != None:
            return self._search(value, node.left)
        elif value > node.value and node.right != None:
            return self._search(value, node.right)
        else:
            return False


    def find_node(self, value):
        if self.root != None:
            return self._find_node(value, self.root)
        else:
            return None
    def _find_node(self, value, node):
        if node.value == value:
            return node
        elif value < node.value and node.left != None:
            return self._find_node(value, node.left)
        elif value > node.value and node.right != None:
            return self._find_node(value, node.right)
        else:
            return 'Value not present'

    def num_children(self, node):
        children = 0
        if node.left != None: children += 1
        if node.right != None: children += 1
        return children


    def min_node(self, node):
        curr = node
        while curr.left != None:
            curr = curr.left
        return curr


    def delete(self, value):
        return self._delete(self.find_node(value))


    def _delete(self, node):
        parent = node.parent
        children = self.num_children(node)
        if children == 0:
            if parent.left == node:
                parent.left = None
            else:
                parent.right = None

        if children == 1:
            if node.left != None:
                child = node.left
            else:
                child = node.right
            if parent.left == node:
                parent.left = child
            else:
                parent.right = child
            child.parent = parent

        if children == 2:
            min_child = self.min_node(node.right)
            node.value = min_child.value
            self.delete(min_child)




    # def delete(self, value):
    #
    #
    #
    # def _delete(self, value, node):
    #     return




def populate_tree(tree):
    from random import randint
    max_items = 10
    max_num = 1000
    for _ in range(max_items):
        i = randint(0, max_num)
        print(i)
        tree.insert(i)
    return tree


def validate_bst(root, min = -sys.maxsize, max = sys.maxsize):
    if root == None:
        return True
    if root.value > min and root.value < max and validate_bst(root.left, min = -sys.maxsize, max = root.value) and validate_bst(root.right, min = root.value, max = sys.maxsize):
        return True
    else:
        return False


def invert_tree(t):
    if t.root != None:
        left = t.root.left
        right = t.root.right
        t.root = invert_node(t.root)
        return t
    else:
        return 'Tree is empty'

def invert_node(node):
    if node == None:
        return None
    elif node.left != None and node.right != None:
        left = invert_node(node.left)
        right = invert_node(node.right)
        node.left, node.right = right, left
        return node
    elif node.left == None and node.right != None:
        right = invert_node(node.right)
        node.left, node.right = right, None
        return node
    elif node.right == None and node.left != None:
        left = invert_node(node.left)
        node.right, node.left = left, None
        return node
    else:
        return node


# Implementating graphs
# Graph with adjaceny list
class Vertex():
    def __init__(self, name):
        self.name = name
        self.neighbours = []
        self.color = 'black'
        self.distance = sys.maxsize

    def add_neighbour(self, vertex_name):
        if vertex_name not in self.neighbours:
            self.neighbours.append(vertex_name)
            self.neighbours.sort()




class Graph():
    def __init__(self):
        self.vertices = {}

    def add_vertex(self, vertex):
        if isinstance(vertex, Vertex) and vertex.name not in self.vertices.keys():
            self.vertices[vertex.name] = vertex
        else:
            print('Vertex already in the graph')

    def add_edge(self, u, v):
        if u in self.vertices and v in self.vertices:
            for key, value in self.vertices.items():
                if key == u:
                    value.add_neighbour(v)
                if key == v:
                    value.add_neighbour(u)
            return True
        else:
            return False

    def bfs(self, vert_name):
        q = []
        node_vert = self.vertices[vert_name]
        node_vert.distance = 0
        node_vert.color = 'red'

        for u in node_vert.neighbours:
            node_u = self.vertices[u]
            node_u.distance = node_vert.distance + 1
            q.append(u)

        while len(q) != 0:
            v = q.pop(0)
            node_v = self.vertices[v]
            node_v.color = 'red'

            for n in node_v.neighbours:
                node_n = self.vertices[n]
                if node_n.color == 'black':
                    q.append(n)
                    if node_n.distance > node_v.distance + 1:
                        node_n.distance = node_v.distance + 1

    def print_graph(self):
        for key in sorted(list(self.vertices.keys())):
            print(key + ':' + ''.join(self.vertices[key].neighbours) + ' Distance from source(A):' + str(self.vertices[key].distance))








# Graph with adjaceny matrix
# class Vertex():
#     def __init__(self, name):
#         self.name = name
#
# class Graph():
#     def __init__(self):
#         self.edges = []
#         self.vertices = {}
#         self.edge_vertices = {}
#
#     def add_vertex(self, vertex):
#         if isinstance(vertex, Vertex) and vertex not in self.vertices:
#             self.vertices[vertex.name] = vertex
#             for row in range(len(self.edges)):
#                 self.edges[row].append(0)
#             self.edges.append([0] * (len(self.edges) + 1))
#             self.edge_vertices[vertex.name] = len(self.edge_vertices)
#             return True
#         else:
#             return False
#
#     def add_edge(self, u, v, weight):
#         if u in self.vertices and v in self.vertices:
#             self.edges[self.edge_vertices[u]][self.edge_vertices[v]] = weight
#             self.edges[self.edge_vertices[v]][self.edge_vertices[u]] = weight
#             return True
#         else:
#             return False
#
#     def graph_display(self):
#         for key, value in sorted(self.edge_vertices.items()):
#             print(key, end = ' ')
#             for j in self.edges[value]:
#                 print(j, end = ' ')
#             print('')



# Implementating Depth First Search
class Vertex():
    def __init__(self, name):
        self.name = name
        self.color = 'black'
        self.discovery = 0
        self.finish = 0
        self.neighbours = []

    def add_neighbour(self, vert_name):
        if vert_name not in set(self.neighbours):
            self.neighbours.append(vert_name)
            sorted(self.neighbours)



class Graph():
    def __init__(self):
        self.vertices = {}
        time = 0


    def add_vertex(self, vertex):
        if isinstance(vertex, Vertex) and vertex.name not in self.vertices.keys():
            self.vertices[vertex.name] = vertex
        else:
            print('Vertex already in the graph')

    def add_edge(self, u, v):
        if u in self.vertices and v in self.vertices:
            for key, value in self.vertices.items():
                if key == u:
                    value.add_neighbour(v)
                if key == v:
                    value.add_neighbour(u)
            return True
        else:
            return False

    def dfs(self, vertex):
        global time
        time = 1
        self._dfs(vertex)

    def _dfs(self, vertex):
        global time
        time += 1
        node_v = self.vertices[vertex]
        node_v.color = 'red'
        node_v.discovery = time
        for u in node_v.neighbours:
            node_u = self.vertices[u]
            if node_u.color == 'black':
                self._dfs(u)
        node_v.color = 'blue'
        node_v.finish = time + 1
        time += 1


    def print_graph(self):
        for key in sorted(list(self.vertices.keys())):
            print(key + ':' + ''.join(self.vertices[key].neighbours) + ' Discovery_time:' + str(self.vertices[key].discovery) + ' Finish_time:' + str(self.vertices[key].finish))




class Stack():
    def __init__(self):
        self.list = []

    def push(self, element):
        self.list.append(element)
        return None

    def pop(self):
        element = self.list.pop(len(self.list) - 1)
        return element

    def print_stack(self):
        print(self.list)


class Queue():
    def __init__(self):
        self.entryStack = Stack()
        self.exitStack = Stack()


    def enqueue(self, element):
        self.entryStack.push(element)
        return None
    def dequeue(self):
        if len(self.exitStack) != 0:
            element = self.exitStack.pop()
            return element


class patternMatchingAlgo(object):

    def __init__(self, string, substring):
        '''
        A class implementating pattern matching algorithms. For instanstiating, a substring and a string is required
        :param string:
        :param substring:
        '''
        self.string = string
        self.substring = substring
        self.n = len(string)
        self.m = len(substring)

    def brute_force_match(self):
        '''
        Return the lowest index of string where the substring begins or -1
        :return:
        '''
        for i in range(self.n - self.m + 1):
            k = 0
            while k < self.m and self.string[i + k] == self.substring[k]:
                k += 1
                if k == self.m:
                    return i
        return -1








if __name__=='__main__':
    obj = patternMatchingAlgo('Himanshu', 'man')
    print(obj.brute_force_match())
    # print('Start')
    # k = 'sddasd 10 sdsd 30.4 sdsdsad '
    #
    # s = Stack()
    # s.push(5)
    # s.push(4)
    # s.push(3)
    # s.push(2)
    # s.push(1)
    # s.print_stack()
    # s.pop()
    # s.print_stack()

    # g = Graph()
    # a = Vertex('A')
    # b = Vertex('B')
    # c = Vertex('C')
    # d = Vertex('D')
    # e = Vertex('E')
    #
    # g.add_vertex(a)
    # g.add_vertex(b)
    # g.add_vertex(c)
    # g.add_vertex(d)
    # g.add_vertex(e)
    #
    # g.add_edge('A', 'B')
    # g.add_edge('A', 'E')
    # g.add_edge('A', 'C')
    # g.add_edge('B', 'C')
    # g.add_edge('C', 'E')
    # g.add_edge('C', 'D')
    #
    # g.dfs('A')
    #
    # g.print_graph()

    # t = binary_tree()
    # t.insert(0)
    # t.insert(13)
    # t.insert(15)
    # k = t.display()
    # print(k)
    # # h = t.height()
    # # print(h)
    # # print(t.search(13))
    # # t.delete(15)
    # # t.display()
    # print(validate_bst(t.root.left))
    # print('''Inverted''')
    # k = invert_tree(t)
    # k.display()

    # def solve(a, p, k, N):
    #     # Write your code here
    #     attach_scores = []
    #     for i in range(len(p)):
    #         l = []
    #         l.append(a[i])
    #         for j in range(i + 1, len(p)):
    #             if abs(p[i] - p[j]) < k:
    #                 l.append(a[j])
    #         attach_scores.append(sum(k for k in l))
    #     return max(attach_scores)
    #
    # print(solve([9, 10, 5, 12, 1], [5, 2, 9, 3, 1], 1, 5))


    # print(countingValleys(8, 'UDDDUDUU'))

    # from sys import stdin
    #
    # line = stdin.readline()
    # c = 0
    # while line:
    #     if c = 0:
    #         size = line.split()[0]
    #         e = line.split()[1]
    #         c += 1
    #     else:
    #
    #     line = stdin.readline()

    # from sys import stdin
    #
    # n, k = stdin.readline().split()
    # n = int(n)
    # k = int(k)
    #
    # cnt = 0
    # lines = stdin.readlines()
    # for line in lines:
    #     if int(line) % k == 0:
    #         cnt += 1





    # Hackerearth code monk palindrome problem
    # lines = ['3', 'abc', 'abba', 'aba']
    # n = int(lines[0])
    # for s in range(n):
    #     k = lines[s + 1]
    #     if len(k) % 2 != 0 and k[:len(k) // 2] == k[len(k) // 2 + 1:][::-1]:
    #         print('YES ODD')
    #     elif len(k) % 2 == 0 and k[:len(k) // 2] == k[len(k) // 2:][::-1]:
    #         print('YES EVEN')
    #     else:
    #         print('Issue')


    #
    # m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    # count = 0
    # for i in range(len(m)):
    #     for j in range(len(m[0])):
    #         e = m[i][j]
    #         for k in range(i, len(m)):
    #             for l in range(j, len(m[0])):
    #                 next = m[k][l]
    #                 if e > next:
    #                     count += 1
    # print(count)

    # from sys import stdin
    #
    # n = int(stdin.readline())
    # l = sorted(r)
    # t = int(stdin.readline())
    #
    # for each_t in range(t):
    #     e = int(stdin.readline())
    #     low = 0
    #     high = n
    #     while (low <= high):
    #         med = (low + high) // 2
    #         if e < l[med]:
    #             high = med
    #         elif e > l[med]:
    #             low = med
    #         else:
    #             print(med + 1)
    #             break


    # k = 6
    # m = 8
    # def func(x):
    #     return 2 * x * x - 12 * x + 7
    #
    #
    # def ts(l, r):
    #     m1 = l + ((r - l) + 1) // 3
    #     m2 = r - ((r - l) + 1) // 3
    #     if m1 == m2:
    #         return m1
    #     elif func(m1) < func(m2):
    #         ts(l, m2 - 1)
    #     else:
    #         ts(m1 + 1, r)
    #
    #
    # k = ts(k, m)
    # print(k)

    # from sys import stdin
    #
    # n = int(stdin.readline())
    # l = [int(i) for i in stdin.readline().split()]
    # def bubble_sort(l):
    #     n = len(l)
    #     swap = 0
    #     for i in range(n):
    #         for k in range(n - i - 1):
    #             if l[k] > l[k + 1]:
    #                 swap += 1
    #                 temp = l[k]
    #                 l[k] = l[k + 1]
    #                 l[k + 1] = temp
    #     print(swap)
    #
    # n = 4
    # s = 2
    # l = [7, 5, 4, 2]
    #
    # def selection_sort(l):
    #     steps = 0
    #     n = len(l)
    #     for i in range(n):
    #         m = min(l[i:n])
    #         print(m)
    #         temp = l[i]
    #         l[l.index(m)] = temp
    #         print(l)
    #         l[i] = m
    #         print(l)
    #         steps += 1
    #         if steps == s:
    #             print(' '.join(str(k) for k in l))
    #             break
    #
    # n = 4
    # s = 2
    # l = [7, 5, 2, 4, 10]
    # def selection_sort_recursive(l):
    #     if len(l) != 0:
    #         k = min(l)
    #         l.pop(l.index(k))
    #         return [k] + selection_sort_recursive(l)
    #     else:
    #         return []
    # print(selection_sort_recursive(l))
    # steps = 0
    # for i in range(n):
    #     minimum = i
    #     for j in range(i+1, n):
    #         if l[j] < l[minimum]:
    #             minimum = j
    #     steps += 1
    #     if steps == s:
    #         print(' '.join(str(k) for k in l))
    #         break

    # n = 4
    # l = [7, 4, 5, 2]
    # o = copy(l)
    # for i in range(n):
    #     for j in range(i, 0, -1):
    #         if l[j - 1] > l[j]:
    #             temp = l[j - 1]
    #             l[j - 1] = l[j]
    #             l[j] = temp
    # print(' '.join(str(l.index(k)) for k in o))

    def insertion_sort(l):
        for i in range(len(l)):
            for j in range(i, 0, -1):
                if l[j - 1] > l[j]:
                    temp = l[j]
                    l[j] = l[j - 1]
                    l[j - 1] = temp
        return l

    # print(insertion_sort(l))

    # def merge(a, b):
    #     c = []
    #     a_ix, b_ix = 0, 0
    #     while a_ix < len(a) and b_ix < len(b):
    #         if a[a_ix] < b[b_ix]:
    #             c.append(a[a_ix])
    #             a_ix += 1
    #         else:
    #             c.append(b[b_ix])
    #             b_ix += 1
    #     if a_ix == len(a):
    #         c.extend(b[b_ix:])
    #     else:
    #         c.extend(a[a_ix:])
    #     return c
    #
    # # Idea is to split a given list into individual elements and to build from bottom up a sorted array
    # def merge_sort(a):
    #     if len(a) <= 1:
    #         return a
    #     else:
    #         left, right = merge_sort(a[:len(a) // 2]), merge_sort(a[len(a) // 2:])
    #     return merge(left, right)
    # #
    # l = [1, 4, 3, 2, 5, 3, 2]
    # print(merge_sort(l))

    # Idea is to define a random pivot for an array and sort elements based on this pivot
    # def quick_sort(a):
    #     if len(a) <= 1: return a
    #     piv = randint(0, (len(a) - 1))
    #     left, center, right = [], [], []
    #     for i in a:
    #         if i < a[piv]:
    #             left.append(i)
    #         elif i == a[piv]:
    #             center.append(i)
    #         else:
    #             right.append(i)
    #     return quick_sort(left) + center + quick_sort(right)
    #
    # print(quick_sort(l))

    # Helpful when array has repetitive elements
    # def count_sort(a):
    #     m = max(a)
    #     aux = [0 for i in range(m + 1)]
    #     sorted = []
    #     for k in a:
    #         c = a.count(k)
    #         aux[k] = c
    #     for ind in range(len(aux)):
    #         if aux[ind] > 1:
    #         #     print(ind, aux[ind])
    #             for i in range(aux[ind]):
    #                 sorted.append(ind)
    #         elif aux[ind] == 1:
    #             sorted.append(ind)
    #     print(sorted)



    # count_sort(l)

    # def findNumber(arr, k):
    #     for i in arr:
    #         if i == k:
    #             ans = 'YES'
    #             break
    #     else:
    #         ans = 'NO'
    #     return ans
    #
    # arr = [1,2,3,4,5]
    # k = 1
    # findNumber(arr = arr, k = k)

    # def numberOfAlerts(preceedingMinutes, alertThreshold, numberCalls):
    #     # Write your code here
    #     alert = 0
    #     for t in range(len(numberCalls) - (preceedingMinutes - 1)):
    #
    #         avg = sum(k for k in numberCalls[t : t +  preceedingMinutes]) / preceedingMinutes
    #         if avg > alertThreshold:
    #             alert += 1
    #     return alert

    # m = 3
    # aT = 10
    # n = [0, 11, 10, 10, 7]
    #
    #
    # def reassignedPriorities(issuePriorities):
    #     d = {}
    #     aux = copy(issuePriorities)
    #     for i in range(len(set(aux))):
    #         m = min(aux)
    #         d[m] = i + 1
    #         for k in range(aux.count(m)):
    #             aux.remove(m)
    #     return [d[j] for j in issuePriorities]
    #
    # res = reassignedPriorities(n)
    #
    # print(res)

    # def bin_sort(l, place):
    #     n = len(l)
    #     bins = [0 for i in range(10)]
    #     l_s = [0 for i in range(n)]
    #
    #     for k in range(n):
    #         bins[(l[k] // place) % 10] += 1
    #
    #     for j in range(1, len(bins)):
    #         bins[j] += bins[j - 1]
    #
    #
    #     i = n - 1
    #     while i >= 0:
    #         l_s[bins[(l[i] // place) % 10] - 1] = l[i]
    #         bins[(l[i] // place) % 10] -= 1
    #         i -= 1
    #
    #     return l_s
    #
    #
    # def radix_sort(l):
    #     n = len(l)
    #     m = max(l)
    #     mul = 1
    #     while m:
    #         l = bin_sort(l, mul)
    #         mul *= 10
    #         m //= 10
    #     return l
    #
    #

    # def radix_sort_python(l):
    #     n = len(l)
    #     m = max(l)
    #
    #     rounds = len(str(m))
    #     for round in range(rounds - 1, -1, -1):
    #         buckets = [[] for i in range(10)]
    #         for i in l:
    #             i = '0' * (rounds - len(str(i))) + str(i)
    #             sig_dig = int(i[round])
    #             buckets[sig_dig % 10].append(int(i))
    #         l = [item for sublist in buckets for item in sorted(sublist)]
    #
    #     return l
    #
    # print(radix_sort_python([719, 23, 81]))




    # l = [170, 45, 90, 70, 2]
    # print(radix_sort(l))



    # Batman code
    # import sys
    # import math
    #
    # # Auto-generated code below aims at helping you parse
    # # the standard input according to the problem statement.
    #
    # # w: width of the building.
    # # h: height of the building.
    # w, h = [int(i) for i in input().split()]
    # w, h = [10, 10]
    # n = 6  # maximum number of turns before game over.
    # x0, y0 = [2, 5]
    # bomb_dir = (7, 4)
    # # game loop
    # while True:
    #     bomb_dir = (7, 4) # the direction of the bombs from batman's current location (U, UR, R, DR, D, DL, L or UL)
    #
    #     # Write an action using print
    #     # To debug: print("Debug messages...", file=sys.stderr)
    # k = 0
    # d = {'U': 0, 'R': 10, 'L': 0, 'D': 10, 'UR': (0, 10), 'DR': (10, 10), 'UL': (0, 0), 'DL': (10, 0)}
    # while k <= n or (x0, y0) != bomb_dir:
    #     for i in range(2):
    #         if len(bomb_dir) == 2 and i == 0:
    #             move = d[bomb_dir]
    #             if move[1] == 10:
    #                 x0 += 1
    #             else:
    #                 x0 -= 1
    #             if move[0] == 10:
    #                 y0 += 1
    #             else:
    #                 y0 -= 1
    #
    #         if len(bomb_dir) == 1 and i == 0:
    #             move = d[bomb_dir]
    #             if move == 10 and bomb_dir == 'D':
    #                 y0 += 1
    #             elif move == 0 and bomb_dir == 'U':
    #                 y0 -= 1
    #             elif move == 10 and bomb_dir == 'R':
    #                 x0 += 1
    #             elif move == 0 and bomb_dir == 'L':
    #                 x0 -= 1
    #             print('{} {}'.format(x0, y0))
    #         if len(bomb_dir) == 2 and i == 1:
    #             move = d[bomb_dir]
    #             if move[0] == 0 and move[1] == 10:
    #                 x0 = ((move[1] - 1) + x0 + 1) // 2
    #                 y0 = (move[0] + (y0 - 1)) // 2
    #             if move[0] == 10 and move[1] == 10:
    #                 x0 = ((move[1] - 1) + x0 + 1) // 2
    #                 y0 = ((move[0] - 1) + y0 + 1) // 2
    #             if move[0] == 0 and move[1] == 0:
    #                 x0 = (move[1] + (x0 - 1)) // 2
    #                 y0 = (move[0] + (y0 - 1)) // 2
    #             if move[0] == 10 and move[1] == 0:
    #                 x0 = (move[1] + (x0 - 1)) // 2
    #                 y0 = ((move[0] - 1) + y0 + 1) // 2
    #             print('{} {}'.format(x0, y0))
    #         if len(bomb_dir) == 1 and i == 1:
    #             move = d[bomb_dir]
    #             if move == 10 and bomb_dir == 'D':
    #                 y0 = ((move[0] - 1) + y0 + 1) // 2
    #             elif move == 0 and bomb_dir == 'U':
    #                 y0 = (move[0] + (y0 - 1)) // 2
    #             elif move == 10 and bomb_dir == 'R':
    #                 x0 = ((move[1] - 1) + x0 + 1) // 2
    #             elif move == 0 and bomb_dir == 'L':
    #                 x0 = (move[1] + (x0 - 1)) // 2
    #             print('{} {}'.format(x0, y0))
    # print(x0, y0)

    # temp = [float(i) for i in input().split()]
    # input = 'azxxzyyyddddyzzz'
    # l = [i for i in input]
    #
    # for i in range(len(l)):
    #     c = l.count(l[i])
    #     k = 0
    #     if c == 1:
    #         pass
    #     elif i == 0 or l[i - 1] != l[i]:
    #         k += 1
    #         even = [k for k in range(2, c + 1) if k % 2 == 0]
    #         even = sorted(even, reverse = True)
    #         for j in even:
    #             s = ''.join(l[i:i + j])
    #             p = '{}'.format(l[i] * j)
    #             s_1 = ''.join(l[i:i + j + 1])
    #             p_1 = '{}'.format(l[i] * (j + 1))
    #             if s == p and s_1 != p_1:
    #                 l = l[0:i] + l[i+j:]
    #
    #                 break
    #     break


    # def rm_even(s):
    #     to_join = []
    #     for _, g in groupby(s):
    #         chars = list(g)
    #         if len(chars) % 2:
    #             to_join.extend(chars)
    #     if to_join == s:
    #         return ''.join(to_join)
    #     return rm_even(to_join)
    #
    # print(rm_even(input))

    # def heapify(l, n):
    #     if n == 1:
    #         return l
    #     else:
    #         s = max_heap(l)
    #         temp = s[0]
    #         s[0] = s[n - 1]
    #         s[n - 1] = temp
    #         return heapify(s[0:n - 1], n - 1) + [temp]
    #
    # def max_heap(l):
    #     m = max(l)
    #     temp = l.index(m)
    #     l[temp] = l[0]
    #     l[0] = m
    #     return l
    # k = [2, 8, 5, 3, 9, 1]
    #
    # print(heapify(k, 6))

    # def bucket_sort(l):
    #     # No of buckets to be created
    #     size = max('{0:b}'.format(e).count('1') for e in l)
    #     # Creating buckets
    #     bucket = [[] for i in range(size)]
    #     for each in l:
    #         # Counting set bits for the each integer
    #         c = '{0:b}'.format(each).count('1')
    #         # Appending integer in respective bucket
    #         bucket[c - 1].append(each)
    #     # Insertion sort for each list of bucket
    #     for each_l in bucket:
    #         for k in range(len(each_l)):
    #             j = k + 1
    #             while j < len(each_l):
    #                 if each_l[k] > each_l[j]:
    #                     temp = each_l[j]
    #                     each_l[j] = each_l[k]
    #                     each_l[k] = temp
    #                 j += 1
    #
    #     return bucket
    # l = [2, 3, 5, 1, 8]
    # for each in bucket_sort(l):
    #     print(' '.join(str(i) for i in each))


    # def max_bottles_greedy(l, cap):
    #     l = sorted(l)
    #     j = 0
    #     c = 0
    #     while c <= cap and c + l[j] <= cap:
    #         c += l[j]
    #         j += 1
    #     return j
    #
    # l = [8, 4, 5, 3, 2]
    # cap = 10
    # print(max_bottles_greedy(l, cap))

    # def birds(l):
    #     k = Counter(l)
    #     m = max(k.values())
    #     p = [x for x,v in k.items() if v == m]
    #     return min(p)
    # l = [1, 1, 2, 2, 3]



    # def sockMerchant(n, ar):
    #     p = Counter(ar)
    #     k = 0
    #     for x,v in p.items():
    #         k += v//2
    #     return k

    # print(sockMerchant(5, l))

    # def pageCount(n, p):
    #     if n % 2 == 0:
    #         if n == p or n == 1:
    #             return 0
    #         elif p > n // 2:
    #             start = n
    #             return ((start - p) % 2 + (start - p) // 2)
    #         else:
    #             start = 1
    #             return ((p - start) // 2 + (p - start) % 2)
    #     else:
    #         if n == p or n == 1:
    #             return 0
    #         elif p > n // 2:
    #             start = n
    #             return (start - p) // 2
    #         else:
    #             start = 1
    #             return ((p - start) // 2 + (p - start) % 2)
    # print(pageCount(9, 5))

    # print(Fraction('0.474612399239923992399').limit_denominator())
    # def get_all_substrings(input_string):
    #     length = len(input_string)
    #     return [input_string[i:j + 1] for i in range(length) for j in range(i, length)]
    #
    # def segments(s):
    #     n = len(s)
    #     if n == 2:
    #         if s[0] == s[1]:
    #             print('lucky')
    #     elif n > 2:
    #         for each in range(len(s)):
    #             compare = s[0:each + 1]
    #             combo_list = get_all_substrings(s[each+1:])
    #             print(combo_list)
    #             for c in combo_list:
    #                 e = sum(int(i) for i in c)
    #                 if int(compare) == e:
    #                         len_list = [k for k in combo_list if len(c) == len(k)]
    #                         for j in len_list:
    #                             p = sum(int(x) for x in j)
    #                             if int(compare) == p:
    #                                 print('lucky')
    #                                 break
    #
    #         else:
    #             print('unlucky')
    # print(segments('73452'))

    # def database_backup(host, database_name, user_name, schema_name, database_password):
    #     command = 'pg_dump -h {0} -d {1} -U {2} -p 5432 -n {3} | gzip > /Users/himanshusanjivjagtap/Desktop/backup.gz'.format(host,database_name,user_name,schema_name)
    #     # p = Popen(command, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE)
    #     # p.communicate(input = b'mopdevmaster')
    #
    #     with gzip.open('backup.gz', 'wb') as f:
    #         p = Popen(command, shell = True, stdout=PIPE, universal_newlines=True)
    #         p.communicate(input=b'mopdevmaster')
    #         for stdout_line in p.stdout.readline():
    #             f.write(stdout_line.encode('utf-8'))
    #
    #             p.stdout.close()
    #             p.wait()
    #
    #
    #
    # database_backup('mopdevdb.cxq6ek6k9vz7.ap-south-1.rds.amazonaws.com', 'mopdevdb','mop_dev_user', 'sch_mopdev','mopdevmaster')




    # for i in range(3):
    #     for j in list(string.ascii_uppercase):
    #         if i == 0:
    #             print('context.{}=row1.{}'.format(j, j))
    #         if i == 1:
    #             print('context.{}{}=row1.{}{}'.format('A', j, 'A', j))
    #         if i == 2:
    #             print('context.{}{}=row1.{}{}'.format('B', j, 'B', j))


    # Code for testing linked_list class
    # n = linked_list()
    # n.append(1)
    # n.append(2)
    # n.display()
    # k = n.get(1)
    # print(k)


    # Simple code for Caesar's cipher
    # def caesar(string, rotate_by):
    #     pass


    # Palindrome check for string
    # def palindrome_string(s):
    #     #     return print('Yes') if s == s[::-1] else print('No')
    #     #
    #     # palindrome_string('start')
    #     #
    #     # # Palindrome check for number
    #     #
    #     # def palindrome_number(num):
    #     #     n = num
    #     #     rev_num = 0
    #     #     while n != 0:
    #     #         d = n % 10
    #     #         rev_num = rev_num * 10 + d
    #     #         n = n // 10
    #     #     return print('Yes') if num == rev_num else print('No')
    #     #
    #     #
    #     # palindrome_number(6446)



    # Max sum of not adjacent elements

    # def max_sum(l):
    #     if len(l) % 2 == 0:
    #         even_sum = sum(l[i] for i in range(len(l)) if i % 2 == 0)
    #         odd_sum = sum(l[i] for i in range(len(l)) if i % 2 != 0)
    #         return max(even_sum, odd_sum)
    #     if len(l) % 2 != 0:
    #         even_sum = sum(l[i] for i in range(len(l)) if i % 2 == 0)
    #         odd_sum = sum(l[i] for i in range(len(l)) if i % 2 != 0)
    #         return max(even_sum, odd_sum)
    # k = [5, 5, 10, 100, 10, 5]
    # print(max_sum(k))
    #
    #
    # def min_coins(value, coins_list):
    #     l = []
    #     for i in coins_list:
    #         c = 0
    #         v = value
    #         while v >= i:
    #             v -= i
    #             c += 1
    #         if v == 0:
    #             l.append(c)
    #             break
    #         for j in coins_list[(coins_list.index(i) + 1): ]:
    #             while v >= j:
    #                 v -= j
    #                 c += 1
    #             if v == 0:
    #                 l.append(c)
    #                 break
    #     return min(l)
    # #
    # def min_coins_dp(value, coins_list):
    #     l = [sys.maxsize for i in range(value + 1)]
    #     l[0] = 0
    #     for i in range(1, len(l)):
    #         for j in coins_list:
    #             if j <= i:
    #                 sub_res = l[i - j]
    #                 if sub_res + 1 < l[i]:
    #                     l[i] = sub_res + 1
    #                     break
    #     return l[value]
    # #
    # start1 = time.time()
    # coins = min_coins(31, [9, 6, 5, 1])
    # end1 = time.time()
    # print(end1 - start1)
    # start2 = time.time()
    # coins_dp = min_coins_dp(31, [9, 6, 5, 1])
    # end2 = time.time()
    # print(end2 - start2)
    # print(coins)
    # print(coins_dp)


    # Levenshtein distance(DP approach)
    # Repr distance between 2 strings, count of edits(insertion, deletions or substitutions) required to convert one to other
    # def levenshtein_distance_dp(s, t):
    #     m = len(s)
    #     n = len(t)
    #     memoize_array = [[sys.maxsize for i in range(n + 1)] for j in range(m + 1)]
    #     for each in range(m + 1):
    #         memoize_array[each][0] = each
    #     for each in range(n + 1):
    #         memoize_array[0][each] = each
    #     for i in range(1, m + 1):
    #         for j in range(1, n + 1):
    #             if s[i - 1] == t[j - 1]:
    #                 cost = 0
    #             else:
    #                 cost = 1
    #             memoize_array[i][j] = min(memoize_array[i - 1][j] + 1, memoize_array[i][j - 1] + 1, memoize_array[i - 1][j - 1] + cost)
    #     return memoize_array[m][n]
    #
    # k = levenshtein_distance_dp('CZTA', 'CAT')
    # print(k)

    # Kadane's algo
    # Max sum of contigous subarray in an array
    # def kadane_sum(l):
    #     sum = 0
    #     start = 0
    #     finish = 0
    #     for i in range(1, len(l)):
    #         if l[i] > 0:
    #             sum += l[i]
    #     return sum
    # #
    # def max_sum_2D_array(mat):
    #     max_sum = 0
    #     rows = len(mat)
    #     columns = len(mat[0])
    #     for left in range(columns):
    #         temp_row_sum = [0] * rows
    #         for right in range(left, columns):
    #             for i in range(rows):
    #                 temp_row_sum[i] += mat[i][right]
    #             k_sum = kadane_sum(temp_row_sum)
    #             if k_sum > max_sum:
    #                 max_sum = k_sum
    #     return max_sum
    #
    #
    # print(max_sum_2D_array([[1, 2],[3, 4],[5, 6]]))
    #
    # def aVeryBigSum(ar):
    #     # sig_dig_list = []
    #     # dig_list = []
    #     # for i in ar:
    #     #     sig_dig = ''
    #     #     while i % 10 != 0:
    #     #         sig_dig = int(str(i % 10) + str(sig_dig))
    #     #         sig_dig_list.append(sig_dig)
    #     #         i = i // 10
    #     #     dig_list.append(i)
    #     # sum_sig_dig_list = sum(sig_dig_list)
    #     # sum_dig_list = sum(dig_list)
    #     # return int(str(sum_dig_list)[:-(len(str(sum_sig_dig_list)) - 1)] + str(sum_sig_dig_list))
    #     return sum(ar)
    #
    # print(aVeryBigSum([1001458909, 1004570889, 1007019111, 1003302837, 1002514638, 1006431461, 1002575010, 1007514041, 1007548981, 1004402249]))

    # def kangaroo(x1, v1, x2, v2):
    #     if x1 > x2 and v2 > v1:
    #         if x2 != 0:
    #             return 'Yes' if ((x1 - x2) % (v2 - v1)) == 0 else 'No'
    #         # if v1 >= v2:
    #         #     return 'NO'
    #         # else:
    #         #     if x2 != 0:
    #         #         return 'YES' if (x1 % x2 == 0) and (v2 % v1 == 0) else 'NO'
    #         else:
    #             return 'YES' if ((x1 + v1) % v2 == 0) else 'NO'
    #     elif x2 > x1 and v1 > v2:
    #         if x1 != 0:
    #             return 'Yes' if ((x2 - x1) % (v1 - v2)) == 0 else 'NO'
    #         # if v2 >= v1:
    #         #     return 'NO'
    #         # else:
    #         #     if x1 != 0:
    #         #         return 'YES' if (x2 % x1 == 0) and (v1 % v2 == 0) else 'NO'
    #         else:
    #             return 'YES' if ((x2 + v2) % v1 == 0) else 'NO'
    #     else:
    #         if v1 == v2:
    #             return 'YES'
    #         else:
    #             return 'NO'
    #
    # print(kangaroo(21, 6, 47, 3))

    # def Find_It(X, K, S, N):
    #     # Write your code here
    #     num_blocks = N // X
    #     blocks = []
    #     for n in range(0, N, X):
    #         blocks.append(sorted(S[n : (n + X)]))
    #     Z = 1
    #     counter = 0
    #     for i in range(0, N, X):
    #         counter += 1
    #         if i <= K <= (i + X):
    #             num = blocks[0][counter - 1]
    #             for j in range(X):
    #                 Z += 1
    #                 if Z == K:
    #                     return num + ''.join(blocks[n][j] for n in range(1, len(blocks)))
    #
    #         Z += 4
    #
    #
    # print(Find_It(3, 3, '123456789', 9))


    # def solution(n):
    #     # Write your code here
    #     pairs = []
    #     for i in range(1, n + 1):
    #         power = pow(i, 1.5)
    #         if isinstance(power, int) and power <= n:
    #             pairs.append((i, pow()))
    #     return len(pairs)
    #
    #
    # print(solution(50))

    # Complete the countInversions function below.


    # Implementing merge sort to return swaps required.

    class MS:
        def __init__(self):
            self.swaps = 0

        def countInversions(self, arr):
            if len(arr) == 1:
                return arr
            else:
                left, right = self.countInversions(arr[:len(arr) // 2]), self.countInversions(arr[len(arr) // 2:])
                return self.merge(left, right)


        def merge(self, left, right):
            c = []
            a_ix, b_ix = 0, 0
            ll, lr = len(left), len(right)

            while a_ix < len(left) and b_ix < len(right):
                if left[a_ix] > right[b_ix]:
                    c.append(right[b_ix])
                    b_ix += 1
                    self.swaps += len(left) - a_ix
                else:
                    c.append(left[a_ix])
                    a_ix += 1

            if a_ix == len(left):
                c.extend(right[b_ix:])
            else:
                c.extend(left[a_ix:])
            return c

    ms = MS()
    print(ms.countInversions([2, 1, 3, 1, 2]))
    print(ms.swaps)


    # def maximumToys(prices, k):
    #     toys_list = []
    #     reduced_prices = [i for i in sorted(prices) if i < k]
    #     n = 0
    #     # for i in range(len(reduced_prices)):
    #     #     amt = 0
    #     #     number = 0
    #     #     while (i + number) < len(reduced_prices) and amt + reduced_prices[i + number] < k:
    #     #         amt += reduced_prices[i + number]
    #     #         number += 1
    #     #     toys_list.append(number)
    #     while n < len(reduced_prices) and k - reduced_prices[0] > 0:
    #         reduced_prices.pop(0)
    #         n += 1
    #     return n
    #
    # print(maximumToys([3, 7, 2, 9, 4], 15))

    # def activityNotifications(expenditure, d):
    #     notificaiton = 0
    #     for i in range(d, len(expenditure)):
    #         exp = sorted(expenditure[i - d:i])
    #         median = exp[len(exp) // 2] if len(exp) % 2 != 0 else (exp[len(exp) // 2 - 1] + exp[len(exp) // 2]) / 2
    #
    #         if expenditure[i] >= 2 * median:
    #             notificaiton += 1
    #     return notificaiton
    #
    #
    # print(activityNotifications([1, 2, 3, 4, 4], 4))

    # def maxMin(k, arr):
    #     unfairness = sys.maxsize
    #     arr = sorted(arr)
    #     for i in range(0, len(arr) - k):
    #         if (arr[i + (k - 1)] - arr[i]) < unfairness:
    #             unfairness = arr[i + (k - 1)] - arr[i]
    #     return unfairness
    #
    # print(maxMin(3, [100, 200, 300, 350, 400, 401, 402]))

    # def solve(S):
    #     # Reverse S
    #     S = S[::-1]
    #
    #     # Count each character in S.
    #     count = defaultdict(int)
    #     for c in set(S):
    #         count[c] = S.count(c)
    #
    #     need = {}
    #     for c in count:
    #         need[c] = count[c] / 2
    #
    #     solution = []
    #     min_char_at = -1
    #     while len(solution) < len(S) / 2:
    #         min_char_at = -1
    #         while True:
    #             c = S[i]
    #             if need[c] > 0 and (min_char_at < 0 or c < S[min_char_at]):
    #                 min_char_at = i
    #             count[c] -= 1
    #             if count[c] < need[c]:
    #                 break
    #             i += 1
    #
    #         # Restore all chars right of the minimum character.
    #         for j in range(min_char_at + 1, i + 1):
    #             count[S[j]] += 1
    #
    #         need[S[min_char_at]] -= 1
    #         solution.append(S[min_char_at])
    #
    #         i = min_char_at + 1
    #
    #
    #
    #     return ''.join(solution)
    #
    #
    # print(solve('abcdefgabcdefg'))


    # def primes(n):
    #     l = []
    #     for i in range(4, n):
    #         for j in range(2, int(pow(i, .5) + 1)):
    #             if i % j == 0:
    #                 break
    #         else:
    #             l.append(i)
    #     return l
    #
    # print(primes(50))

    # def intersection(s, l):
    #     d = {}
    #     result = []
    #     for i in set(s):
    #         d[i] = s.count(i)
    #     for j in l:
    #         if j in d.keys() and d[j] > 0:
    #             result.append(j)
    #             d[j] -= 1
    #     return result
    #
    # print(intersection([1,2,2,3], [2, 2, 3, 4]))


    # def largest_substring(s):
    #     l = 0
    #     max_str_len = 0
    #     sub_str = []
    #     while l < len(s) - 1:
    #         i = 0
    #         start = l
    #         while l < len(s) - 1 and s[l] <= s[l + 1]:
    #             i += 1
    #             l += 1
    #         if i > max_str_len:
    #             sub_str.append(s[start : l + 1])
    #             max_str_len = i
    #         l += 1
    #     return max(sub_str, key = lambda x : len(x))
    #
    # print(largest_substring('azcbobobegghakl'))


    # def closest_to_zero(l):
    #     neg, pos = [], []
    #     min_score = sys.maxsize
    #     x, y = 0, 0
    #     for i in l:
    #         if i < 0:
    #             neg.append(i)
    #         else:
    #             pos.append(i)
    #     for i in sorted(neg, reverse = True):
    #         for j in sorted(pos):
    #             if abs(j + i) < min_score:
    #                 min_score = abs(j + i)
    #                 x ,y = j, i
    #     return x , y
    #
    # print(closest_to_zero([15, 5, -20, 30, -45]))


    # An implementation on the lines of DFS(Depth First Search)
    # def search_islands(m):
    #     islands = 0
    #     # visited = [[False for col in range(len(m[0])) for row in len(m)]]
    #     for row in range(len(m)):
    #         for col in range(len(m[0])):
    #             if m[row][col] == 1:
    #                 islands += 1
    #                 find_adjacent(m, row, col)
    #     return islands
    #
    # def find_adjacent(m, row, col):
    #     if row >= len(m) or col >= len(m[0]) or row < 0 or col < 0:
    #         return None
    #     if m[row][col] == 0:
    #         return None
    #     m[row][col] = 0
    #
    #     for i in range(row - 1, row + 2):
    #         for j in range(col - 1, col + 2):
    #             if i != row or j != col:
    #                 find_adjacent(m, i, j)
    #
    # print('Islands:' + str(search_islands([[0, 1, 0, 1, 1], [0 ,0, 0, 1, 1], [0, 0, 0, 1, 0], [1, 0, 0, 0, 1]])))


    # def makeAnagram(a, b):
    #     common_string = list(set(a + b))
    #     count_dict = {}
    #     for i in range(len(common_string)):
    #         count_a = a.count(common_string[i])
    #         count_b = b.count(common_string[i])
    #         count_dict[i] = abs(count_a - count_b)
    #     uncommon = [None for key in count_dict.keys() for value in range(count_dict[key])]
    #     return len(uncommon)
    #
    #
    # print(makeAnagram('fcrxzwscanmligyxyvym', 'jxwtrhvujlmrpdoqbisbwhmgpmeoke'))

    # def isValid(s):
    #     count_dict = {i: s.count(i) for i in s}
    #     values_set = sorted(count_dict.values())
    #     max_values = values_set[-1]
    #     # remain_count_dict = values_set[:len(values_set) - 1]
    #
    #     if values_set[0] == max_values:
    #         return 'YES'
    #     # elif values_set[0] + 1 == max_values and len(values_set[:-1]) * \
    #     #         values_set[0] == sum(values_set[:-1]):
    #     #     return 'YES'
    #     # elif len(values_set[1:]) * values_set[1] == sum(values_set[1:]) and values_set[0] == 1:
    #     #     return 'YES'
    #     elif values_set[0] + 1 == max_values and len(set(values_set[:-1])) == 1:
    #         return 'YES'
    #     elif len(set(values_set[1:])) == 1 and values_set[0] == 1:
    #         return 'YES'
    #     else:
    #         return 'NO'
    #
    #
    # print(isValid('ibfdgaeadiaefgbhbdghhhbgdfgeiccbiehhfcggchgghadhdhagfbahhddgghbdehidbibaeaagaeeigffcebfbaieggabcfbiiedcabfihchdfabifahcbhagccbdfifhghcadfiadeeaheeddddiecaicbgigccageicehfdhdgafaddhffadigfhhcaedcedecafeacbdacgfgfeeibgaiffdehigebhhehiaahfidibccdcdagifgaihacihadecgifihbebffebdfbchbgigeccahgihbcbcaggebaaafgfedbfgagfediddghdgbgehhhifhgcedechahidcbchebheihaadbbbiaiccededchdagfhccfdefigfibifabeiaccghcegfbcghaefifbachebaacbhbfgfddeceababbacgffbagidebeadfihaefefegbghgddbbgddeehgfbhafbccidebgehifafgbghafacgfdccgifdcbbbidfifhdaibgigebigaedeaaiadegfefbhacgddhchgcbgcaeaieiegiffchbgbebgbehbbfcebciiagacaiechdigbgbghefcahgbhfibhedaeeiffebdiabcifgccdefabccdghehfibfiifdaicfedagahhdcbhbicdgibgcedieihcichadgchgbdcdagaihebbabhibcihicadgadfcihdheefbhffiageddhgahaidfdhhdbgciiaciegchiiebfbcbhaeagccfhbfhaddagnfieihghfbaggiffbbfbecgaiiidccdceadbbdfgigibgcgchafccdchgifdeieicbaididhfcfdedbhaadedfageigfdehgcdaecaebebebfcieaecfagfdieaefdiedbcadchabhebgehiidfcgahcdhcdhgchhiiheffiifeegcfdgbdeffhgeghdfhbfbifgidcafbfcd'))

    def triangular_number(n):
        return (pow(n, 2) + n) // 2


    def substrCount(n, s):
        # no_of_substr = n
        # for i in range(2, n + 1):
        #     j = 0
        #     while j + i <= n:
        #         sub_str = s[j : j + i]
        #         len_sub = len(sub_str)
        #         if len_sub % 2 == 0:
        #             if sub_str == sub_str[::-1]:
        #                 no_of_substr += 1
        #         else:
        #             if sub_str[:len_sub // 2] == sub_str[len_sub // 2 + 1:]:
        #                     no_of_substr += 1
        #         j += 1
        # return no_of_substr
        count = len(s)
        exp1 = r'(([a-z])\2*)(?!\1)(?=[a-z]\1)'
        m = re.finditer(exp1, s)
        count += sum([len(x.group(0)) for x in m])

        exp2 = r'([a-z])\1+'
        m = re.finditer(exp2, s)
        count += sum([triangular_number(len(x.group(0)) - 1) for x in m])

        return count

    print(substrCount(7, 'abcbaba'))

    # def char_list(s, l):
    #     # Dictionary
    #     d = {i:l.count(i) for i in set(l)}
    #     for i in s:
    #         try:
    #             if d[i] > 0:
    #                 d[i] -= 1
    #             elif d[i] == 0:
    #                 return False
    #         except:
    #             return False
    #     return True
    #     # for i in s:
    #     #     for j in l:
    #     #         if i == j:
    #     #             break
    #     #         else:
    #     #             pass
    #     #     return False
    #     # return True
    #
    #
    # print(char_list('apple', ['a', 'p', 'l', 'e']))

    # def fib_series(n):
    #     i = 0
    #     j = 1
    #     l = [0, 1]
    #     while i + j <= n:
    #         tmp = i + j
    #         i = j
    #         j = tmp
    #         l.append(tmp)
    #     return l
    #
    # def fibonacci_varint(n):
    #     l = fib_series(n)
    #     d = []
    #     for i in l:
    #         if i % 10 == 1 and i // 10 > 0:
    #           d.append(i % 10)
    #           break
    #         elif i // 10 > 0:
    #             d.append(i % 10)
    #         else:
    #             d.append(i)
    #     while len(d) > 1:
    #         d = [d[i] for i in range(len(d)) if i % 2 != 0]
    #     return d
    #
    #
    # print(fibonacci_varint(100))







































































