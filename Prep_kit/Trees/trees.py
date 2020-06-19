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
    starting with lowest count characters, ie 'B' and 'C' as leaves and their root containing their total count(2).
            #,5
          0     1
        #,2     A,3

    B,1     C,1

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