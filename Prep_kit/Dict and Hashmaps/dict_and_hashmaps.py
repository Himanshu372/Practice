from collections import defaultdict

def checkMagazine(magazine, note):
    result = 'Yes'
    if len(note) > len(magazine):
        result = 'No'
        return result
    else:
        mag_dict = defaultdict(int)
        for each in magazine:
            mag_dict[each] += 1
        for each in note:
            if mag_dict[each] == 0:
                result = 'No'
                break
            else:
                mag_dict[each] -= 1
        return result


if __name__ == '__main__':
    print(checkMagazine(['two', 'times', 'three', 'is', 'not', 'four'], ['two', 'times', 'two', 'is', 'four']))
