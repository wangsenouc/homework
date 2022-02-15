import argparse
from Apriori import apriori
from FPTree import fp_tree_method
from utils import load_groceries
from utils import load_unix_usage
from utils import process_sessions
from Dummy import dummy
import math
import functools


def cmp(x, y):
    if x[0] > y[0]:
        return -1
    elif x[0] < y[0]:
        return 1
    else:
        return 0


def subsets(set):
    size = len(set)
    flags = [1]*(size + 1)          #flag标志有无
    result = []                     #存储结果
    while flags[0] == 1:            #实现2进制减1操作
        i = size
        while flags[i] != 1:
            flags[i] = 1
            i = i - 1
        flags[i] = 0
        subset = []                 # 存储单个结果
        for j in range(1, size + 1):
            if flags[j] == 1:
                subset.append(set[j-1])
        if 0 < len(subset) < size:
            result.append(subset)
    return result


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--support', type=float, default=0.01)
    parser.add_argument('--confidence', type=float, default=0.5)
    parser.add_argument('--dataset', type=int, default=0)
    parser.add_argument('--algori', type=int, default=2)
    args = parser.parse_args()
    support = args.support
    confidence = args.confidence
    dataset = args.dataset
    algori = args.algori

    if dataset == 0:
        database, itemset = load_groceries()
    else:
        database, itemset = load_unix_usage()

    datasize = len(database)
    min_sup = math.ceil(support * datasize)
    if algori == 1:
        freq_list = apriori(min_sup, database, itemset)
    elif algori == 2:
        freq_list = fp_tree_method(min_sup, database, itemset)
    else:
        freq_list = dummy(min_sup, database, itemset)

    freq_dict = dict()
    # 字典的构造过程可能会出问题
    for l in freq_list:
        key = ''.join(l[0: -1])
        freq_dict[key] = l[-1]

    for j in range(1, 100):
        cnt = 0
        for itemset in freq_list:
            if len(itemset) == j + 1:
                cnt += 1
        if cnt == 0:
            break
        print(j, cnt)
    for l in freq_list:
        for s in subsets(l[0: -1]):
            key = ''.join(s)
            if l[-1] / float(freq_dict[key]) >= confidence:
                sup = l[-1] / float(datasize)
                confi = l[-1] / float(freq_dict[key])
                l_s = []
                for item in l[0: -1]:
                    if item not in s:
                        l_s.append(item)
                lift = confi / sup
                print(s, '-->', l_s, 'support: ', sup,
                      'confidence: ', confi, 'lift: ', lift)


if __name__ == '__main__':
    main()
