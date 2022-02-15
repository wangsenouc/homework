import functools
import time
import argparse
import copy
from utils import find_frequent_1_itemsets


def has_infrequent_subset(c, Lk):
    '''
    用于判断剪枝，如果c共有k+1个元素，那么它的所有大小为k的子集必须存在于Lk中，这是因为frequent (k+1)-itemset 的所有非空子集也一定是频繁的
    :param c: 由Lk生成的(k+1)-itemset
    :param Lk:
    :return: 如果c的某个k-itemset 不在Lk中，则返回True，否则返回False
    '''
    for i in range(len(c)):
        new_c = c.copy()
        del new_c[i]
        if new_c not in Lk:
            return True

    return False


def apriori_gen(Lk):
    '''
    由Lk生成Ck+1候选集，里面用到了剪枝操作
    :param Lk: frequent k-itemsets
    :return: candidate (k+1)-itemsets
    '''
    size = len(Lk)
    Ck_1 = list()
    Lk_prime = []
    for trans in Lk:
        Lk_prime.append(trans[0: -1])
    for i in range(0, size):
        for j in range(i + 1, size):
            if Lk[i][0: -2] == Lk[j][0: -2]:
                c = Lk[i][0: -1]
                c.append(Lk[j][-2])
                if not has_infrequent_subset(c, Lk_prime):
                    Ck_1.append(c)
    return Ck_1


def subset(Ck_1: set, t: set, c_count):
    '''
    计算Ck+1中的每个 itemset 出现在t中次数
    :param Ck_1: 由Lk生成的k+1侯选集
    :param t: 数据库中的一条交易
    :param c_count: c_count[i]记录了Ck+1中第i个 itemset 在数据库中出现的次数
    :return:
    '''
    for idx, c in enumerate(Ck_1):
        if set(c) <= t:  # if c is the subset of t
            c_count[idx] += 1
    return c_count


def apriori(min_sup, DB, itemset):
    L = list()
    t_start = time.time()
    Lk = find_frequent_1_itemsets(DB, itemset, min_sup)
    while len(Lk) > 0:
        L.extend(Lk)
        Lk_1 = []  # Lk+1
        Ck_1 = apriori_gen(Lk)  # Ck+1
        c_count = [0 for _ in range(len(Ck_1))]
        Ck_1_set = []
        # 将list转为set的目的是方便集合的子集运算
        for c in Ck_1:
            Ck_1_set.append(set(c))
        for t in DB:
            c_count = subset(Ck_1_set, set(t), c_count)
        for idx, c in enumerate(Ck_1):
            if c_count[idx] >= min_sup:
                c.append(c_count[idx])
                Lk_1.append(c)
        Lk = Lk_1

    t_end = time.time()
    print('Apriori用时：', t_end - t_start)
    return L

