import time
import itertools

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


def dummy(min_sup, database, itemset):
    L = []
    t_start = time.time()
    for k in range(1, len(itemset) + 1):
        Lk_1 = []
        Ck_1 = [set(itemset) for itemset in itertools.combinations(itemset, k)]
        c_count = [0 for _ in range(len(Ck_1))]
        for t in database:
            c_count = subset(Ck_1, set(t), c_count)
        for idx, c in enumerate(Ck_1):
            if c_count[idx] >= min_sup:
                c = list(c)
                c.append(c_count[idx])
                Lk_1.append(c)
        if not Lk_1:
            break
        L.extend(Lk_1)
    t_end = time.time()
    with open('log.txt', 'at') as f:
        f.write('dummy: ' + str((t_end - t_start)) + '\n')
    return L
