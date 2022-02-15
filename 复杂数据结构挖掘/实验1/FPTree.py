import functools
from utils import cmp
from utils import find_frequent_1_itemsets
import time


def sort_frequent_item(database, F_list):
    '''
    对database中的每一条transaction按照F_list排序，并且移除不在F_list中的item
    :param database:
    :param F_list:
    :return:
    '''
    new_database = []
    for trans in database:
        t = []
        for item in F_list:
            if item[0] in trans:
                t.append(item[0])
        new_database.append(t)
    return new_database


class FPTree:
    class Node:
        '''
        FP树上的节点
        '''
        def __init__(self, name):
            self.name = name
            self.frequency = 1
            self.parent = None
            self.children = []
            self.nextit = None

    def __init__(self):
        self.nodes = []
        node = self.Node('root')
        self.nodes.append(node)
        self.headtables = dict()

    def match(self, t, t_idx, node_idx, fre=None):
        '''
        寻找transaction t在FP树中的插入位置
        :param t: 一个itemset，即一条transaction
        :param t_idx:
        :param node_idx:
        :param fre: 若是由数据库构建FP树，则fre为None，若是由pattern-base构建FP树，则fre为t的frequency
        :return:
        '''
        if fre is None:
            self.nodes[node_idx].frequency += 1
        else:
            self.nodes[node_idx].frequency += fre
        if t_idx == len(t):
            return t_idx, node_idx
        for child_idx in self.nodes[node_idx].children:
            if self.nodes[child_idx].name == t[t_idx]:
                return self.match(t, t_idx + 1, child_idx, fre)

        return t_idx, node_idx

    def node_num(self):
        return len(self.nodes)

    def add_node(self, t, t_idx, node_idx, fre=None):
        '''
        将itemset t中的第t_idx及其之后的item添加到node_idx上
        :param t: itemset
        :param t_idx: t中要插入的第一个item的索引
        :param node_idx: 在FP树的node_idx节点上进行插入
        :param fre: 若fre为None，则新创建的节点的frequency为1，否则frequency为fre
        :return:
        '''
        for idx in range(t_idx, len(t)):
            node = self.Node(t[idx])
            if fre is not None:
                node.frequency = fre
            node.parent = node_idx
            now_idx = self.node_num()
            self.nodes[node_idx].children.append(now_idx)
            self.nodes.append(node)
            node_idx = now_idx
            if self.headtables.get(t[idx]) is None:
                self.headtables[t[idx]] = now_idx
            else:
                nextit = self.headtables[t[idx]]
                while True:
                    temp = nextit
                    nextit = self.nodes[nextit].nextit
                    if nextit is None:
                        self.nodes[temp].nextit = now_idx
                        break

    def create_tree(self, database, is_fre):
        '''
        由数据库database创建FP树
        :param database: 数据库
        :param is_fre: 此字段是为了让create_tree函数既能创建普通的FP树，又能创建条件模式树而添加的
        :return:
        '''
        for t in database:
            fre = None
            t_prime = t
            if is_fre is True:
                fre = t[-1]
                t_prime = t[0: -1]
            t_idx, node_idx = self.match(t_prime, 0, 0, fre)
            self.add_node(t_prime, t_idx, node_idx, fre)


def rm_infre_item(pattern_bases, min_sup):
    '''
    移除那些非频繁的item
    :param pattern_bases: 由FP树得到的条件模式基
    :param min_sup:
    :return: 新的条件模式基和它所包含的所有items
    '''
    item_fre_cnt = dict()
    for pattern_base in pattern_bases:
        for j in range(0, len(pattern_base) - 1):
            item = pattern_base[j]
            if item_fre_cnt.get(item) is None:
                item_fre_cnt[item] = pattern_base[-1]
            else:
                item_fre_cnt[item] += pattern_base[-1]

    new_pattern_bases = []
    for pattern_base in pattern_bases:
        new_pattern_base = []
        for j in range(0, len(pattern_base) - 1):
            item = pattern_base[j]
            if item_fre_cnt[item] >= min_sup:
                new_pattern_base.append(item)
        if len(new_pattern_base) > 0:
            new_pattern_base.append(pattern_base[-1])
            new_pattern_bases.append(new_pattern_base)
    pat_list = []
    for key, value in item_fre_cnt.items():
        if value >= min_sup:
            pat_list.append([key, value])
    pat_list.sort(key=functools.cmp_to_key(cmp))

    return new_pattern_bases, pat_list


def gen_pat_base(fptree, head_idx, min_sup):
    '''
    生成head_idx节点关于fptree的所有条件模式基
    :param fptree:
    :param head_idx:
    :param min_sup:
    :return: 条件模式基和它包含的所有items
    '''
    pattern_bases = []
    while head_idx is not None:
        p_idx = fptree.nodes[head_idx].parent
        pattern_base = []
        frequency = fptree.nodes[head_idx].frequency
        while p_idx != 0:
            pattern_base.append(fptree.nodes[p_idx].name)
            p_idx = fptree.nodes[p_idx].parent
        if len(pattern_base) > 0:
            pattern_base.reverse()
            pattern_base.append(frequency)
            pattern_bases.append(pattern_base)
        head_idx = fptree.nodes[head_idx].nextit
    return rm_infre_item(pattern_bases, min_sup)
    pattern_bases, pat_list = rm_infre_item(pattern_bases, min_sup)
    return pattern_bases, pat_list


def generate_fre_itemset(pat_tree: FPTree, min_sup, preFix: list, freqList, tree_items):
    '''
    由条件模式树递归的生成所有frequent itemsets
    :param pat_tree: 条件模式树，本质上就是一棵FP树
    :param min_sup:
    :param preFix: 前缀列表，前n-1个元素为前缀，第n个
    :param freqList: 生成的所有frequent itemsets，每个itemset是一个list对象，此对象的最后一个元素为支持度
    :param tree_items: pat_tree包含的所有items及其频数
    :return:
    '''
    tree_items.reverse()

    for basePat in tree_items:
        newFreqList = preFix.copy()
        frequency = min(basePat[1], preFix[-1])
        if frequency >= min_sup:
            newFreqList.insert(0, basePat[0])
            newFreqList[-1] = frequency
            freqList.append(newFreqList)
            pattern_bases, pat_items = gen_pat_base(pat_tree, pat_tree.headtables[basePat[0]], min_sup)
            if len(pat_items) > 0:
                new_pat_tree = FPTree()
                new_pat_tree.create_tree(pattern_bases, True)
                generate_fre_itemset(new_pat_tree, min_sup, newFreqList, freqList, pat_items)


def fp_tree_method(min_sup, database, itemsets):
    t_start = time.time()
    # tree_items中的每个元素是一个包含两个元素的list对象，第一个元素为item，第二个元素为此item的frequency
    tree_items = find_frequent_1_itemsets(database, itemsets, min_sup)
    tree_items.sort(key=functools.cmp_to_key(cmp))  # 按照frequency进行降序排序
    database = sort_frequent_item(database, tree_items)
    fptree = FPTree()
    fptree.create_tree(database, False)
    freq_list = []
    generate_fre_itemset(fptree, min_sup, [100000], freq_list, tree_items)
    t_end = time.time()
    print('FPTree用时：', t_end - t_start)

    return freq_list

