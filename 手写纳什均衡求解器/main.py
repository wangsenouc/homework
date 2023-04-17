import re
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection
import os


def write_rst(equilibria, out_path):
    """
    将求得的纳什均衡点写入文件中
    :param equilibria: 所有纳什均衡点
    :param out_path: 要写入的文件路径
    :return:
    """
    with open(out_path, 'wt') as f:
        for eq in equilibria:
            write_content = ''
            for idx, value in enumerate(eq):
                if type(value) is int:
                    value = str(abs(value))
                else:
                    value = str(abs(value))

                if write_content == '':
                    write_content = value
                else:
                    write_content += ',' + value
            f.write(write_content + '\n')


def read_payoff(in_path):
    """
    读取.nfg文件，并将列表形式的收益解析成矩阵形式
    :param in_path: .nfg文件的路径
    :return: 每个玩家矩阵形式的收益，每个玩家可选择的策略数量
    """
    with open(in_path) as fin:
        all_lines = fin.readlines()
        payoff = all_lines[-1].strip().split()
        player_info = all_lines[-3]
    res = re.search('{ (\d+ )*}', player_info)
    strategy_space = [int(num) for num in res.group()[1: -1].split()]
    player_num = len(strategy_space)
    all_payoff = []
    for player in range(player_num):
        payoff_matrix = [int(pay) for idx, pay in enumerate(payoff) if idx % player_num == player]
        payoff_matrix = np.array(payoff_matrix).reshape(strategy_space, order='F')
        all_payoff.append(payoff_matrix)
    return all_payoff, strategy_space


def create_half_spaces(payoff_matrix, is_p=False):
    """
    根据收益矩阵创建half_spaces
    :param payoff_matrix: 收益矩阵，正是根据它来创建half_spaces
    :param is_p: 根据参考书上的算法，polyhedron P和Q中half_spaces的顺序不一样，此变量用于确定是哪一种情况
    :return: 创建好的half_spaces，加入payoff_matrix的维度维d1 * d2，那么half_spaces的维度就是(d1+d2) * (d2 + 1)
    """
    d1, d2 = payoff_matrix.shape  # 收益矩阵的第一维和第二维的大小
    if is_p:
        b = np.append(np.zeros(d2), -np.ones(d1))
        payoff_matrix = np.append(-np.eye(d2), payoff_matrix, axis=0)
    else:
        b = np.append(-np.ones(d1), np.zeros(d2))
        payoff_matrix = np.append(payoff_matrix, -np.eye(d2), axis=0)

    half_spaces = np.column_stack((payoff_matrix, b.transpose()))
    return half_spaces


def find_feasible_point(half_spaces):
    """
    求线性规划空间内部的点，参考了HalfspaceIntersection官方文档，
    地址：https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.HalfspaceIntersection.html?highlight=halfspaceintersection#scipy.spatial.HalfspaceIntersection
    :param half_spaces:
    :return: half_spaces所表示的线性空间的内部的点
    """
    norm_vector = np.reshape(
        np.linalg.norm(half_spaces[:, :-1], axis=1), (half_spaces.shape[0], 1)
    )
    c = np.zeros((half_spaces.shape[1],))
    c[-1] = -1
    A_ub = np.hstack((half_spaces[:, :-1], norm_vector))
    b_ub = -half_spaces[:, -1:]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub)
    return res.x[:-1]


def compute_vertices_and_labels(half_spaces):
    """
    根据half_spaces来计算vertices，需要注意结果中不能含有0向量
    :param half_spaces:
    :return:
    """
    feasible_point = find_feasible_point(half_spaces)
    hs = HalfspaceIntersection(half_spaces, feasible_point)
    hs.close()
    rst = []
    for v in hs.intersections:
        if not np.all(np.isclose(v, 0)):  # 点v不能是(0, 0)，也不能是无穷大（小）
            # 根据Av - b = 0来寻找点v的标记
            b = half_spaces[:, -1]
            A = half_spaces[:, :-1]
            labeled_v = np.where(np.isclose(np.dot(A, v), -b))[0]
            rst.append((v, labeled_v))
    return rst


def mixed_nash_equilibrium(all_payoff):
    """
    计算两个玩家的混合策略纳什均衡（含纯策略）
    :param all_payoff: 两个玩家的收益矩阵
    :return: 求得的所有纯策略纳什均衡点
    """
    payoff_mat1, payoff_mat2 = all_payoff
    # 对收益矩阵平移不会影响纳什均衡点，根据参考书中的说法，当收益矩阵非负且没有全0列时，可以简化掉u和v
    if np.min(payoff_mat1) <= 0:
        payoff_mat1 = payoff_mat1 + (1 - np.min(payoff_mat1))
    if np.min(payoff_mat2) <= 0:
        payoff_mat2 = payoff_mat2 + (1 - np.min(payoff_mat2))

    p1_strategy_num, p2_strategy_num = payoff_mat1.shape  # 玩家1的策略数量和玩家2的策略数量
    all_strategy_num = p1_strategy_num + p2_strategy_num

    p1_half_spaces = create_half_spaces(payoff_mat2.transpose(), is_p=True)
    p2_half_spaces = create_half_spaces(payoff_mat1, is_p=False)
    rst = []
    p_vertices = compute_vertices_and_labels(p1_half_spaces)  # 玩家1的所有顶点，这里用p是想与参考书统一起来，因为参考书用P表示的polyhedron，下同
    q_vertices = compute_vertices_and_labels(p2_half_spaces)
    for vec_x, x_l in p_vertices:  # 点x（以向量的形式表示）和x的标记，分别表示交点的坐标，以及是哪几个不等式相交
        all_labels = np.zeros(all_strategy_num, dtype=int)
        all_labels[x_l] = 1
        for vec_y, y_l in q_vertices:
            all_labels[y_l] += 1
            if np.all(all_labels):
                rst.append(np.append(vec_x / sum(vec_x), vec_y / sum(vec_y)))  # 做归一化
            all_labels[list(y_l)] -= 1
    return rst


def compare_eq(computed_NE, expected_NEs):
    '''
    用于测试求解结果是否正确
    :param computed_NE: 通过程序计算出来的均衡点
    :param expected_NEs: 正确的纳什均衡点
    :return: 如果计算出来的NE是正确的，则返回True，否则返回False
    '''
    for one_NE in expected_NEs:
        diff = np.abs(computed_NE - one_NE)
        if np.sum(diff) <= 1e-4:
            return True
    return False


def pure_nash_equilibrium(all_payoff):
    """
    计算三个及以上的玩家的混合策略纳什均衡
    :param all_payoff: 是一个list，第i个元素是第i个玩家的收益矩阵
    :return: 求得的所有混合策略纳什均衡点
    """
    player_num = len(all_payoff)
    strategy_space = all_payoff[0].shape
    max_payoff_coordinate = set()  # 玩家的最大收益在收益矩阵中的坐标
    for player, payoff_matrix in enumerate(all_payoff):
        # 第i个玩家在第i维上选取动作
        temp_max_payoff_coordinate = set()
        max_pos = np.where(payoff_matrix == np.max(payoff_matrix, axis=player, keepdims=True))
        max_pos = list(max_pos)
        # 因为np.where返回的是tuple，所以需要对其进一步解析才能得到坐标
        for i in range(len(max_pos[0])):
            temp_list = []
            for j in range(player_num):
                temp_list.append(max_pos[j][i])
            temp_max_payoff_coordinate.add(tuple(temp_list))
        if player == 0:
            max_payoff_coordinate = temp_max_payoff_coordinate
        else:
            max_payoff_coordinate = max_payoff_coordinate.intersection(temp_max_payoff_coordinate)

    # 将求得的纯策略纳什均衡转换为指定的输出格式
    equilibria = []
    for equilibrium_coord in max_payoff_coordinate:
        equilibrium = [0] * (sum(strategy_space))
        pre_len = 0
        for pos, strategy_num in zip(equilibrium_coord, strategy_space):
            equilibrium[pre_len + pos] = 1
            pre_len += strategy_num
        equilibria.append(equilibrium)
    return equilibria


def nash(in_path, out_path):
    """
    求解纳什均衡，对于两个玩家的博弈，需要求解所有的纯策略均衡点和混合策略均衡点，对于三个及以上玩家的博弈，只需求解纯策略均衡点
    :param in_path: .nfg文件的路径，次文件包含了与博弈有关的所有信息
    :param out_path: 均衡点输出文件的路径
    :return:
    """
    all_payoff, strategy_space = read_payoff(in_path)
    if len(strategy_space) == 2:
        # 二人博弈，求解混合策略纳什均衡
        equilibria = mixed_nash_equilibrium(all_payoff)
    else:
        # 三人以上博弈，求解纯策略纳什均衡
        equilibria = pure_nash_equilibrium(all_payoff)

    write_rst(equilibria, out_path)


def main():
    for f in os.listdir('input'):
        if f.endswith('.nfg') and not f.startswith('._'):
            nash('input/' + f, 'output2/' + f.replace('nfg', 'ne'))


def test():
    A = np.array([[2, 2], [3, 0], [0, 1]])
    B = np.array([[2, 2], [1, 2], [2, 1]])
    C = (A, B)
    rst = mixed_nash_equilibrium(C)
    print(rst)


if __name__ == '__main__':
    test()
    # main()
