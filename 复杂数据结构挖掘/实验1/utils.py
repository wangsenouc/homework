import re


def load_groceries():
    database = []
    itemset = set()
    filename = 'Groceries.csv'
    with open(filename) as f:
        for line in f.readlines():
            start = line.find('{')
            end = line.find('}')
            if start == -1:
                continue
            line = line[start + 1: end]
            items = line.split(',')
            for item in items:
                itemset.add(item)
            database.append(items)
    return database, list(itemset)


def load_unix_usage():
    database = []
    session = []
    itemset = set()
    argument = ['&', ';', '|', '1', '`']
    for user_id in range(0, 9):
        filename = f'UNIX_usage/USER{0}/sanitized_all.981115184025'.format(user_id)

        with open(filename) as f:
            for line in f.readlines():
                line = line.strip()
                if line == '**SOF**':
                    continue
                if line in argument:
                    continue
                if line[0] == '-' or line[0] == '<' or line[0] == '>' or line[0] == '%' or line[0] == '$':
                    continue
                elif line == '**EOF**':
                    if session:
                        database.append(session.copy())
                        session.clear()
                else:
                    session.append(line)
                    itemset.add(line)
    return database, itemset


def is_cmd(tok):
    return len(tok) > 0 and tok[0] != '-' and re.search('[a-zA-Z]', tok) is not None


def process_sessions(cmd_only=True, no_repeat=True):
    sessions = []
    sess = []
    itemset = set()
    for user_id in range(0, 9):
        filename = f'UNIX_usage/USER{0}/sanitized_all.981115184025'.format(user_id)
        with open(filename) as f:

            for r in f.readlines():
                r = r.strip()
                if r == "**SOF**":
                    # a new session
                    sess = []
                elif r == "**EOF**":
                    # end of session
                    if len(sess) != 0:
                        sessions.append(sess)
                else:
                    if cmd_only and not is_cmd(r):
                        continue
                    if no_repeat and r in sess:
                        continue
                    sess.append(r)
                    itemset.add(r)

    return sessions, itemset




def cmp(x, y):
    if x[1] > y[1]:
        return -1
    if x[1] < y[1]:
        return 1
    if x[0] > y[0]:
        return 1
    elif x[0] < y[0]:
        return -1
    return 0


def find_frequent_1_itemsets(database, itemsets, min_sup):
    L1_candi = dict()
    L1 = list()
    for item in itemsets:
        L1_candi[item] = 0

    for trans in database:
        for item in trans:
            L1_candi[item] += 1
    for key in L1_candi:
        if L1_candi[key] >= min_sup:
            L1.append([key, L1_candi[key]])

    return L1


if __name__ == '__main__':
    load_unix_usage()

