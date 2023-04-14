import re
import os
import copy
import torch as th
from torch_geometric.data import Data
import numpy as np
import random
import networkx as nx
import argparse
from collections import defaultdict

depnum, deplevel = 0, 0


class Gate:
    inputs = []
    outputs = []

    def __init__(self):
        self.inputs = []
        self.outputs = []

    def push_input(self, cur_input):
        self.inputs.append(cur_input)

    def push_output(self, cur_output):
        self.outputs.append(cur_output)

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs


class Candidate:
    cand_id = 0
    coh = 0
    incoh = 0
    neighbors = []
    dep_num = 0
    dep_level = 0

    def __init__(self, id, coh, incoh, nei):
        self.cand_id = id
        self.coh = coh
        self.incoh = incoh
        self.neighbors = nei

    def print_cand(self):
        print(self.cand_id)
        print(self.coh, self.incoh)
        print(self.neighbors)


def get_base_dict(bench_path):
    my_dict = {}
    with open(bench_path, 'r') as bench_file:
        for line in bench_file.readlines():
            if re.match(r"OUTPUT", line) or re.match(r"INPUT", line) or re.match(r'#', line) or line == '\n':
                continue
            else:
                nodes = line.split(' ')
                cur_output = nodes[0]
                if cur_output not in my_dict:
                    my_dict[cur_output] = Gate()
                if len(nodes) == 3:  # not
                    target = nodes[2][4:-2]
                    my_dict[cur_output].push_input(target)
                    if target not in my_dict:
                        my_dict[target] = Gate()
                    my_dict[target].push_output(cur_output)
                else:
                    for i in range(2, len(nodes)):
                        if i == 2:
                            l = len(nodes[i].split('(')[0])
                            target = nodes[i][l + 1:-1]
                        elif i == len(nodes) - 1:
                            target = nodes[i][0:-2]
                        else:
                            target = nodes[i][0:-1]
                        my_dict[cur_output].push_input(target)
                        if target not in my_dict:
                            my_dict[target] = Gate()
                        my_dict[target].push_output(cur_output)
    src_list = []
    tar_list = []
    graph_inputs = []
    graph_outputs = []
    node_id = {}
    id_ = 0

    for keys, values in my_dict.items():
        node_id[keys] = id_
        id_ += 1
        if len(values.outputs) > 1:
            for i in range(len(values.outputs)):
                s = keys + '->' + values.outputs[i]
                if s not in node_id:
                    node_id[s] = id_
                    id_ += 1

    for keys, values in my_dict.items():
        if len(values.get_inputs()) == 0:
            graph_inputs.append(node_id[keys])
        if len(values.get_outputs()) == 0:
            graph_outputs.append(node_id[keys])
        if len(values.outputs) == 1:
            src_list.append(node_id[keys])
            tar_list.append(node_id[values.outputs[0]])
        elif len(values.outputs) > 1:
            for t in values.outputs:
                src_list.append(node_id[keys])
                tar_list.append(node_id[keys + '->' + t])
                src_list.append(node_id[keys + '->' + t])
                tar_list.append(node_id[t])

    return src_list, tar_list, graph_inputs, graph_outputs, my_dict, node_id


def embedding(cur_node_feat, candidates, node_dep, dict, train_mask, mask0):
    cand_num = len(candidates)

    for cand in candidates:
        id = dict[cand.cand_id]

        train_mask[dict[cand.cand_id]] = True
        mask0[dict[cand.cand_id]] = True

        cur_node_feat[id][5] = 0
        cur_node_feat[id][6] = 0
        cur_node_feat[id][4] = 1

        cur_node_feat[id][5] += cand.coh
        cur_node_feat[id][6] += cand.incoh

    no_dep_feat = copy.deepcopy(cur_node_feat)
    for i in range(len(cur_node_feat)):
        cur_node_feat[i][7] = cand_num
        no_dep_feat[i][7] = cand_num
        no_dep_feat[i][8] = 0
        no_dep_feat[i][9] = 0
        cur_node_feat[i][8] = node_dep[i][0]
        cur_node_feat[i][9] = node_dep[i][1]

    return cur_node_feat, no_dep_feat


def get_ans_index(ans_path, name):
    if re.search(r"and", name) or re.search(r"msl", name) or re.search(r"or", name):
        with open(ans_path, 'r') as f:
            anslist = []
            if "msl" in name:
                anss = f.readline().split(" + ")
                for i in range(len(anss)):
                    curans = anss[i]
                    if i == 0:
                        curans = curans[1:]
                    elif i == len(anss) - 1:
                        curans = curans[:-1]
                    line = curans.split(',')
                    anslist.append(line[0][1:])
            elif "or" in name or "and" in name:
                line = f.readline().split(',')
                anslist.append(line[0][2:])
                anslist.append(line[2].split('(')[-1])
        f.close()
    else:
        exit()
    return anslist


def cal_dep(ori_candidates, cur, node, visited):
    global depnum, deplevel
    if node not in ori_candidates or len(ori_candidates[node]) == 0 or depnum > 5000:
        deplevel = max(deplevel, cur)
        depnum += 1
        return
    else:
        res = 0
        for nei in ori_candidates[node]:
            if not visited.get(nei, False):
                visited[nei] = True
                cal_dep(ori_candidates, cur + 1, nei, visited)
                visited[nei] = False


def get_obv(my_dict, node_num, node_id):
    inputs, outputs = [], []
    graph = nx.DiGraph()
    no_virtual_node_id = {}
    cur_id = 0
    for node in my_dict:
        if node not in no_virtual_node_id:
            if len(my_dict[node].inputs) == 0:
                inputs.append(cur_id)
            if len(my_dict[node].outputs) == 0:
                outputs.append(cur_id)
            no_virtual_node_id[node] = cur_id
            cur_id += 1
    for i in range(cur_id):
        graph.add_node(i)
    for node in my_dict:
        for op in my_dict[node].outputs:
            graph.add_edge(no_virtual_node_id[node], no_virtual_node_id[op])

    no_virtual_ctrl = [0 for i in range(cur_id)]
    no_virtual_obv = [1000 for i in range(cur_id)]

    all_pairs_dis = dict(nx.all_pairs_bellman_ford_path_length(graph))
    for i in range(cur_id):
        for ip in inputs:
            if i in all_pairs_dis[ip]:
                no_virtual_ctrl[i] = max(
                    no_virtual_ctrl[i], all_pairs_dis[ip][i])
        for op in outputs:
            if op in all_pairs_dis[i]:
                no_virtual_obv[i] = min(
                    no_virtual_obv[i], all_pairs_dis[i][op])
    ctrl = [0 for i in range(node_num)]
    obv = [1000 for i in range(node_num)]

    for keys, values in node_id.items():
        ori_node = keys.split('-')[0]
        ctrl[values] = no_virtual_ctrl[no_virtual_node_id[ori_node]]
        obv[values] = no_virtual_obv[no_virtual_node_id[ori_node]]
    return obv, ctrl


def judge(index, candidates, node_to_node):
    if len(index) == 0:
        return False
    for i in range(len(index) - 1):
        if node_to_node == candidates[index[i]].cand_id:
            return True
    return False


def build_dataset(bench_path, cand_dir, ans_dir, args, begin_file_id=0):
    name = args.dataset
    no_file_num, no_cand_file_num, less_than_1, only_1_class = 0, 0, 0, 0
    print(name)
    with open(args.dataset + '.index') as f:
        line = f.readline()
        testss = line.split(' ')

    begin, end, inputs, outputs, my_dict, dict = get_base_dict(bench_path)

    rbegin, rend = begin + end, end + begin

    node_num = len(dict)
    with open('dict.txt', 'w') as f:
        f.truncate()
        f.close()
    with open('dict.txt', 'a') as f:
        for keys, values in dict.items():
            f.write(keys + ' ' + str(values) + '\n')
        f.close()

    if os.path.exists(saved_obv_path):
        obv, depths = th.load(saved_obv_path)
    else:
        obv, depths = get_obv(my_dict, node_num, dict)
        th.save([obv, depths], saved_obv_path)
        print("loading obv info")
    if args.dir == 'dir':
        temp_edge = [begin, end]
    else:
        temp_edge = [rbegin, rend]
    edges = th.tensor(temp_edge, dtype=th.long)

    node_feat = np.zeros((node_num, 10))

    for i in range(node_num):
        node_feat[i][0] = depths[i]
        node_feat[i][1] = obv[i]
        node_feat[i][5] = random.random() / 10
        node_feat[i][6] = random.random() / 10
        node_feat[i][8] = random.random() / 10
        node_feat[i][9] = random.random() / 10

    dataset, test_set = [], []

    for dir in os.listdir(cand_dir):
        file_id = dir.split('_')[-1]

        cand_path = os.path.join(cand_dir, name + '__' + str(file_id), '0.cover-R')
        if not os.path.exists(cand_path):
            pos = name.find('-')
            new_name = name[:pos] + '.' + name[pos + 1:]
            cand_path = os.path.join(
                cand_dir, new_name + '__' + str(file_id), '0.cover-R')

        ans_path = os.path.join(ans_dir, str(file_id) + '.mf')

        if os.path.exists(ans_path):
            true_fault = get_ans_index(ans_path, name)
        else:
            continue
        candidates = []
        ori_candidates = defaultdict(list)

        failing_states = {}
        failing, passing = False, False
        val, incoh, nei_num = 0, 0, 0
        got_ans = False
        nei_list = []
        index = []
        used = {}
        if not os.path.exists(cand_path):
            continue
        with open(cand_path, 'r') as f:
            for line in f:
                if re.match(r"site", line):
                    node_to_node = line.split()[-1]
                    src_name = node_to_node.split('-')[0]
                    if (node_to_node in true_fault) and not judge(index, candidates, node_to_node):
                        index.append(len(candidates))
                        got_ans = True

                elif re.match(r"neighbor", line):
                    tar_name = line.split()[2:]
                    for tar in tar_name:
                        nei_list.append(tar)

                    nei_list.sort()
                    if src_name in used and used[src_name] == nei_list:
                        if (node_to_node not in true_fault) or judge(index, candidates, node_to_node):
                            repeat = True
                        else:
                            fake_ans_index = 0
                            for i in range(len(candidates)):
                                if candidates[i].cand_id.split('-')[0] == src_name and nei_list == candidates[
                                    i].neighbors:
                                    fake_ans_index = i
                                    break
                            candidates.pop(fake_ans_index)
                            for ip in range(len(index)):
                                if index[ip] > fake_ans_index:
                                    index[ip] -= 1

                    else:
                        used[src_name] = nei_list

                elif re.match(r"failing-states", line) and not repeat:
                    passing = False
                    failing = True
                    continue
                elif re.match(r"passing-states", line) and not repeat:
                    passing = True
                    failing = False
                    continue
                elif re.match(r"==", line):
                    if incoh > 0 or val > 0:
                        if node_to_node in dict:
                            cand = Candidate(
                                node_to_node, val, incoh, nei_list)
                        else:
                            cand = Candidate(src_name, val, incoh, nei_list)
                        candidates.append(cand)

                        for nei in nei_list:
                            if nei in my_dict[src_name].get_inputs():
                                cand_name = nei + '->' + src_name
                                if len(my_dict[nei].get_outputs()) == 1:
                                    cand_name = nei
                                if node_to_node in dict:
                                    ori_candidates[node_to_node].append(cand_name)
                                else:
                                    ori_candidates[src_name].append(cand_name)
                            else:
                                for output in my_dict[src_name].get_outputs():
                                    if nei in my_dict[output].get_inputs():
                                        cand_name = nei + '->' + output
                                        if len(my_dict[nei].get_outputs()) == 1:
                                            cand_name = nei
                                        if node_to_node in dict:
                                            ori_candidates[node_to_node].append(
                                                cand_name)
                                        else:
                                            ori_candidates[src_name].append(
                                                cand_name)
                    if len(index) >= 1 and index[-1] >= len(candidates):
                        index.pop()

                    failing = False
                    passing = False
                    nei_list = []
                    repeat = False
                    failing_states = {}
                    incoh = 0
                    val = 0
                if failing and line[0].isdigit() and not repeat:
                    state = line.split()
                    failing_states[state[0]] = int(state[2])
                    val += int(state[2])
                elif passing and line[0].isdigit() and not repeat:
                    state = line.split()
                    if state[0] in failing_states:
                        val -= failing_states[state[0]]
                        incoh += int(state[2]) + failing_states[state[0]]
                    else:
                        val += int(state[2])
            f.close()
        if not got_ans:
            no_cand_file_num += 1
            continue
        if len(candidates) <= args.num:
            less_than_1 += 1
            continue
        for p in index:
            if candidates[p].cand_id not in true_fault:
                break

        node_dep = np.zeros((node_num, 2))

        global depnum, deplevel

        if args.dep == 'dep':
            for cand_1 in ori_candidates.keys():
                visited = {}
                depnum = 0
                deplevel = 0
                cal_dep(ori_candidates, 0, cand_1, visited)
                node_dep[dict[cand_1]][0] += deplevel
                node_dep[dict[cand_1]][1] += depnum

        train_mask = [False for i in range(node_num)]
        label_mask1 = [False for i in range(node_num)]
        label_mask0 = [False for i in range(node_num)]

        cur_node_feat = copy.deepcopy(node_feat)
        cur_node_feat, no_dep_feat = embedding(
            cur_node_feat, candidates, node_dep, dict, train_mask, label_mask0)

        for id in index:
            ans = candidates[id]
            label_mask0[dict[ans.cand_id]] = False
            label_mask1[dict[ans.cand_id]] = True
        if sum(label_mask1) == 0:
            only_1_class += 1
            continue

        mask = th.tensor(train_mask)
        mask0 = th.tensor(label_mask0)
        mask1 = th.tensor(label_mask1)

        temp_y = np.zeros(node_num)
        for true_pos in index:
            temp_y[dict[candidates[true_pos].cand_id]] = 1
        y = th.tensor(temp_y, dtype=th.long)

        if args.dep == "dep":
            x = th.tensor(cur_node_feat, dtype=th.float32)
            graph = Data(x=x, edge_index=edges, y=y,
                         train_mask=mask, mask0=mask0, mask1=mask1)
        else:
            x = th.tensor(no_dep_feat, dtype=th.float32)
            graph = Data(x=x, edge_index=edges, y=y, train_mask=mask, mask0=mask0, mask1=mask1)

        if file_id in testss:
            test_set.append(graph)
        else:
            dataset.append(graph)

    dataset_path = f"{name}-{str(args.num)}-dataset-{args.dir}"
    if args.dep == 'nodep':
        dataset_path += '-nodep'

    print('len of train', len(dataset), 'len of test', len(test_set), 'saving', args.dataset, 'to', dataset_path)
    th.save([dataset, test_set], dataset_path)
    print("exceptions: ", no_file_num, no_cand_file_num,
          less_than_1, only_1_class)

    return dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--dataset", type=str, default='adder-msl', help="choose dataset")
    parser.add_argument("--dir", type=str, default='undir', help="dir or undir")
    parser.add_argument("--num", type=int, default=5, help="min num of candidates")
    parser.add_argument("--dep", type=str, default='dep', help="dep or nodep")
    args = parser.parse_args()
    print(args)
    [chip, fault] = args.dataset.split('-')
    assert fault in ['msl', 'and', 'or']

    root = os.path.join('../../', chip)
    bench_path = os.path.join(root, chip + '.bench')
    cand_path = os.path.join(root, 'toolB', fault, 'data', args.dataset + '-candidates/')
    ans_path = os.path.join(root, 'toolB', fault, 'data', args.dataset + '/')
    saved_obv_path = os.path.join(args.dataset.split('-')[0] + '.sobv')

    dataset = build_dataset(bench_path, cand_path, ans_path, args)
