import torch as th
import re
import os
import argparse
from collections import defaultdict


def get_chip_list():
    dict = defaultdict(ChipData)
    chiplist = set()
    for file in os.listdir('./'):
        if re.search(r'dataset', file):
            chip_name = file.split('-')[0]
            fault = file.split('-')[1]
            num = file.split('-')[2]

            if re.search(r'-merge', file) or re.search(r'-em', file) or re.search(r'-tmax', file):
                continue

            if re.search(r'undir', file):
                dict[chip_name].undir_faults.append(fault)
                dict[chip_name].undir.append(file)

            else:
                dict[chip_name].dir_faults.append(fault)
                dict[chip_name].dir.append(file)
        elif re.search(r'candidates', file):
            chiplist.add(file.split('-')[0])
    return dict, chiplist


class ChipData:
    # used
    def __init__(self):
        self.undir = []
        self.dir = []
        self.dir_faults = []
        self.undir_faults = []


def merge_data_of_6_fault_models(args):
    # used
    dict, _ = get_chip_list()
    chip = args.dataset
    dict[chip].undir.sort()
    dict[chip].dir.sort()
    finals = [[], []]
    if args.dir == "undir":
        for data in dict[chip].undir:
            if args.dep == 'dep' and re.search('nodep', data):
                continue
            if args.dep == 'nodep' and (not re.search('nodep', data)):
                continue
            print(data)
            temp = th.load(data)
            finals[0] += temp[0][:1000]
            finals[1] += temp[1]
    elif args.dir == "dir":
        for data in dict[chip].dir:
            if args.dep == 'dep' and re.search(r'nodep', data):
                continue
            if args.dep == 'nodep' and (not re.search(r'nodep', data)):
                continue
            print(data)
            temp = th.load(data)
            finals[0] += temp[0][:1000]
            finals[1] += temp[1]
    if len(finals[0]) != 0:
        print(chip, 'train len:', len(finals[0]), 'test len:', len(finals[1]))
    else:
        print("no dataset in chips")
        exit()
    dataset = finals
    
    test_set = dataset[1]
    trainl = min(int(len(dataset[0])*0.9), 1000)
    train_set = dataset[0][:trainl]
    valid_set = dataset[0][trainl:]
    print("len of train: ", len(train_set), "len of valid: ", len(valid_set), "len of test: ", len(test_set))
    trainpath = f"{args.dataset}-trainset-{args.dir}"
    validpath = f"{args.dataset}-validset-{args.dir}"
    testpath = f"{args.dataset}-testset-{args.dir}"
    if args.dep=='nodep':
        for path in [trainpath, validpath, testpath]:
            path = path+'-nodep'
    th.save(train_set, trainpath)
    th.save(valid_set, validpath)
    th.save(test_set, testpath)
    print("The train, valid, test datasets have saved.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='i2c')
    parser.add_argument("--dir", type=str, default='undir')
    parser.add_argument("--dep", type=str, default='dep')
    args = parser.parse_args()
    assert args.dir in ['dir', 'undir']
    assert args.dep in ['dep', 'nodep']
    merge_data_of_6_fault_models(args)
