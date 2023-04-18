import torch as th
import re
import os
import argparse
import random
from collections import defaultdict
from sklearn.model_selection import KFold


def get_chip_list():
    dict = defaultdict(ChipData)
    chiplist = set()
    for file in os.listdir('./'):
        if re.search(r'dataset', file):
            chip_name = file.split('-')[0]
            fault = file.split('-')[1]
            if re.search(r'-merge', file) or re.search(r'-em', file):
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
    def __init__(self):
        self.undir = []
        self.dir = []
        self.dir_faults = []
        self.undir_faults = []


def merge_data_of_6_fault_models(args):
    assert args.dir in ["dir", "undir"]
    assert args.dep in ["dep", "nodep"]
    assert args.tool in ['A', 'B']
    dict, chip_list = get_chip_list()
    chip = args.dataset
    finals = []
    if args.dir == "undir":
        dict[chip].undir.sort()
        collection = dict[chip].undir
        for data in collection:
            if args.dep == "dep" and re.search("nodep", data):
                continue
            elif args.dep == "nodep" and not re.search("nodep", data):
                continue
            if re.search('-(ssl|and|or|msl|dom|fe)(A|B)-', data) and re.search(f'{args.tool}', data):
                print(data)
                finals += th.load(data)
        print('merge data len:', len(finals))
    else:
        dict[chip].dir.sort()
        collection = dict[chip].dir
        for data in collection:
            if args.dep == "dep" and re.search("nodep", data):
                continue
            elif args.dep == "nodep" and not re.search("nodep", data):
                continue
            if re.search('-(ssl|and|or|msl|dom|fe)(A|B)-', data):
                print(data)
                finals += th.load(data)
        print('merge data len:', len(finals))

    dataset = finals
    if args.tool=='A':
        seed = 44
    elif args.tool=='B':
        seed = 88
    random.seed(seed)
    random.shuffle(dataset)
    kf = KFold(n_splits=10, shuffle=False, random_state=None)
    fold = 0
    for tup in kf.split(dataset):
        if fold==4 and args.tool=='A':
            trainandvalid, test = tup
            break
        elif fold==7 and args.tool=='B':
            trainandvalid, test = tup
            break
        else:
            fold+=1
    train_and_valid = [dataset[i] for i in trainandvalid]
    test_set = [dataset[i] for i in test]
    train_set = train_and_valid[len(test):]
    valid_set = train_and_valid[:len(test)]
    print("len of train: ", len(train_set), "len of valid: ", len(valid_set), "len of test: ", len(test_set))
    trainpath = f"{args.dataset}-trainset-{args.tool}-{args.dir}"
    validpath = f"{args.dataset}-validset-{args.tool}-{args.dir}"
    testpath = f"{args.dataset}-testset-{args.tool}-{args.dir}"
    if args.dep=='nodep':
        for path in [trainpath, validpath, testpath]:
            path = path+'-nodep'
    th.save(train_set, trainpath)
    th.save(valid_set, validpath)
    th.save(test_set, testpath)
    print("The train, valid, test datasets have saved.")
    

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="i2c")
    parser.add_argument("--dir", type=str, default='undir')
    parser.add_argument("--dep", type=str, default='dep')
    parser.add_argument("--tool", type=str, default='B', help='choose tool A or B')
    args = parser.parse_args()
    merge_data_of_6_fault_models(args)
