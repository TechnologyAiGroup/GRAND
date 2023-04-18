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
    dict, chip_list = get_chip_list()
    chips = args.dataset
    if args.dir == "undir":
        for chip in chips:
            finals = []
            dict[chip].undir.sort()
            collection = dict[chip].undir
            for data in collection:
                if args.dep == "dep" and re.search("nodep", data):
                    continue
                elif args.dep == "nodep" and not re.search("nodep", data):
                    continue
                if re.search('-(ssl|and|or|msl|dom|fe)(A|B)-', data) and re.search('B', data):
                    print(data)
                    finals += th.load(data)
            print('merge data len:', len(finals))
            th.save(finals, f'{chip}-merge-dataset-undir')
    else:
        for chip in chips:
            finals = []
            dict[chip].dir.sort()
            collection = dict[chip].dir
            for data in collection:
                if args.dep == "dep" and re.search("nodep", data):
                    continue
                elif args.dep == "nodep" and not re.search("nodep", data):
                    continue
                if re.search('-(ssl|and|or|msl|dom|fe)-', data):
                    print(data)
                    finals += th.load(data)
            print('merge data len:', len(finals))
            th.save(finals, f'{chip}-merge-dataset-dir')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=[""], nargs="+")
    parser.add_argument("--dir", type=str, default='undir')
    parser.add_argument("--dep", type=str, default='dep')
    args = parser.parse_args()
    merge_data_of_6_fault_models(args)
