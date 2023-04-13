import os
import sys
import re


class Gate:
    def __init__(self, inputs):
        super(Gate, self).__init__()
        self.inputs = inputs

    def show(self):
        print(self.output, self.gate_class, self.inputs)


def parse_verilog():
    gate_name = ''
    inputs = []

    with open(verilog_path, 'r') as f:
        for line in f.readlines():
            if re.search(' g_', line):
                parts = line.split()
                gate_name = parts[1][2:]
                inputs = [parts[i].replace('(', '').replace(',', '') for i in range(2, len(parts) - 1)]
                inputs.append(parts[-1][:-2])
                dict[gate_name] = Gate(inputs)

        for keys, values in dict.items():
            print(keys, values.inputs, end=' ')


def parse_diag_file(diag_file):
    path = os.path.join(diag_dir, diag_file)
    begin = False
    cand_list = []
    B1, B2, B3, B4 = [], [], [], []
    p1, p2, p3, cur_p = 0.0, 0.0, 0.0, 0.0
    with open(path, 'r') as f:
        for line in f.readlines():
            if re.match(r' Defect', line):
                p1, p2, p3 = 0.0, 0.0, 0.0
            if re.match(r' match', line):
                begin = True
                cur_p = float(line.split('%')[0].split('=')[-1])
                if cur_p >= p1 or len(B1) == 0:
                    p1 = cur_p
                elif cur_p >= p2 or len(B2) == 0:
                    p2 = cur_p
                elif cur_p >= p3 or len(B3) == 0:
                    p3 = cur_p
            elif re.match(r' ------', line):
                begin = False
            elif begin:
                parts = line.split()
                if len(parts) < 4 or re.search(r'DFF', parts[2]) or re.search(r'MUX', parts[2]):
                    continue
                stem = parts[2].split('/')[0]
                if re.match(r'g_', stem):
                    stem = stem[2:]
                if len(parts[2].split('/')) == 1:
                    cand = stem
                else:
                    branch = parts[2].split('/')[1]
                    if branch == 'ZN' or branch == 'Z' or branch == 'Y':
                        cand = stem
                    elif branch == 'A' or branch == 'A1':
                        cand = dict[stem].inputs[0] + '->' + stem
                    elif branch == 'B' or branch == 'A2':
                        cand = dict[stem].inputs[1] + '->' + stem
                    elif branch == 'A3' or branch == 'C':
                        cand = dict[stem].inputs[2] + '->' + stem
                    elif branch == 'A4' or branch == 'D':
                        cand = dict[stem].inputs[3] + '->' + stem
                if cur_p >= p1:
                    B1.append(cand)
                elif cur_p >= p2:
                    B2.append(cand)
                elif cur_p >= p3:
                    B3.append(cand)
                else:
                    B4.append(cand)
        f.close()
    mf_id = int(diag_file.split('.')[0]) - 1

    for mf in fault_list[mf_id]:
        if mf in B1:
            datas[0] += 1
            index[0].append(diag_file.split('.')[0])
            return
        elif mf in B2:
            datas[1] += 1
            test_set.append(diag_file.split('.')[0])
            index[1].append(diag_file.split('.')[0])
            return
        elif mf in B3:
            datas[2] += 1
            index[2].append(diag_file.split('.')[0])
            test_set.append(diag_file.split('.')[0])
            return
        elif mf in B4:
            index[3].append(diag_file.split('.')[0])
            test_set.append(diag_file.split('.')[0])
            datas[3] += 1
            return
        else:
            return


def parse_faults(fault_path):
    with open(fault_path, 'r') as f:
        for line in f.readlines():
            parts = line.split(',')
            if fault == 'ssl':
                true_fault = [line.split(',')[0].split('(')[-1]]
            elif fault == 'msl':
                parts = line.split(' + ')
                true_fault = []
                for p in parts:
                    true_fault.append(p.split(',')[0].split('(')[-1])
            elif fault == 'fe':
                true_fault = [parts[-3].split('(')[-1]]
            else:
                true_fault = [parts[2].split('(')[-1], parts[6].split('(')[-1]]
            fault_list.append(true_fault)


chip = sys.argv[1]
fault = sys.argv[2]

root = '../../'
diag_dir = os.path.join(root, chip, 'toolB', fault, 'diagnosis_report')
faults_dir = os.path.join(root, chip, 'toolB', fault, chip + '.faults')

verilog_path = os.path.join(root, chip, chip + '.v')
datas = [0, 0, 0, 0]
index = [[] for i in range(4)]
fault_list = []
dict = {}
circuit_inputs = []
test_set = []

parse_faults(faults_dir)

print(verilog_path)
parse_verilog()

for diag_file in os.listdir(diag_dir):
    parse_diag_file(diag_file)
print("\n**********************")
print(datas)
print(test_set)
with open(chip + '-' + fault + '.index', 'w') as f:
    f.write(' '.join(test_set))
