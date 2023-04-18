import torch as th
import torch.nn as nn
import random
import os
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv
import argparse
import time
from my_utils import evaluate_seperate


class GNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, drop_rate=0.2):
        super(GNN, self).__init__()

        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, data, name=''):
        h = data.x
        edge_index = data.edge_index

        h = self.conv1(h, edge_index)
        h = F.relu(h)
        h = self.conv2(h, edge_index)
        out = self.classify(h)

        return out


def train(model, trainloader, optimizer, criterion, rate, epoch):
    model.train()
    total_loss = 0

    for data in trainloader:
        out = model(data, epoch)
        if rate == -1:
            logits = out[data.train_mask]
            labels = data.y[data.train_mask]
        else:
            all_logits_1 = out[data.mask1]
            all_labels_1 = data.y[data.mask1]
            all_logits_0 = out[data.mask0]
            all_labels_0 = data.y[data.mask0]

            index = [i for i in range(len(all_logits_0))]
            l = len(all_logits_1)
            m = random.sample(index, min(rate * l, len(index)))
            mask = th.tensor(m)

            logits_0 = all_logits_0[mask]
            labels_0 = all_labels_0[mask]

            logits = th.cat((all_logits_1, logits_0), 0)
            labels = th.cat((all_labels_1, labels_0), 0)

        loss = criterion(logits, labels)
        total_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    total_loss = total_loss / len(trainloader)
    return total_loss


def normal_train(args):
    trainpath = f"{args.dataset}-trainset-{args.dir}"
    validpath = f"{args.dataset}-validset-{args.dir}"
    testpath = f"{args.dataset}-testset-{args.dir}"
    valid_set = th.load(validpath)
    train_set = th.load(trainpath)
    test_set = th.load(testpath)

    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    for graph in train_set:
        graph = graph.to(device)
    for graph in test_set:
        graph = graph.to(device)
    for graph in valid_set:
        graph = graph.to(device)

    fold = 0
    res = []
    epoch_list, acc_list, f1_list, auc_list = [], [], [], []
    print("len of train: ", len(train_set), "len of valid: ", len(valid_set), "len of test: ", len(test_set))

    model = GNN(in_dim=args.feat_num, hidden_dim=args.hidden, n_classes=2)
    model = model.to(device)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    trainloader = DataLoader(train_set, batch_size=args.bs, shuffle=True)
    testloader = DataLoader(test_set, batch_size=1, shuffle=True)
    validloader = DataLoader(valid_set, batch_size=1, shuffle=True)

    w = th.tensor([1, args.w1], dtype=th.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=w)

    model_dir = './model/' + args.dataset + '/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    train_time = 0.0
    for i in range(1, args.epochs + 1):
        old_time = time.process_time()
        train_loss = train(model, trainloader, optimizer, criterion, args.r, i)
        train_time += time.process_time() - old_time
        if i % 20 == 0:
            acc, precision, recall, f1_score, pos, auc, rep_R, count, valid_time, count_s, f_beta_reall05, f_beta_reall2 = evaluate_seperate(model, validloader)
            print(f"""Epoch: {i:03d}, \
Count: {count:.4f}, \
DR: {pos:.4f}, \
DR_tool: {rep_R:.4f}, \
F2 Score: {f_beta_reall2:.4f}, \
AUC: {auc:.4f}, \
Time: {train_time:.4f} s""")
            
    th.save(model, model_dir + 'sage-best.pkl')

    acc, precision, recall, f1_score, pos, auc, rep_R, count, test_time, count_s, f_beta_reall05, f_beta_reall2 = evaluate_seperate(model, testloader)
    print(f"""Count: {count:.4f}, \
DR: {pos:.4f}, \
DR_tool: {rep_R:.4f}, \
F2 Score: {f_beta_reall2:.4f}, \
AUC: {auc:.4f}, \
train_time: {train_time:.4f} s, \
prediction_time: {test_time:.4f} s""")
    print("------------------")
    print(f"#Chips: {len(test_set)}, \
#Identified: {round(len(test_set)*count)}, \
Ratio: {count:.4f}, \
DR: {pos:.4f}, \
DR_tool: {rep_R:.4f}, \
Imp: {round(rep_R.item()/pos.item(), 2)}")
    res.append(
        [precision, recall, f1_score, acc, auc, pos.item(), rep_R.item(), count, train_time, test_time, f_beta_reall05,
         f_beta_reall2, count_s])

    model_path = os.path.join(model_dir, args.dataset + '-model.pkl')
    th.save(model, model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='correction')
    parser.add_argument("--epochs", type=int, default=800, help="training epochs")
    parser.add_argument("--hidden", type=int, default=32, help="number of hidden gcn units")
    parser.add_argument("--r", type=int, default=10, help="rate when sampling 0 class for computing loss")
    parser.add_argument("--w1", type=int, default=20, help="weight of 0 class and 1 class when computing loss")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--bs", type=int, default=128, help="batchsize")
    parser.add_argument("--feat_num", type=int, default=10, help="feature num")
    parser.add_argument("--len", type=int, default=-1, help="dataset len for dann compare")
    parser.add_argument("--dir", type=str, default="undir", help="dir graph or undir graph")
    parser.add_argument("--dep", type=str, default="dep", help="dep or nodep: need dependency info or not")
    parser.add_argument("--dataset", type=str, default="i2c", help="dataset name")
    args = parser.parse_args()
    print(args)

    normal_train(args)
