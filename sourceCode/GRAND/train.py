import matplotlib.pyplot as plt
import torch as th
import torch.nn as nn
import random
import os
import torch.optim as optim
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
from torch_geometric.nn import GraphConv
from torch_geometric.nn import GATConv
from torch_geometric.nn import SAGEConv
import argparse
import time
import re
from utils import save_res,evaluate_seperate

class GNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes,drop_rate=0.2):
        super(GNN, self).__init__()
        if args.layer == 'gcn':
            self.conv1 = GraphConv(in_dim, hidden_dim)
            self.conv2 = GraphConv(hidden_dim, hidden_dim)
        elif args.layer == 'gat':       
            self.conv1 = GATConv(in_dim, hidden_dim,heads=3)
            self.conv2 = GATConv(3*hidden_dim, hidden_dim,heads=1,concat=False)
        elif args.layer == 'sage':
            self.conv1 = SAGEConv(in_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        else:
            print('layer type not supported!')
            exit()
        
        self.classify = nn.Linear(hidden_dim, n_classes)
        
    def forward(self, data, name=''):
  
        h = data.x
        edge_index = data.edge_index
       
        h = self.conv1(h, edge_index)
        h=F.relu(h)
        h = self.conv2(h, edge_index)
        out = self.classify(h)
        
        return out
       
def train(model, trainloader, optimizer, criterion, rate, epoch):

    # rate：计算loss时采样的比例，即0类节点是1类节点的多少倍
    model.train()
    total_loss=0
    
    for data in trainloader: 
        
        out = model(data, epoch)
        # rate为-1不进行采样
        if rate==-1:
            # 网络输出中candidates的部分
            logits=out[data.train_mask]
            # 节点标签中candidates的部分
            labels=data.y[data.train_mask]
        else:
            all_logits_1=out[data.mask1]                        # 所有1类节点的输出
            all_labels_1=data.y[data.mask1]                     # 所有1类节点的标签
            all_logits_0=out[data.mask0]                        # 所有0类节点的输出
            all_labels_0=data.y[data.mask0]                     # 所有0类节点的标签

            index=[i for i in range(len(all_logits_0))] 
            # 1类节点的数量
            l = len(all_logits_1)

            # 进行采样，采样l*rate倍的0类节点
            m=random.sample(index,min(rate*l,len(index)))       
            mask=th.tensor(m)
            
            logits_0=all_logits_0[mask]                         # 采样后0类节点的输出
            labels_0=all_labels_0[mask]                         # 采样后0类节点的标签
            
            logits=th.cat((all_logits_1,logits_0),0)            # 将0类和1类的输出叠加
            labels=th.cat((all_labels_1,labels_0),0)            # 将0类和1类的标签叠加


        loss = criterion(logits,labels)                         # 计算loss
        total_loss+=loss

        optimizer.zero_grad()                                   # Clear gradients.
        loss.backward()                                         # Derive gradients.
        optimizer.step()                                        # Update parameters based on gradients.
        
    total_loss=total_loss/len(trainloader)
    return total_loss

def normal_train(args):
    if re.search('merge', args.dataset):
        dataset_path=f'{args.dataset}-dataset-{args.dir}'
    else:
        dataset_path=f'{args.dataset}-{args.num}-dataset-{args.dir}'
    if args.dep=='nodep':
        dataset_path+='-nodep'
    print('loading data from',dataset_path)
    
    dataset = th.load(dataset_path)
    if args.len!=-1:
        dataset = dataset[:args.len]
    random.seed(68)  
    random.shuffle(dataset)
    # dataset = dataset[:600]
    
    best_count, fold=0, 0
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    for graph in dataset:
        graph=graph.to(device)


    # 把数据集均匀地分成10份，每次取9份作训练和验证，1份作测试
    # 最后取测试集上效果最好的那一份
    kf = KFold(n_splits = 10,shuffle=False,random_state=None)
    
    # 保存每一份测试集上的分类效果，最后输出
    res=[]

    # 遍历每一份
    for train_d,test_d in kf.split(dataset):
        print('fold: ', fold)

        train_and_valid = [dataset[i] for i in train_d]
        test_set = [dataset[i] for i in test_d]
        train_set = train_and_valid[len(test_d):]
        valid_set = train_and_valid[:len(test_d)]
        print("len of train: ",len(train_set),"len of valid: ",len(valid_set),"len of test: ",len(test_set))
        
        model=GNN(in_dim=args.feat_num,hidden_dim=args.hidden,n_classes=2)
        model=model.to(device)
        print(model)
        
        optimizer = optim.Adam(model.parameters(), lr=args.lr,weight_decay=5e-4)

        trainloader=DataLoader(train_set,batch_size=args.bs,shuffle=True)
        testloader=DataLoader(test_set,batch_size=1,shuffle=True)
        validloader=DataLoader(valid_set,batch_size=1,shuffle=True)

        # 计算loss时0类和1类的权重     
        w=th.tensor([1,args.w1],dtype=th.float32).to(device)    
        criterion = nn.CrossEntropyLoss(weight=w) 

        # 保存模型的路径 
        model_dir='./model/'+args.dataset+'/'                   
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # 训练总时间
        train_time = 0.0

        for i in range(1,args.epochs+1):
            old_time = time.process_time()
            train_loss = train(model,trainloader,optimizer,criterion,args.r,i)
            train_time += time.process_time() - old_time
        
            # 进行验证，并输出一堆指标
            if i%20==0:
                acc,precision,recall,f1_score,pos,auc,rep_R,count,valid_time,count_s,f_beta_reall05,f_beta_reall2=evaluate_seperate(model,validloader)
                print(f'Epoch: {i:03d}, acc: {acc:.4f}, prcs: {precision:.4f}, recall: {recall:.4f}, \
f1: {f1_score:.4f}, f2: {f_beta_reall2:.4f}, f05: {f_beta_reall05:.4f}, auc: {auc:.4f}, avg_R: {pos:.4f}, rep_R: {rep_R:.4f}, count: {count:.4f}, Time: {train_time:.4f}s')
              
                if count>best_count:
                    best_count=count
                    th.save(model, model_dir+'sage-best.pkl')
        fold+=1
       
        # 进行测试
        acc,precision,recall,f1_score,pos,auc,rep_R,count,test_time,count_s,f_beta_reall05,f_beta_reall2 = evaluate_seperate(model,testloader)
        print(f'fold: {fold}, acc: {acc:.4f}, prcs: {precision:.4f}, recall: {recall:.4f}, f1: {f1_score:.4f}, f2: {f_beta_reall2:.4f}, \
f05: {f_beta_reall05:.4f}, auc: {auc:.4f}, avg_R: {pos:.4f}, rep_R: {rep_R:.4f}, count: {count:.4f}, train_time: {train_time:.4f}s, test_time: {test_time:.4f}s')

        # 把测试集上的分类结果加到列表里
        res.append([precision,recall,f1_score,acc,auc,pos.item(),rep_R.item(),count,train_time,test_time,f_beta_reall05,f_beta_reall2,count_s])      
        model_path = os.path.join(model_dir, args.dataset+'-fold'+str(fold)+'-model.pkl')
        th.save(model, model_path)
    
    # 输出res到csv文件，便于查看
    save_res(res,args.dataset)



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--lr", type=float, default=0.01,help="learning rate")
    parser.add_argument("--epochs", type=int, default=800,help="training epochs")
    parser.add_argument("--hidden", type=int, default=32,help="number of hidden gcn units")
    parser.add_argument("--r", type=int, default=20,help="rate when sampling 0 class for computing loss")
    parser.add_argument("--bs", type=int, default=128,help="batchsize")
    parser.add_argument("--feat_num", type=int, default=10,help="feature num")
    parser.add_argument("--len", type=int, default=-1,help="dataset len for dann compare")
    parser.add_argument("--w1", type=int, default=10,help="weight of 0 class and 1 class when computing loss")
    parser.add_argument("--dir", type=str, default="undir",help="dir graph or undir graph")   
    parser.add_argument("--dep", type=str, default="dep",help="dep or nodep: need dependency info or not")
    parser.add_argument("--dataset", type=str, default="adder-merge",help="dataset name")
    parser.add_argument("--layer", type=str, default="sage",help="gat,sage or gcn")
    parser.add_argument("--num", type=int, default=5, help="min num of candidates")
    args = parser.parse_args()
    print(args)
    assert args.dir in ['dir', 'undir']
    assert args.dep in ['dep', 'nodep']
    
    normal_train(args)
    