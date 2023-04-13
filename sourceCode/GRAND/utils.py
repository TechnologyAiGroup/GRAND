from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score, confusion_matrix, \
    fbeta_score
import torch as th
import csv
import time


# used


def cal_acc_seperate(out, label, mask):
    pred = out.argmax(dim=1)

    softmax = th.nn.Softmax(dim=1)
    f = softmax(out)[mask].cpu()
    target = [k[1].item() for k in f]

    assert len(pred) == len(label)
    label_cpu = label[mask].cpu()
    pred_cpu = pred[mask].cpu()

    precision = precision_score(label_cpu, pred_cpu, zero_division=0)
    recall = recall_score(label_cpu, pred_cpu)
    acc = accuracy_score(label_cpu, pred_cpu)
    tp = 0
    tn, fp, fn, tp = confusion_matrix(label_cpu, pred_cpu).ravel()
    auc = roc_auc_score(label_cpu, target)

    f1 = f1_score(label_cpu, pred_cpu)
    f_beta_reall2 = fbeta_score(label_cpu, pred_cpu, beta=2.0)
    f_beta_reall05 = fbeta_score(label_cpu, pred_cpu, beta=0.5)

    pos = sum(pred[mask])

    return acc, precision, recall, f1, f_beta_reall2, f_beta_reall05, pos, auc, tp


# used
def evaluate_seperate(model, loader):
    model.eval()
    l = len(loader)

    aacc, ap, ar, af1, apos, aauc, rep_R, acount, acount_s, afbeta2, afbeta05 = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    test_time = 0.0
    for data in loader:
        begin_time = time.process_time()
        out = model(data)
        test_time += time.process_time() - begin_time
        acc, precision, recall, f1, f_beta_reall2, f_beta_reall05, pos, auc, tp = cal_acc_seperate(
            out, data.y, data.train_mask)
        aacc += acc
        ap += precision
        ar += recall
        af1 += f1
        ans_num = sum(data.mask1)
        # print(ans_num)
        # assert ans_num==1 or ans_num==2
        apos += pos / ans_num
        if tp > 0:
            acount += 1
        if tp == ans_num:
            acount_s += 1
        aauc += auc
        afbeta2 += f_beta_reall2
        afbeta05 += f_beta_reall05
        if pos > sum(data.train_mask):
            print(pos, sum(data.train_mask))
            exit()
        rep_R += sum(data.train_mask) / ans_num

    return aacc / l, ap / l, ar / l, af1 / l, apos / l, aauc / l, rep_R / l, acount / l, test_time, acount_s / l, afbeta05 / l, afbeta2 / l


# used
def save_res(res, epname):
    for line in res:
        print(line)
    with open('features.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([epname])
        for line in res:
            writer.writerow(line)
        writer.writerow([])
        f.close()
