import numpy as np
import pandas as pd
import itertools
import torch
from torchvision import transforms as T
import matplotlib.pyplot as plt
import pickle

THRES = 2000
DIM = 3
ITER = 400
STEP = 0.00005

dataset = ()

def init_dataset():
    global dataset
    chunks = pd.read_csv('dota-2-matches/match.csv', sep=',', skiprows=0, chunksize = 50)
    chunks = itertools.takewhile(lambda chunk: int(chunk['match_id'].iloc[-1]) < THRES, chunks)
    d_match = pd.concat(chunks)

    chunks = pd.read_csv('dota-2-matches/player_time.csv', sep=',', skiprows=0, chunksize = 50)
    chunks = itertools.takewhile(lambda chunk: int(chunk['match_id'].iloc[-1]) < THRES, chunks)
    d_time = pd.concat(chunks)

    print d_match.columns
    print d_match.shape
    print d_time.columns
    print d_time.shape

    dataset = (d_match, d_time)
    return dataset

def range_count(start, count, step):
    return range(start, start + step * count, step)

W = 3
def generate_match(matchid):
    global dataset
    d_match, d_time = dataset
    all_gold1 = d_time.loc[d_time['match_id'] == matchid].ix[:, range_count(2, 5, 3)].sum(axis = 1)
    all_lh1 = d_time.loc[d_time['match_id'] == matchid].ix[:, range_count(3, 5, 3)].sum(axis = 1)
    all_xp1 = d_time.loc[d_time['match_id'] == matchid].ix[:, range_count(4, 5, 3)].sum(axis = 1)
    all_gold2 = d_time.loc[d_time['match_id'] == matchid].ix[:, range_count(2+15, 5, 3)].sum(axis = 1)
    all_lh2 = d_time.loc[d_time['match_id'] == matchid].ix[:, range_count(3+15, 5, 3)].sum(axis = 1)
    all_xp2 = d_time.loc[d_time['match_id'] == matchid].ix[:, range_count(4+15, 5, 3)].sum(axis = 1)
    delta_gold = all_gold1 - all_gold2
    delta_lh = all_lh1 - all_lh2
    delta_xp = all_xp1 - all_xp2

    # radiant_win = d_match[d_match['match_id'] == matchid].ix[0, 9]
    radiant_win = d_match[d_match['match_id'] == matchid].loc[matchid, 'radiant_win']
    (R, ) =  delta_gold.shape # whose type is Series
    # xs = pd.DataFrame(columns = map(lambda i: "In" + str(i), range(3 * W)))
    # ys = pd.DataFrame(columns = ["T"])
    xs = pd.DataFrame()
    ys = pd.DataFrame()
    for i in xrange(0, R - W):
        x = pd.concat([delta_gold[i:i+W], delta_lh[i:i+W], delta_xp[i:i+W]])
        x = x.reset_index(drop=True) # Remove original index
        y = 1 if radiant_win else 0
        y = float(y)
        xs = xs.append(x, ignore_index=True)
        ys = ys.append(pd.Series(y), ignore_index=True)

    return xs, ys

def norm_mat_min(Xs):
    C = Xs.min()
    Range = Xs.max() - C
    Range = Range.mask(Range == 0, Xs.max())
    assert not np.any(Range.values == 0)
    Xs = (Xs - C) / Range
    return Xs, C, Range

def norm_mat_mean(Xs):
    C = Xs.mean()
    Range = Xs.max() - C
    Range = Range.mask(Range == 0, Xs.max())
    assert not np.any(Range.values == 0)
    Xs = (Xs - C) / Range
    return Xs, C, Range


def make_dataset(S, T):
    Xs = pd.DataFrame()
    Ys = pd.DataFrame()
    for i in xrange(S, T):
        print "MAKE MATCH", i
        (xs, ys) = generate_match(i)
        Xs = pd.concat([Xs, xs])
        Ys = pd.concat([Ys, ys])

    # Normalize
    Xs, C, Range = norm_mat_mean(Xs)
    return Xs, Ys, C, Range

# https://blog.csdn.net/m0_37306360/article/details/79307818

class LR(torch.nn.Module):
    def __init__(self, in_dim, out):
        super(LR, self).__init__()
        self.linear = torch.nn.Linear(in_dim, out)
        self.sigmoid = torch.nn.Sigmoid()
 
    def forward(self, x):
        y_pred = self.sigmoid(self.linear(x))
        return y_pred

def Train(MAXN):
    Xs, Ys, C, Range = make_dataset(0, MAXN)

    loss_map = np.array([])
    model = LR(DIM * W, 1)
    criterion = torch.nn.BCELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr = STEP)

    x_data = torch.tensor(np.array(Xs)).type('torch.FloatTensor')
    y_data = torch.tensor(np.array(Ys)).type('torch.FloatTensor')
    print x_data.size()
    print y_data.size()

    for epoch in range(ITER):
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)
        print "====>", epoch, loss.data
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_map = np.append(loss_map, loss.data)

    torch.save(model.state_dict(), 'checkpoint/lr_xgl.pkl')
    pickle.dump([C, Range], open('checkpoint/lr_xgl.norm', "w"))
    pickle.dump(loss_map, open('checkpoint/lr_xgl.loss', "w"))
    plt.plot(loss_map)
    plt.savefig("loss_lr_xgl.png")
    plt.close()

def test_match(matchid):
    model = LR(DIM * W, 1)
    model.load_state_dict(torch.load('checkpoint/lr_xgl.pkl'))
    [C, Range] = pickle.load(open('checkpoint/lr_xgl.norm', "r"))
    TXs, Tys = generate_match(matchid)
    TXs = (TXs - C) / Range
    tx_data = torch.tensor(np.array(TXs)).type('torch.FloatTensor')
    ty_data = torch.tensor(np.array(Tys)).type('torch.FloatTensor')
    y_pred = model(tx_data)

    yT = ty_data.detach().numpy()
    yO = y_pred.detach().numpy()
    yO = np.vectorize(lambda x: 1 if x > 0.5 else 0)(yO)
    yT = yT.astype(np.int64)
    yO = yO.astype(np.int64)
    loss = (yT == yO).astype(np.int64)
    res = np.concatenate((yT, yO, loss), axis=1)
    return loss

def add_vec(a, b):
    if len(a) < len(b):
        c = b.copy()
        c[:len(a)] += a
        return c
    else:
        c = a.copy()
        c[:len(b)] += b
        return c

def test(S, E):
    toth = np.array([0]).reshape((-1, ))
    totl = np.array([0]).reshape((-1, ))
    for i in xrange(S, E):
        print "PREDICT", i
        l = test_match(i)
        l = l.reshape((-1, ))
        totl = add_vec(l, totl)
        toth = add_vec(toth, np.ones(l.shape))

    percent = totl.astype(np.float64) / toth.astype(np.float64)
    plt.plot(percent)
    pickle.dump(percent, open('checkpoint/lr_xgl.percent', 'w'))
    plt.savefig("res_lr_xgl.png")
    plt.close()
    return percent

if __name__ == '__main__':
    init_dataset()
    ITER = 4000
    print "dataset", dataset
    Train(1700)
    test(1700, 2000)
