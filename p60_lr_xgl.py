import numpy as np
import pandas as pd
import itertools
import torch
from torchvision import transforms as T
import matplotlib.pyplot as plt
import pickle

THRES = 7000
DIM = 3
ITER = 4000
W = 5
STEP = 0.00005
newl = 60
STRIDE = W

dataset = ()

def init_dataset():
    global dataset
    chunks = pd.read_csv('dota-2-matches/match.csv', sep=',', skiprows=0, chunksize = 50)
    chunks = itertools.takewhile(lambda chunk: int(chunk['match_id'].iloc[-1]) < THRES, chunks)
    d_match = pd.concat(chunks)

    chunks = pd.read_csv('dota-2-matches/player_time.csv', sep=',', skiprows=0, chunksize = 50)
    chunks = itertools.takewhile(lambda chunk: int(chunk['match_id'].iloc[-1]) < THRES, chunks)
    d_time = pd.concat(chunks)

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
    l = len(all_gold1)

    xs = pd.DataFrame()
    radiant_win = d_match[d_match['match_id'] == matchid].loc[matchid, 'radiant_win']
    (R, ) =  delta_gold.shape # whose type is Series
    SHAPE = delta_gold.shape
    assert delta_gold.shape == SHAPE
    assert delta_lh.shape == SHAPE
    assert delta_xp.shape == SHAPE
    assert R == l
    if R <= newl:
        return (None, None)

    CIN = 3
    for x in [delta_gold, delta_lh, delta_xp]:
        x = x.reset_index(drop=True) # Remove original index
        # print "xsssss", x.iloc[0:newl].shape
        xs = xs.append(x.iloc[0:newl], ignore_index=True)
    xs = xs.T
    ys = pd.DataFrame(pd.Series([1.0 if radiant_win else 0.0]))
    finalxs = xs.values.flatten()
    return pd.DataFrame(finalxs), ys

def norm_mat_mean(Xs):
    C = Xs.mean()
    Range = Xs.max() - Xs.min()
    Range = Range.mask(Range == 0, Xs.max())
    assert not np.any(Range.values == 0)
    Xs = (Xs - C) / Range
    return Xs, C, Range


def make_dataset(S, T):
    Xs = pd.DataFrame()
    Ys = pd.DataFrame()
    for i in xrange(S, T):
        (xs, ys) = generate_match(i)
        if xs is None:
            # print "IGNORE MATCH", i
            pass
        else:
            Xs = Xs.append(xs.T, ignore_index = True) 
            Ys = Ys.append(ys.T, ignore_index = True) 

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
    in_len = Xs.shape[1]
    print "Xs", Xs.shape, "Ys", Ys.shape, 'in_len', in_len
    model = LR(in_len, 1)
    criterion = torch.nn.BCELoss(size_average=False)
    optimizer = torch.optim.Adam(model.parameters(), lr = STEP)

    x_data = torch.tensor(np.array(Xs)).type('torch.FloatTensor')
    y_data = torch.tensor(np.array(Ys)).type('torch.FloatTensor')
    print x_data.size()
    print y_data.size()

    for epoch in range(ITER):
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_map = np.append(loss_map, loss.data)

    torch.save(model.state_dict(), 'checkpoint/p60_lr_xgl.pkl')
    pickle.dump([C, Range], open('checkpoint/p60_lr_xgl.norm', "w"))
    pickle.dump(loss_map, open('checkpoint/p60_lr_xgl.loss', "w"))
    plt.plot(loss_map)
    plt.savefig("loss_p60_lr_xgl.png")
    plt.close()

def test_match(matchid):
    [C, Range] = pickle.load(open('checkpoint/p60_lr_xgl.norm', "r"))
    TXs, Tys = generate_match(matchid)
    if TXs is None:
        return None
    in_len = TXs.shape[0]
    print "len", in_len
    model = LR(in_len, 1)
    model.load_state_dict(torch.load('checkpoint/p60_lr_xgl.pkl'))
    assert not np.any(Range.values == 0)
    TXs = TXs.T
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
    tot = 0
    acc = 0
    for i in xrange(S, E):
        l = test_match(i)
        if l is None:
            continue
        tot += 1
        print "l", l
        acc += l[0]
    percent = acc * 1.0 / tot
    print "acc", acc, tot
    pickle.dump(percent, open('checkpoint/p90_lr_xgl.percent', 'w'))
    return percent

if __name__ == '__main__':
    init_dataset()

    TRAIN = 5000
    # Train(TRAIN)
    print test(TRAIN, TRAIN+1000)
    # test(1700, 2000)
