import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import pickle
import torch
from torchvision import transforms as T
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense
import tensorflow as tf
import torch.nn.functional as F

THRES = 7000
DIM = 3
ITER = 4000
W = 5
STEP = 0.005
newl = 60
STRIDE = W

dataset = ()
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

def init_dataset():
    global dataset
    chunks = pd.read_csv('dota-2-matches/match.csv', sep=',', skiprows=0, chunksize = 200)
    chunks = itertools.takewhile(lambda chunk: int(chunk['match_id'].iloc[-1]) < THRES, chunks)
    d_match = pd.concat(chunks)

    chunks = pd.read_csv('dota-2-matches/player_time.csv', sep=',', skiprows=0, chunksize = 200)
    chunks = itertools.takewhile(lambda chunk: int(chunk['match_id'].iloc[-1]) < THRES, chunks)
    d_time = pd.concat(chunks)

    chunks = pd.read_csv('dota-2-matches/teamfights.csv', sep=',', skiprows=0, chunksize = 200)
    chunks = itertools.takewhile(lambda chunk: int(chunk['match_id'].iloc[-1]) < THRES, chunks)
    d_fight = pd.concat(chunks)

    chunks = pd.read_csv('dota-2-matches/teamfights_players.csv', sep=',', skiprows=0, chunksize = 200)
    chunks = itertools.takewhile(lambda chunk: int(chunk['match_id'].iloc[-1]) < THRES, chunks)
    d_fightpls = pd.concat(chunks)

    chunks = pd.read_csv('dota-2-matches/players.csv', sep=',', skiprows=0, chunksize = 400)
    chunks = itertools.takewhile(lambda chunk: int(chunk['match_id'].iloc[-1]) < THRES, chunks)
    d_players = pd.concat(chunks)

    d_heros = pd.read_csv('dota-2-matches/hero_names.csv', sep=',')

    dataset = (d_match, d_time, d_fight, d_fightpls, d_players, d_heros)
    return dataset

def range_count(start, count, step):
    return range(start, start + step * count, step)

def generate_hero(match_id):
    d_match, d_time, d_fight, d_fightpls, d_players, d_heros = dataset
    len_heros = len(d_heros) + 1

    radiant_win = d_match.loc[d_match['match_id'] == match_id]['radiant_win']
    h = np.array([0] * (len_heros * 2))
    allh1 = d_players.loc[d_players['match_id'] == match_id].loc[lambda r: r['player_slot'] < 5]
    allh2 = d_players.loc[d_players['match_id'] == match_id].loc[lambda r: r['player_slot'] >= 128]

    for items in allh1['hero_id'].iteritems():
        h[items[1]] = 1
    for items in allh2['hero_id'].iteritems():
        h[items[1] + len_heros] = 1
    
    return pd.Series(h), radiant_win

def test_match_prior(matchid):
    d_match, d_time, d_fight, d_fightpls, d_players, d_heros = dataset
    len_heros = len(d_heros) + 1
    model = LR(len_heros * 2, 1)
    model.load_state_dict(torch.load('checkpoint/lr_prior.pkl'))
    [C, Range] = pickle.load(open('checkpoint/lr_prior.norm', "r"))
    TXs, Tys = generate_hero(matchid)
    TXs = (TXs - C) / Range
    tx_data = torch.tensor(np.array(TXs)).type('torch.FloatTensor')
    ty_data = torch.tensor(np.array(Tys)).type('torch.FloatTensor')
    y_pred = model(tx_data)

    yO = y_pred.detach().numpy()
    return yO

def generate_die_trends(match_id, i):
    global dataset
    d_match, d_time, d_fight, d_fightpls, d_players, d_heros = dataset
    this_match = d_fightpls.loc[d_fightpls['match_id']==match_id, ['match_id', 'player_slot', 'deaths']]
    this_fight = this_match.iloc[10*i:10*i+10, :].loc[:, ['deaths']]
    die1, die2 = this_fight.iloc[0:5, :].sum(), this_fight.iloc[5:10, :].sum()
    die_count1 = die1.values[0].item()
    die_count2 = die2.values[0].item()
    arr1 = np.arange(die_count1, 0, -1)
    arr2 = np.arange(die_count2, 0, -1)
    return die_count1, die_count2, arr1, arr2

def generate_deaths(match_id):
    # WARNING: No d_fight.loc[match_id, indexer]
    global dataset
    d_match, d_time, d_fight, d_fightpls, d_players, d_heros = dataset
    indexer = ['match_id', 'deaths']
    get_indexer = indexer + ['start', 'end']
    all_timepoint = d_fight.loc[d_fight['match_id']==match_id, get_indexer]
    all_timepoint['start'] = all_timepoint['start'].apply(lambda x: int(x / 60.0))
    all_timepoint['end'] = all_timepoint['end'].apply(lambda x: int(x / 60.0))
    ts1 = np.array([])
    ts2 = np.array([])
    i = 0
    # IMPORTANT iterrows (i, row), where i is global index
    for index, row in all_timepoint.iterrows():
        # for all fight
        die_count1, die_count2, arr1, arr2 = generate_die_trends(match_id, i)
        len1 = len(arr1); len2 = len(arr2)
        if row['end']+len1 >= len(ts1):
            ts1.resize(row['end']+len1+1)
        if row['end']+len2 >= len(ts2):
            ts2.resize(row['end']+len2+1)
        ts1[row['end']:row['end']+len1] = arr1
        ts2[row['end']:row['end']+len2] = arr2
        i += 1

    # ts1 and ts2 may have mismatched length.
    return ts1, ts2

def generate_hero_ts(match_id):
    global dataset
    d_match, d_time, d_fight, d_fightpls, d_players, d_heros = dataset
    len_heros = len(d_heros) + 1
    ts = pickle.load(open('checkpoint/lr_prior.pos', "r"))
    t1 = np.array([]); t2 = np.array([])
    t1.resize(ts.shape[0]); t2.resize(ts.shape[0])

    allh1 = d_players.loc[d_players['match_id'] == match_id].loc[lambda r: r['player_slot'] < 5]
    allh2 = d_players.loc[d_players['match_id'] == match_id].loc[lambda r: r['player_slot'] >= 128]
    for items in allh1['hero_id'].iteritems():
        t1 += ts[:, items[1]]
    for items in allh2['hero_id'].iteritems():
        t2 += ts[:, items[1]]
    return t1, t2

def generate_match(matchid):
    global dataset
    d_match, d_time, d_fight, d_fightpls, d_players, d_heros = dataset
    # Sum up all heros' xp/gold/lh
    all_gold1 = d_time.loc[d_time['match_id'] == matchid].ix[:, range_count(2, 5, 3)].sum(axis = 1)
    all_lh1 = d_time.loc[d_time['match_id'] == matchid].ix[:, range_count(3, 5, 3)].sum(axis = 1)
    all_xp1 = d_time.loc[d_time['match_id'] == matchid].ix[:, range_count(4, 5, 3)].sum(axis = 1)
    all_gold2 = d_time.loc[d_time['match_id'] == matchid].ix[:, range_count(2+15, 5, 3)].sum(axis = 1)
    all_lh2 = d_time.loc[d_time['match_id'] == matchid].ix[:, range_count(3+15, 5, 3)].sum(axis = 1)
    all_xp2 = d_time.loc[d_time['match_id'] == matchid].ix[:, range_count(4+15, 5, 3)].sum(axis = 1)
    dies1, dies2 = generate_deaths(matchid)
    l = len(all_gold1)
    dies1.resize(l); dies2.resize(l)
    dies1 = pd.Series(dies1); dies2 = pd.Series(dies2)
    time_series = pd.Series(np.arange(l))
    prior_rate = test_match_prior(matchid).item()
    prior_series = pd.Series([prior_rate] * l)
    hts1, hts2 = generate_hero_ts(matchid)
    hts1.resize(l); hts2.resize(l)
    # print "SSSHAPE", hts1.shape, hts2.shape, l
    priorts_series = pd.Series(hts1) - pd.Series(hts2)

    delta_gold = all_gold1 - all_gold2
    delta_lh = all_lh1 - all_lh2
    delta_xp = all_xp1 - all_xp2
    delta_die = dies1 - dies2

    xs = pd.DataFrame()
    radiant_win = d_match[d_match['match_id'] == matchid].loc[matchid, 'radiant_win']
    (R, ) =  delta_gold.shape # whose type is Series
    SHAPE = delta_gold.shape
    assert delta_gold.shape == SHAPE
    assert delta_lh.shape == SHAPE
    assert delta_xp.shape == SHAPE
    assert delta_die.shape == SHAPE
    assert prior_series.shape == SHAPE
    assert priorts_series.shape == SHAPE
    assert R == l
    if R <= newl:
        return (None, None)

    CIN = 4
    # for x in [delta_gold, delta_lh, delta_xp, delta_die, prior_series, priorts_series]:
    for x in [delta_gold, delta_lh, delta_xp, delta_die]:
        x = x.reset_index(drop=True) # Remove original index
        # print "xsssss", x.iloc[0:newl].shape
        xs = xs.append(x.iloc[0:newl], ignore_index=True)
    xs = xs.T
    # print "xs.shape", xs.shape, len(xs)
    xs3d = xs.values.reshape((1, newl, CIN))

    w = tf.constant(1, shape=(W, CIN, DIM), dtype=tf.float64, name='w1')
    txs = tf.nn.conv1d(xs3d, w, STRIDE, 'VALID')
    nxs = sess.run(txs)
    tlen = nxs.shape[1]
    ys = pd.DataFrame(pd.Series([1.0 if radiant_win else 0.0]))
    print "shape", nxs.shape, nxs[0].shape
    finalxs = nxs[0].flatten()
    return pd.DataFrame(finalxs), ys

def make_dataset(S, T):
    Xs = pd.DataFrame()
    Ys = pd.DataFrame()
    for i in xrange(S, T):
        (xs, ys) = generate_match(i)
        if xs is None:
            # print "IGNORE MATCH", i
            pass
        else:
            print "MAKE MATCH", i
            Xs = Xs.append(xs.T, ignore_index = True) 
            Ys = Ys.append(ys.T, ignore_index = True) 
            print 'xs.shape, ys.shape, Xs.shape', xs.shape, ys.shape, Xs.shape

    # Normalize
    C = Xs.mean()
    Range = Xs.max() - Xs.min()
    Range = Range.mask(Range == 0, Xs.max())
    assert not np.any(Range.values == 0)
    Xs = (Xs - C) / Range
    print 'Xs.shape', Xs.shape
    print 'Ys.shape', Ys.shape
    return Xs, Ys, C, Range

class LR(torch.nn.Module):
    def __init__(self, in_dim, out):
        super(LR, self).__init__()
        self.linear = torch.nn.Linear(in_dim, out)
        self.sigmoid = torch.nn.Sigmoid()
 
    def forward(self, x):
        y_pred = self.sigmoid(self.linear(x))
        return y_pred

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_output, n_hidden = 10): 
        super(Net, self).__init__() 
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.out = torch.nn.Linear(n_hidden, n_output)
        # self.sigmoid = torch.nn.Sigmoid()
  
    def forward(self, x): 
        x = self.hidden(x)
        x = F.sigmoid(self.out(x))
        return x

def Train(MAXN, S = 0):
    Xs, Ys, C, Range = make_dataset(S, MAXN)

    loss_map = np.array([])
    # in_len = DIM * (newl - W + 1)
    in_len = Xs.shape[1]
    print "Xs", Xs.shape, "Ys", Ys.shape, 'in_len', in_len
    model = Net(in_len, 1)
    criterion = torch.nn.BCELoss(size_average=False)
    optimizer = torch.optim.Adam(model.parameters(), lr = STEP)

    x_data = torch.from_numpy(Xs.values).type('torch.FloatTensor')
    y_data = torch.from_numpy(Ys.values).type('torch.FloatTensor')

    for epoch in range(ITER):
        y_pred = model(x_data)
        loss = criterion(y_pred, y_data)
        # print "====>", epoch, loss.data
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_map = np.append(loss_map, loss.data)

    torch.save(model.state_dict(), 'checkpoint/p90_cnn_killpriorts.pkl')
    pickle.dump([C, Range], open('checkpoint/p90_cnn_killpriorts.norm', "w"))
    pickle.dump(loss_map, open('checkpoint/p90_cnn_killpriorts.loss', "w"))
    plt.plot(loss_map)
    plt.savefig("loss_p90_cnn_killpriorts.png")
    plt.close()

def test_match(matchid):
    # in_len = DIM * (newl - W + 1)
    [C, Range] = pickle.load(open('checkpoint/p90_cnn_killpriorts.norm', "r"))
    TXs, Tys = generate_match(matchid)
    if TXs is None:
        return None
    in_len = TXs.shape[0]
    print "len", in_len
    model = Net(in_len, 1)
    model.load_state_dict(torch.load('checkpoint/p90_cnn_killpriorts.pkl'))
    assert not np.any(Range.values == 0)
    # print 'TXsz222', TXs.shape, Tys.shape, C.shape, Range.shape
    TXs = TXs.T
    TXs = (TXs - C) / Range
    tx_data = torch.tensor(np.array(TXs)).type('torch.FloatTensor')
    ty_data = torch.tensor(np.array(Tys)).type('torch.FloatTensor')
    y_pred = model(tx_data)

    # print 'TXs', TXs.shape, Tys.shape, y_pred.shape
    yT = ty_data.detach().numpy()
    yO = y_pred.detach().numpy()
    yO = np.vectorize(lambda x: 1 if x > 0.5 else 0)(yO)
    yT = yT.astype(np.int64); yO = yO.astype(np.int64)
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
        # print "PREDICT", i
        l = test_match(i)
        if l is None:
            continue
        tot += 1
        print "l", l
        acc += l[0]
    percent = acc * 1.0 / tot
    pickle.dump(percent, open('checkpoint/p90_cnn_killpriorts.percent', 'w'))
    return percent

if __name__ == '__main__':
    # THRES = 300
    init_dataset()

    generate_match(0)

    TRAIN = 5000
    Train(TRAIN)
    print test(TRAIN, TRAIN+1000)
    # for i in xrange(1500, 1700):
    #     X,Y = generate_hero_ts(0)
    #     print X