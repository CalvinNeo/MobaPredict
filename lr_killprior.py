import numpy as np
import pandas as pd
import itertools
import torch
from torchvision import transforms as T
import matplotlib.pyplot as plt
import pickle

THRES = 2000
DIM = 6
ITER = 400
W = 3

dataset = ()

def init_dataset():
    global dataset
    chunks = pd.read_csv('dota-2-matches/match.csv', sep=',', skiprows=0, chunksize = 50)
    chunks = itertools.takewhile(lambda chunk: int(chunk['match_id'].iloc[-1]) < THRES, chunks)
    d_match = pd.concat(chunks)

    chunks = pd.read_csv('dota-2-matches/player_time.csv', sep=',', skiprows=0, chunksize = 50)
    chunks = itertools.takewhile(lambda chunk: int(chunk['match_id'].iloc[-1]) < THRES, chunks)
    d_time = pd.concat(chunks)

    chunks = pd.read_csv('dota-2-matches/teamfights.csv', sep=',', skiprows=0, chunksize = 50)
    chunks = itertools.takewhile(lambda chunk: int(chunk['match_id'].iloc[-1]) < THRES, chunks)
    d_fight = pd.concat(chunks)

    chunks = pd.read_csv('dota-2-matches/teamfights_players.csv', sep=',', skiprows=0, chunksize = 50)
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
    model.load_state_dict(torch.load('checkpoint/params_lrprior.pkl'))
    [Mean, Range] = pickle.load(open('checkpoint/params_lrprior.norm', "r"))
    TXs, Tys = generate_hero(matchid)
    TXs = (TXs - Mean) / Range
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

def generate_match(matchid):
    global dataset
    d_match, d_time, d_fight, d_fightpls, d_players, d_heros = dataset
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

    delta_gold = all_gold1 - all_gold2
    delta_lh = all_lh1 - all_lh2
    delta_xp = all_xp1 - all_xp2
    delta_die = dies1 - dies2
    # print (delta_die == 0).all()

    radiant_win = d_match[d_match['match_id'] == matchid].loc[matchid, 'radiant_win']
    (R, ) =  delta_gold.shape # whose type is Series
    # xs = pd.DataFrame(columns = map(lambda i: "In" + str(i), range(3 * W)))
    # ys = pd.DataFrame(columns = ["T"])
    xs = pd.DataFrame()
    ys = pd.DataFrame()
    for i in xrange(0, R - W):
        x = pd.concat([delta_gold[i:i+W], delta_lh[i:i+W], delta_xp[i:i+W], 
            delta_die[i:i+W], time_series[i:i+W], prior_series[i:i+W]])
        x = x.reset_index(drop=True) # Remove original index
        y = 1 if radiant_win else 0
        y = float(y)
        xs = xs.append(x, ignore_index=True)
        ys = ys.append(pd.Series(y), ignore_index=True)

    return xs, ys

def make_dataset(S, T):
    Xs = pd.DataFrame()
    Ys = pd.DataFrame()
    for i in xrange(S, T):
        print "MAKE MATCH", i
        (xs, ys) = generate_match(i)
        Xs = pd.concat([Xs, xs])
        Ys = pd.concat([Ys, ys])

    # Normalize
    Mean = Xs.mean()
    Range = Xs.max() - Xs.min()
    if Range.values[0].item() == 0:
        Xs = Xs - Mean
    else:
        Xs = (Xs - Mean) / Range
    return Xs, Ys, Mean, Range

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
    Xs, Ys, Mean, Range = make_dataset(0, MAXN)

    model = LR(DIM * W, 1)
    criterion = torch.nn.BCELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.0001)

    print 'Xs'
    print Xs
    print 'Ys'
    print Ys
    x_data = torch.from_numpy(Xs.values).type('torch.FloatTensor')
    y_data = torch.from_numpy(Ys.values).type('torch.FloatTensor')
    # x_data = torch.tensor(Xs.detach().numpy()).type('torch.FloatTensor')
    # y_data = torch.tensor(Ys.detach().numpy()).type('torch.FloatTensor')
    print x_data.size()
    print y_data.size()

    for epoch in range(ITER):
        # Forward pass
        y_pred = model(x_data)

        # Compute loss
        loss = criterion(y_pred, y_data)
        # print "====>", epoch, loss.data

        # Zero gradients
        optimizer.zero_grad()
        # perform backward pass
        loss.backward()
        # update weights
        optimizer.step()

    torch.save(model.state_dict(), 'checkpoint/params_lrkillprior.pkl')
    pickle.dump([Mean, Range], open('checkpoint/params_lrkillprior.norm', "w"))

def test_match(matchid):
    model = LR(DIM * W, 1)
    model.load_state_dict(torch.load('checkpoint/params_lrkillprior.pkl'))
    [Mean, Range] = pickle.load(open('checkpoint/params_lrkillprior.norm', "r"))
    TXs, Tys = generate_match(matchid)
    if Range.values[0].item() == 0:
        TXs = TXs - Mean
    else:
        TXs = (TXs - Mean) / Range
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
    np.savetxt(open("test_match/{}.txt".format(matchid), "w"), res, fmt = "%d")
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
    plt.savefig("reskillprior.png")
    return percent

if __name__ == '__main__':
    init_dataset()
    Train(100)
    test(110, 115)