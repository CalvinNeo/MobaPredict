import numpy as np
import pandas as pd
import itertools
import torch
from torchvision import transforms as T
import matplotlib.pyplot as plt
import pickle

THRES = 45000
ITER = 400
STEP = 0.00005

dataset = ()

def init_dataset():
    global dataset
    chunks = pd.read_csv('dota-2-matches/players.csv', sep=',', skiprows=0, chunksize = 400)
    chunks = itertools.takewhile(lambda chunk: int(chunk['match_id'].iloc[-1]) < THRES, chunks)
    d_players = pd.concat(chunks)

    chunks = pd.read_csv('dota-2-matches/match.csv', sep=',', skiprows=0, chunksize = 50)
    chunks = itertools.takewhile(lambda chunk: int(chunk['match_id'].iloc[-1]) < THRES, chunks)
    d_match = pd.concat(chunks)

    d_heros = pd.read_csv('dota-2-matches/hero_names.csv', sep=',')

    dataset = (d_players, d_match, d_heros)
    return dataset


def generate_hero(match_id):
    d_players, d_match, d_heros = dataset
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

def make_dataset(S, T):
    Xs = pd.DataFrame()
    Ys = pd.Series()
    for i in xrange(S, T):
        print "MAKE MATCH", i
        (xs, y) = generate_hero(i)
        Xs = Xs.append(xs, ignore_index=True)
        y = 1 if y.bool() else 0
        y = float(y)
        Ys = Ys.append(pd.Series([y]), ignore_index=True)

    # Normalize
    Min = 0
    Range = 1
    return Xs, Ys, Min, Range

def generate_hero_time(S, T):
    d_players, d_match, d_heros = dataset
    len_heros = len(d_heros) + 1
    ts = np.array([])
    totts = np.array([])
    initial_duration = 100
    ts.resize((initial_duration, len_heros))
    totts.resize((initial_duration, len_heros))
    for match_id in xrange(S, T):
        radiant_win = d_match.loc[d_match['match_id'] == match_id]['radiant_win'].bool()
        duration = int(d_match.loc[d_match['match_id'] == match_id]['duration'].values[0].item() / 60)
        if duration >= ts.shape[0]:
            ts.resize((duration + 1, len_heros))
            totts.resize((duration + 1, len_heros))
        for items in  d_players.loc[d_players['match_id'] == match_id]['hero_id'].iteritems():
            totts[duration, items[1]] += 1
        if radiant_win:
            allh1 = d_players.loc[d_players['match_id'] == match_id].loc[lambda r: r['player_slot'] < 5]
            for items in allh1['hero_id'].iteritems():
                ts[duration, items[1]] += 1
        else:
            allh2 = d_players.loc[d_players['match_id'] == match_id].loc[lambda r: r['player_slot'] >= 128]
            for items in allh2['hero_id'].iteritems():
                ts[duration, items[1]] += 1
    #  sum of win heros at each time point
    print np.sum(ts, axis = 1)
    print np.sum(totts, axis = 1)
    ts = ts * 1.0 / totts
    ts[np.isnan(ts)] = 0.5
    pickle.dump(ts * 1.0, open('checkpoint/lr_prior.pos', "w"))
    return ts

class LR(torch.nn.Module):
    def __init__(self, in_dim, out):
        super(LR, self).__init__()
        self.linear = torch.nn.Linear(in_dim, out)
        self.sigmoid = torch.nn.Sigmoid()
 
    def forward(self, x):
        y_pred = self.sigmoid(self.linear(x))
        return y_pred

def Train(START, END):
    Xs, Ys, Min, Range = make_dataset(START, END)

    d_players, d_match, d_heros = dataset
    len_heros = len(d_heros) + 1
    model = LR(len_heros * 2, 1)

    criterion = torch.nn.BCELoss(size_average=False)
    optimizer = torch.optim.SGD(model.parameters(), lr = STEP)

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

    torch.save(model.state_dict(), 'checkpoint/lr_prior.pkl')
    pickle.dump([Min, Range], open('checkpoint/lr_prior.norm', "w"))

def test_match_prior(matchid):
    d_players, d_match, d_heros = dataset
    len_heros = len(d_heros) + 1
    model = LR(len_heros * 2, 1)
    model.load_state_dict(torch.load('checkpoint/lr_prior.pkl'))
    [Min, Range] = pickle.load(open('checkpoint/lr_prior.norm', "r"))
    TXs, Tys = generate_hero(matchid)
    TXs = (TXs - Min) / Range
    tx_data = torch.tensor(np.array(TXs)).type('torch.FloatTensor')
    ty_data = torch.tensor(np.array(Tys)).type('torch.FloatTensor')
    y_pred = model(tx_data)

    yT = ty_data.detach().numpy()
    yO = y_pred.detach().numpy()
    yO = np.vectorize(lambda x: 1 if x > 0.5 else 0)(yO)
    yT = yT.astype(np.int64)
    yO = yO.astype(np.int64)
    loss = (yT == yO).astype(np.int64)
    return yT, yO, loss


def test(S, E):
    toth = np.array([0]).reshape((-1, ))
    totl = np.array([0]).reshape((-1, ))
    totl = 0
    for i in xrange(S, E):
        print "PREDICT", i
        _, _, l = test_match_prior(i)
        totl += l

    percent = totl * 1.0 / (E - S)
    return percent

if __name__ == '__main__':
    init_dataset()
    Train(35000, 45000)
    # print test(18001, 19990)
    # print generate_hero_time(0, 100)
    print generate_hero_time(35000, 45000)