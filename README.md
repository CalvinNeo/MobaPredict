# MobaPredict
Predict result of moba games. Currently Dota2.

## Dataset
https://www.kaggle.com/devinanzelmo/dota-2-matches
Download and extract as a `dota-2-matches` folder.

## Models
Trained models are stored in folder `checkpoint`.

- lr(xp,gold,lh).py
	Time series LR.
- lr(kill).py
	lr(xp,gold,lh) + kill/death data of teamfights.
