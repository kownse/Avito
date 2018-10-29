# [Avito Demand Prediction Challenge](https://www.kaggle.com/c/avito-demand-prediction)
top 5% solution for Avito Demand Prediction Challenge

The task of this competition is to predict the deal probability of each Ads in Ruassian online 
shop Avito. The organizer provided multiple inputs including:
1. structure data 
2. text for item infomation
3. images to show the item
The total size of training data is 123 GB in which the most part are images.

## Learn to ensemble
I learnt from the failure of last competition in 
[TalkingData AdTracking Fraud Detection Challenge](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection) in which
I didn't use any ensemble at all and dropped from 7% to 16% in the last hour when somebody published a high score solution
which ensembled all the public kernel.
It was a pain, and I learnt from that pain.
So, from the begining of this competition, I kept that in mind to train as many kinds of models as posssible 
and then ensemble ithem.

### Capsule Network on images
It was [Geoffrey Hinton](https://en.wikipedia.org/wiki/Geoffrey_Hinton) who invented the idea of deeplearning 
published the [Capsule Network](https://www.youtube.com/watch?v=6S1_WqE55UQ) to fix the lack of the relative 
position information in traditional CNN.
I've been always curious to try capsule network out and this is the time.
But it seems not working well in regression problems for capsule network with low accuracy.

### Gradient Boost Trees
Then I jump into the GBTs with structure data, TF-IDF from texts and statistic features
calculated from numerical and catergorical data.
I tried lightgbm and catboost in python, and xgboost in R which deliver better results
than python.

### Add more features into GBTs
Then I add following features into the gradient boost model:
1. calculated statistical features from images.
2. Using kmeans to calculate regional statistic features by join the lat and lon from the 
data with actually city coordinates.
3. Aggreated feature for each day.
4. City population.

After add these information, I got top 7% in the leaderboard.
And continue training with kfolds and ensemble all the results, I got top 5%.

## What I missed
The first mistake is that I failed to build a end-to-end neual network which was done by
the winner of this competition.
![winner solution](https://pbs.twimg.com/media/DgvX3pWUYAAlCAK.jpg:large)
The second mistatke is that I didn't do stacking because I don't know how to stack 
at that moment. I learnt to stack one more week after this competition.
