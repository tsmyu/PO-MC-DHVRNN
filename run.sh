#!/bin/bash
# a simple demostration code with small samples
python -u main.py --data nba --n_GorS 1 --n_roles 5 --batchsize 32 --n_epoch 3 -ev_th 200 --model MACRO_VRNN --attention 3 --body --jrk 0.2 --lam_acc 0.2 --pretrain 50 

# attention = -1: w/o embedding 
# attention =  3: individual binary observation

# actual training and test code
# for training, remove --TEST
# 1. Sanity check & 2.RNN-gauss 
# python -u main.py --data nba --n_GorS 100 --n_roles 5 --batchsize 256 --n_epoch 10 -ev_th 200 --model RNN_GAUSS --attention -1 --Sanity --TEST
# 3. VRNN
# python -u main.py --data nba --n_GorS 100 --n_roles 5 --batchsize 256 --n_epoch 8 -ev_th 200 --model MACRO_VRNN --attention -1 --wo_macro --acc 2 --TEST 
# 4. VRNN-macro
# python -u main.py --data nba --n_GorS 100 --n_roles 5 --batchsize 256 --n_epoch 10 -ev_th 200 --model MACRO_VRNN --attention -1 --pretrain 50 --TEST
# 5. VRNN-Mech 
# python -u main.py --data nba --n_GorS 100 --n_roles 5 --batchsize 256 --n_epoch 10 -ev_th 200 --model MACRO_VRNN --attention -1 --wo_macro --body --jrk 0.1 --lam_acc 0.2 --TEST 
# 6. VRNN-bi 
# python -u main.py --data nba --n_GorS 100 --n_roles 5 --batchsize 256 --n_epoch 10 -ev_th 200 --model MACRO_VRNN --attention 3 --wo_macro --TEST
# 7. nba VRNN-macro-bi-Mech
# python -u main.py --data nba --n_GorS 100 --n_roles 5 --batchsize 256 --n_epoch 15 -ev_th 200 --model MACRO_VRNN --attention 3 --body --jrk 0.1 --lam_acc 0.2 --pretrain 50 --TEST