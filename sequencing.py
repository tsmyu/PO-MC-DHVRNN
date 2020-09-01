# sequencing.py
import glob, os, sys, math, warnings, copy, time
import numpy as np
import pandas as pd
from scipy import signal
# Keisuke Fujii, 2020
# modifying the code https://github.com/samshipengs/Coordinated-Multi-Agent-Imitation-Learning

# ===============================================================================
# subsample_sequence ============================================================
# ===============================================================================
def subsample_sequence(events, subsample_factor, random_sample=False):
    if subsample_factor == 0 or round(subsample_factor*10)==10:
        return events
    
    def subsample_sequence_(moments, subsample_factor, random_sample=False):#random_state=42):
        ''' 
            moments: a list of moment 
            subsample_factor: number of folds less than orginal
            random_sample: if true then sample a random one from the window of subsample_factor size
        '''
        seqs = np.copy(moments)
        moments_len = seqs.shape[0]
        if subsample_factor > 0:
            n_intervals = moments_len//subsample_factor # number of subsampling intervals
        else: 

            n_intervals = int(moments_len//-subsample_factor)

        left = moments_len % subsample_factor # reminder

        if random_sample:
            if left != 0:
                rs = [np.random.randint(0, subsample_factor) for _ in range(n_intervals)] + [np.random.randint(0, left)]
            else:
                rs = [np.random.randint(0, subsample_factor) for _ in range(n_intervals)]
            interval_ind = range(0, moments_len, subsample_factor)
            # the final random index relative to the input
            rs_ind = np.array([rs[i] + interval_ind[i] for i in range(len(rs))])
            return seqs[rs_ind, :]
        else:
            if round(subsample_factor*10) == round(subsample_factor)*10: # int
                s_ind = np.arange(0, moments_len, subsample_factor)
                return seqs[s_ind, :]
            else:
                # only when 10 Hz undersampling in NBA (25 Hz)
                if round(subsample_factor*10) == 25:
                    up = 2
                    down = 5
                seqs2 = signal.resample_poly(seqs, up, down, axis=0, padtype='line')
                seqs2 = seqs2[1:-1]

                return seqs2
                          
    return [subsample_sequence_(ms, subsample_factor) for ms in events]



def get_sequences(single_game, policy, sequence_length, overlap, n_pl, k_nearest, n_feat, velocity = 0, in_sma=False):
    ''' create events where each event is a list of sequences from
        single_game with required sequence_legnth and overlap

        single_game: A list of events
        sequence_length: the desired length of each event (a sequence of moments)
        overlap: how much overlap wanted for the sequence generation
 
    '''

    X_all = []
    Y_all = []   
    
    ''' 
    # original---(velocity)
    # basketball:
       0-254: static_feature (positions and angles)
           0-19: positions(xy: DF->OF)
           20-22: ball xyz
           23-25: quarter,time_left,shot clock 
     　    26-35: relations between all players and ball, 36-45:cos(th), 46-55:sin(th), 56-65:theta
     　    66-105: relations between all players and goal (the same above)
           106-505: relations between all players  (the same above)
       506-528: dyanmics_feature (23 velocities)  
     　529-578: one-hot_feature（25 team one-hot but actually 30 teams, DF->OF)

     # soccer:
       0-254: static_feature (positions and angles)
           0-43: positions(xy: DF->OF, each goalkeeper is the last)
           44-45: ball xy
     　    46-133: relations between all players and ball, 22*(dist,cos(th),sin(th), theta)
     　    134-221: relations between all players and goal (the same above)
           222-2157: relations between all players (the same above) 22*22*4
       2158-2203: dyanmics_feature (46 velocities)  
 
    # transform into: 
        In Le's code, for all players,
        0-2+pl*npl: distance, cos, sin with the defender (if oneself, zeros)
        3-7+pl*npl: position and velocity of the player oneself
        8-10+pl*npl: distance, cos, sin with the goal
        9-12+pl*npl: distance, cos, sin with the ball
        0-(k-1)+13*npl*2+pl*k: k nearest players
        + ball position (+ team one-hot)
        total: 13*(22+3)+2 = 327 (soccer) or 13*(10+3)+3+50 = 222 (NBA)

    # original---(acceleration)
    # basketball:
       0-254: static_feature (positions and angles)
           0-19: positions(xy: DF->OF)
           20-22: ball xyz
           23-25: quarter,time_left,shot clock 
     　    26-35: relations between all players and ball, 36-45:cos(th), 46-55:sin(th), 56-65:theta
     　    66-105: relations between all players and goal (the same above)
           106-505: relations between all players  (the same above)
       506-528: dyanmics_feature (23 velocities)  
       529-551: dyanmics_feature (23 acceleration)  
     　552-601: one-hot_feature（25 team one-hot but actually 30 teams, DF->OF)

     # soccer:
       0-254: static_feature (positions and angles)
           0-43: positions(xy: DF->OF, each goalkeeper is the last)
           44-45: ball xy
     　    46-133: relations between all players and ball, 22*(dist,cos(th),sin(th), theta)
     　    134-221: relations between all players and goal (the same above)
           222-2157: relations between all players (the same above) 22*22*4
       2158-2203: dyanmics_feature (46 velocities)  
       2204-2249: dyanmics_feature (46 acceleartion) 
 
    # transform into: 
        In Le's code, for all players,
        0-2+pl*npl: distance, cos, sin with the defender (if oneself, zeros)
        3-7+pl*npl: position and velocity of the player oneself
        8-10+pl*npl: distance, cos, sin with the goal
        9-12+pl*npl: distance, cos, sin with the ball
        0-(k-1)+13*npl*2+pl*k: k nearest players
        + ball pos/vel (+ team one-hot)
        total: 15*22 + 4 = 334 (soccer) or 15*10 + 4 = 154 (NBA)

    ''' 
    npl = n_pl*2
    index0 = np.array(range(single_game[0].shape[1])).astype(int) # length of features

    for p in policy:
        X = []
        Y = []
        # create index
        index = [] 
        if n_pl == 5:
            for pl in range(npl):
                if not in_sma:
                    index = np.append(index,index0[106+pl+p*npl*4]) # distance between players 0
                    index = np.append(index,index0[116+pl+p*npl*4]) # cos 1
                    index = np.append(index,index0[126+pl+p*npl*4]) # sin 2
                index = np.append(index,index0[pl*2:pl*2+2]) # positions 3-4
                if velocity >= 0:
                    index = np.append(index,index0[506+pl*2:506+pl*2+2]) # velocities 5-6
                if velocity == 2:
                    index = np.append(index,index0[529+pl*2:529+pl*2+2]) # acceleration 
                if not in_sma:    
                    index = np.append(index,index0[66+pl:95+pl:10]) # relation with the goal 7-9 (th is not used)
                    index = np.append(index,index0[26+pl:55+pl:10]) # relation with the ball 10-12
            # k nearest players  
            if k_nearest > 0 and k_nearest < 10: # players regardless of attackers and defenders
                index = np.append(index,np.zeros(n_feat*k_nearest)) # temporary
            
            index = np.append(index,index0[20:22]) # ball positions (excluding 3d)
            if velocity >= 0:
                index = np.append(index,index0[526:528])  # ball velocity (excluding 3d)
            #if velocity == 2:
            #    index = np.append(index,index0[549:551])
            # index = np.append(index,index0[529:579]) # team one-hot
        elif n_pl == 11:
            for pl in range(npl):
                if not in_sma:
                    index = np.append(index,index0[222+pl+p*npl*4]) # distance between players 0
                    index = np.append(index,index0[244+pl+p*npl*4]) # cos 1
                    index = np.append(index,index0[266+pl+p*npl*4]) # sin 2
                index = np.append(index,index0[pl*2:pl*2+2]) # positions 3-4
                if velocity >= 0:
                    index = np.append(index,index0[2158+pl*2:2158+pl*2+2]) # velocities 5-6
                if velocity == 2:
                    index = np.append(index,index0[2204+pl*2:2204+pl*2+2]) # velocities 5-6
                if not in_sma:
                    index = np.append(index,index0[134+pl:134+npl*3+pl-1:npl]) # relation with the goal 7-9 (th is not used)
                    index = np.append(index,index0[46+pl:46+npl*3+pl-1:npl]) # relation with the ball 10-12
            # k nearest players    
            if k_nearest > 0 and k_nearest < 10: # players regardless of attackers and defenders
                index = np.append(index,np.zeros(n_feat*k_nearest)) # temporary
            
            index = np.append(index,index0[44:46]) # ball positions
            if velocity >= 0:
                index = np.append(index,index0[2202:2204]) # ball velocity
            #if velocity == 2:
            #    index = np.append(index,index0[2248:2250])

        index = index.astype(int)
        #index = np.array([p*2,p*2+1, \
        #    25+p,35+p,45+p,55+p,65+p,75+p,85+p,95+p,\
        #    p*2+105,p*2+106])
        for i in single_game:
            i_len = len(i)
            i2 = np.array(i) # copy
            sequence0 = np.zeros((i_len,index.shape[0]))
            
            for t in range(i_len):
                # nearest players
                if k_nearest > 0 and k_nearest < 10: # players regardless of attackers and defenders
                    dist = i[t][index[0:npl*n_feat:n_feat]] # index of distances
                    ind_nearest = dist.argsort()[0:(k_nearest+1)] 
                    ind_nearest = ind_nearest[np.nonzero(ind_nearest)][:k_nearest] # eliminate zero and duplication
                    for k in range(k_nearest):
                        index[n_feat*npl+k*n_feat:n_feat*npl+(k+1)*n_feat] = index[ind_nearest[k]*n_feat:ind_nearest[k]*n_feat+n_feat]
                sequence0[t,:] = i2[t,index].T
            
            # create sequences
            if i_len >= sequence_length:
                sequences0 = [sequence0[-sequence_length:,:] if j + sequence_length > i_len-1 else sequence0[j:j+sequence_length,:] \
                    for j in range(0, i_len-overlap, sequence_length-overlap)] # for the states
                #sequences = [np.array(i[-sequence_length:]) if j + sequence_length > i_len-1 else np.array(i[j:j+sequence_length]) \
                #    for j in range(0, i_len-overlap, sequence_length-overlap)] # for the actions     

                state = [np.roll(kk, -1, axis=0)[:-1, :] for kk in sequences0] # state: drop the last row as the rolled-back is not real
                
                if velocity == 2:
                    action = [np.roll(kk[:, p*n_feat+3:p*n_feat+9], -1, axis=0)[:-1, :] for kk in sequences0] 
                    # action2 = [np.roll(kk[:, p*2:p*2+2], -1, axis=0)[:-1, :] for kk in sequences] 
                elif velocity == 1:
                    action = [np.roll(kk[:, p*n_feat+3:p*n_feat+7], -1, axis=0)[:-1, :] for kk in sequences0] 
                elif velocity:
                    action = [np.roll(kk[:, [p*n_feat+5,p*n_feat+6,p*n_feat+3,p*n_feat+4]], -1, axis=0)[:-1, :] for kk in sequences0] 
                else: # position only
                    action = [np.roll(kk[:, p*n_feat+3:p*n_feat+5], -1, axis=0)[:-1, :] for kk in sequences0] 
                    # action = [np.roll(kk[:, p*2:p*2+2], -1, axis=0)[:-1, :] for kk in sequences] # action    
                # sequences = [l[:-1, :] for l in sequences] # since target has dropped one then sequence also drop one
                X += state  
                Y += action  
        X_all.append(X) 
        Y_all.append(Y) 
    return X_all, Y_all

