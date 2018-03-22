__author__ = 'deepak'

import numpy as np
import cPickle
from scipy.special import comb

def calc_states(n, r):
    return comb(n, r).sum()
def get_state_space():
    states_tup = []
    chk = 0
    for cnti in range(herd_size+1):
        for i in range(cnti+1):
            for j in range(cnti-i+1):
               states_tup.append((i,j,cnti-i-j))
               chk+=1
    return states_tup
def get_reward(inState, action):
    cows_left = tuple(np.subtract(inState,action))
    util = cows_left[0]*utility[0] + cows_left[1]*utility[1] + cows_left[2]*utility[2]
    pay =  action[0]*payoff[0] + action[1]*payoff[1] + action[2]*payoff[2]
    reward = util+pay
    return reward
def diff_tuple(tup1, tup2):
    return tuple(np.subtract(tup1, tup2))

# Globals
herd_size = 12
states = calc_states(np.asarray(range(13))+2, np.zeros(13)+2)
gamma = 0.9
all_statesTup = get_state_space()
state2ID = dict((key, val) for (key,val) in zip(all_statesTup,range(int(states))))
ID2state = dict((val, key) for (key,val) in zip(all_statesTup,range(int(states))))
pickleFile = open("p_ss_a.save", "rb")
state_probs = cPickle.load(pickleFile)
utility = np.asarray([.3, .4, .2])
payoff = np.asarray([2, 6, 4])


ValuesT = np.zeros(states)
delta=10
theta = .01

while(delta>theta):
    delta = 0
    ValuesTm1 = ValuesT.copy()
    for state in state_probs.keys():
        v = ValuesTm1[state2ID[state]]
        max_val=0
        for afterState in state_probs[state].keys():
            reward = get_reward(state, diff_tuple(state,afterState))
            val = 0
            for terminalState in state_probs[state][afterState].keys():
                val += (state_probs[state][afterState][terminalState])*(reward+gamma*ValuesTm1[state2ID[terminalState]])
            if val>max_val:
                max_val=val
        ValuesT[state2ID[state]] = max_val
        delta = max(delta, abs(v-ValuesT[state2ID[state]]))
    print ValuesT
