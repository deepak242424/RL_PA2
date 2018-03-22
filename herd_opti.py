__author__ = 'deepak'

import numpy as np
from scipy.special import comb
import itertools

def get_string_combinations(val, range="01"):
    combs = ["".join(seq) for seq in itertools.product(range, repeat=val)]
    return combs

def transition(tup):
    yn, br, ol = tup
    bin_comb = get_string_combinations(sum(tup), "01")
    state_dict = {}
    for comb in bin_comb:
        yn_tr = comb[:yn].count('1')
        br_tr = comb[yn:yn+br].count('1')
        ol_tr = comb[yn+br:].count('1')
        tr_state = (yn-yn_tr, br-br_tr+yn_tr+ol_tr, ol-ol_tr+br_tr)
        prob_state = ((trans_pro[0][0])**(yn-yn_tr))*((trans_pro[0][1])**(yn_tr))*((trans_pro[1][1])**(br-br_tr))*((trans_pro[1][2])**(br_tr))*((trans_pro[2][2])**(ol-ol_tr))*((trans_pro[2][1])**(ol_tr))
        if state_dict.has_key(tr_state):
            state_dict[tr_state] += prob_state
        else:
            state_dict[tr_state] = prob_state
    return state_dict

def breerdable(tup):
    yn, br, ol = tup
    terniary_comb = get_string_combinations(tup[1], "012")
    state_dict = {}
    extra_prob = 0
    for comb in terniary_comb:
        zero_br = comb.count('0')
        one_br  = comb.count('1')
        two_br  = comb.count('2')
        tr_state = (yn+one_br+(2*two_br), br, ol)
        prob_state = (off_pro[1][0]**zero_br)*(off_pro[1][1]**one_br)*(off_pro[1][2]**two_br)
        if sum(tr_state)>12:
            extra_prob+=prob_state
        else:
            if state_dict.has_key(tr_state):
                state_dict[tr_state] += prob_state
            else:
                state_dict[tr_state] = prob_state
    maxValIdx = np.argmax([sum(tup) for tup in state_dict.keys()])
    state_dict[state_dict.keys()[maxValIdx]] += extra_prob
    #print extra_prob
    return state_dict

def get_reward(inState, action):
    cows_left = tuple(np.subtract(inState,action))
    util = cows_left[0]*utility[0] + cows_left[1]*utility[1] + cows_left[2]*utility[2]
    pay =  action[0]*payoff[0] + action[1]*payoff[1] + action[2]*payoff[2]
    reward = util+pay
    return reward

def calc_states(n, r):
    return comb(n, r).sum()

def convertBase10(tup):
    return tup[0]*13*13 + tup[1]*13 + tup[2]

def get_state_space():
    states_tup = []
    chk = 0
    for cnti in range(herd_size+1):
        for i in range(cnti+1):
            for j in range(cnti-i+1):
               states_tup.append((i,j,cnti-i-j))
               chk+=1
    return states_tup

def get_actions(tup):
    actions_tup = []
    chk = 0
    for i in range(tup[0]+1):
        for j in range(tup[1]+1):
            for k in range(tup[2]+1):
                actions_tup.append((i, j, k))
                chk += 1
    return actions_tup

def diff_tuple(tup1, tup2):
    return tuple(np.subtract(tup1, tup2))

def get_tr_br_prob_opti():
    chk=0
    stateSpace = get_state_space()
    dict_tr_br = {}
    dict_after = {}
    for present_state in stateSpace:
        dict_tr_br[present_state]={}
        actions = get_actions(present_state)
        next_states = [diff_tuple(present_state, action) for action in actions]
        for action_state in next_states:
            dict_tr_br[present_state][action_state] = {}
            if action_state in dict_after.keys():
                dict_tr_br[present_state][action_state] = dict_after[action_state].copy()
            else:
                trans_states = transition(action_state)
                for state in trans_states.keys():
                    breed_states = breerdable(state)
                    for br_state in breed_states.keys():
                        dict_tr_br[present_state][action_state][br_state] = breed_states[br_state]*trans_states[state]
                        chk += dict_tr_br[present_state][action_state][br_state]
                dict_after[action_state]=dict_tr_br[present_state][action_state].copy()
    return dict_tr_br

# Globals
herd_size = 12
trans_pro = np.asarray([[.9, .1, 0], [0, .75, .25], [0, .15, .85]])
off_pro = np.asarray([[1, 0, 0], [.05, .8, .15], [1, 0, 0]])
utility = np.asarray([.3, .4, .2])
payoff = np.asarray([2, 6, 4])
states = calc_states(np.asarray(range(13))+2, np.zeros(13)+2)
state_table = np.zeros((states, states))

#print get_state_space()
#print get_actions((2,3,4))

prob_tr_br = get_tr_br_prob_opti()
import cPickle
pickleFile = open('p_ss_a_opti2.save','wb')
cPickle.dump(prob_tr_br, pickleFile, cPickle.HIGHEST_PROTOCOL)
pickleFile.close()
