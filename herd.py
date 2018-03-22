__author__ = 'deepak'

import numpy as np
from scipy.special import comb
import itertools
import cPickle
import matplotlib.pyplot as plt

def plot_graph(x, y, label, axis, xlabel, ylabel):
    for val, lb in zip(y,label):
        plt.plot(x, val, label=lb)
    plt.axis(axis)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title('Value Function')
    plt.show()

def plot_scatter(x, y, label, axis, xlabel, ylabel, colors):
    for val, lb, color in zip(y,label, colors):
        plt.scatter(x, val, label=lb, c=color)
    plt.axis(axis)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc=2)
    plt.title('Policy for Different Gamma')
    plt.show()

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

def get_tr_br_prob():
    chk=0
    stateSpace = get_state_space()
    dict_tr_br = {}
    for present_state in stateSpace:
        actions = get_actions(present_state)
        next_states = [diff_tuple(present_state, action) for action in actions]
        for action_state in next_states:
            dict_tr_br[action_state] = {}
            trans_states = transition(action_state)
            for state in trans_states.keys():
                breed_states = breerdable(state)
                for br_state in breed_states.keys():
                    dict_tr_br[action_state][br_state] = breed_states[br_state]*trans_states[state]
                    chk += dict_tr_br[action_state][br_state]
    return dict_tr_br

def get_one_tr_br(tup):
    chk = 0
    present_state = tup
    dict_tr_br = {}
    actions = get_actions(present_state)
    next_states = [diff_tuple(present_state, action) for action in actions]
    for action_state in next_states:
        dict_tr_br[action_state] = {}
        trans_states = transition(action_state)
        for state in trans_states.keys():
            breed_states = breerdable(state)
            for br_state in breed_states.keys():
                dict_tr_br[action_state][br_state] = breed_states[br_state]*trans_states[state]
                chk += dict_tr_br[action_state][br_state]
    return dict_tr_br

def valueIteratoin(theta = .01, sweeps=1000):
    delta=10
    ValuesT = np.zeros(states)
    swpCnt = 0
    while(delta>theta):
        if swpCnt<sweeps:
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
            print 'delta = ',delta
            swpCnt+=1
        else:
            break
    return ValuesT

def policyEvaluation(theta, Values, pai, gamma=.9):
    delta = 10
    ValuesT = Values.copy()
    while(delta>theta):
        delta = 0
        ValuesTm1 = ValuesT.copy()
        for state in state_probs.keys():
            v = ValuesTm1[state2ID[state]]
            action = pai[state2ID[state]]
            afterState = diff_tuple(state,action)
            reward = get_reward(state, action)
            val = 0
            for terminalState in state_probs[state][afterState].keys():
                val += (state_probs[state][afterState][terminalState])*(reward+gamma*ValuesTm1[state2ID[terminalState]])
            ValuesT[state2ID[state]] = val
            delta = max(delta, abs(v-ValuesT[state2ID[state]]))
    return ValuesT

def policyImprovement(ValuesT, pai, gamma=.9):
    POLICY_STABLE = True
    ValuesTm1 = ValuesT.copy()
    paiDash = [(0,0,0)]*states
    for state in state_probs.keys():
        action = pai[state2ID[state]]
        v = ValuesTm1[state2ID[state]]
        max_action=action
        max_val = 0
        for afterState in state_probs[state].keys():
            reward = get_reward(state, diff_tuple(state,afterState))
            val = 0
            for terminalState in state_probs[state][afterState].keys():
                val += (state_probs[state][afterState][terminalState])*(reward+gamma*ValuesTm1[state2ID[terminalState]])
            if val>max_val:
                max_action=diff_tuple(state, afterState)
                max_val = val
        paiDash[state2ID[state]] = max_action
        if max_action != action:
            POLICY_STABLE = False
    return POLICY_STABLE, paiDash

def policyIteration(sweeps=1000, gamma=.9):
    ValuesT = np.zeros(states)
    pai = [(0, 0, 0)]*states
    FLAG = False
    swpCnt = 0
    while FLAG==False:
        if swpCnt<sweeps:
            ValuesT = policyEvaluation(.01, ValuesT, pai, gamma)
            FLAG, pai = policyImprovement(ValuesT,pai, gamma)
            swpCnt+=1
        else:
            break
    return ValuesT, pai

# Globals
herd_size = 12
trans_pro = np.asarray([[.9, .1, 0], [0, .75, .25], [0, .15, .85]])
off_pro = np.asarray([[1, 0, 0], [.05, .8, .15], [1, 0, 0]])
utility = np.asarray([.3, .4, .2])
payoff = np.asarray([2, 6, 4])
states = calc_states(np.asarray(range(13))+2, np.zeros(13)+2)
state_table = np.zeros((states, states))
gamma = 0.9
all_statesTup = get_state_space()
state2ID = dict((key, val) for (key,val) in zip(all_statesTup,range(int(states))))
ID2state = dict((val, key) for (key,val) in zip(all_statesTup,range(int(states))))
base13Sorted = sorted(all_statesTup)

pickleFile = open('p_ss_a.save','rb')
state_probs = cPickle.load(pickleFile)
pickleFile.close()

# valueFn1 = valueIteratoin(sweeps=1)
# valueFn2 = valueIteratoin(sweeps=10)
# valueFn3 = valueIteratoin()
# plotVy1 = [valueFn1[state2ID[base13Sorted[idx]]] for idx in range(int(states))]
# plotVy2 = [valueFn2[state2ID[base13Sorted[idx]]] for idx in range(int(states))]
# plotVy3 = [valueFn3[state2ID[base13Sorted[idx]]] for idx in range(int(states))]
# plot_graph(range(int(states)), [plotVy1, plotVy2, plotVy3], ['Sweep=1', 'Sweep=10', 'Optimal'], [0, 500, 0, 150], 'States', 'Value')

# valueFn1 = policyIteration(gamma=.9)
# valueFn2 = policyIteration(gamma=.5)
# valueFn3 = policyIteration(gamma=.3)
# #valueFn4 = policyIteration()
# plotVy1 = [valueFn1[state2ID[base13Sorted[idx]]] for idx in range(int(states))]
# plotVy2 = [valueFn2[state2ID[base13Sorted[idx]]] for idx in range(int(states))]
# plotVy3 = [valueFn3[state2ID[base13Sorted[idx]]] for idx in range(int(states))]
# #plotVy4 = [valueFn4[state2ID[base13Sorted[idx]]] for idx in range(int(states))]
# #plot_graph(range(int(states)), [plotVy1, plotVy2, plotVy3, plotVy4], ['Sweep=1', 'Sweep=2', 'Sweep=3', 'Optimal'], [0, 500, 0, 150], 'States', 'Value')
# plot_graph(range(int(states)), [plotVy1, plotVy2, plotVy3], ['Gamma=.9', 'Gamma=.5', 'Gamma=.3'], [0, 500, 0, 150], 'States', 'Value')

# valueFn1 = policyIteration(sweeps=2)
# plotVy1 = [valueFn1[state2ID[base13Sorted[idx]]] for idx in range(int(states))]
# plot_graph(range(int(states)), [plotVy1], ['Sweeps=2'], [0, 500, 0, 150], 'States', 'Value')

#Q3.b
# testStates=[(4, 7, 1), (1, 3, 6), (9, 2, 1)]
# for state in testStates:
#     valueFn1,pai1 = policyIteration(gamma=.9)
#     valueFn2,pai2 = policyIteration(gamma=.5)
#     valueFn3,pai3 = policyIteration(gamma=.3)
#     print pai1[state2ID[state]]
#     print pai3[state2ID[state]]
#     print pai3[state2ID[state]]

# valueFn1,pai1 = policyIteration(gamma=.9)
# valueFn2,pai2 = policyIteration(gamma=.5)
# valueFn3,pai3 = policyIteration(gamma=.3)
#
# plotVy1 = [base13Sorted.index(pai1[state2ID[base13Sorted[idx]]]) for idx in range(int(states))]
# plotVy2 = [base13Sorted.index(pai2[state2ID[base13Sorted[idx]]]) for idx in range(int(states))]
# plotVy3 = [base13Sorted.index(pai3[state2ID[base13Sorted[idx]]]) for idx in range(int(states))]
#
# plot_scatter(range(int(states)), [plotVy1, plotVy2, plotVy3], ['Gamma=.9', 'Gamma=.5', 'Gamma=.3'], [0, 500, 0, 500], 'States', 'Action', ['r','b','g'])

#print valueIteratoin()
#value, policy = policyIteration(gamma=.9)

# pf = open('Policy.save', 'wb')
# cPickle.dump(policy, pf, cPickle.HIGHEST_PROTOCOL)
# pf.close()

pf = open('Policy.save', 'rb')
print cPickle.load(pf)