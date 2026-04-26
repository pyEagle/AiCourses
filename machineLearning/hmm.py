# -*- coding:utf-8 -*-

import random
import numpy as np

states = {0, 1, 2}
pi = np.array([0.25, 0.5, 0.25])
A = np.array([[0,   1, 0],
              [0.5, 0, 0.5],
              [0,   1, 0]])

def simulate_states1(sumofstates):
    start_state = random.randint(0,1) + random.randint(0,1)
    state_list = [start_state]
    pre_state = start_state
    for i in range(sumofstates-1):
        if pre_state==0: 
            new_state = 1
        elif pre_state==2:
            new_state = 1
        else:
            choiced = random.randrange(4,7,2)
            if choiced==4: 
                new_state = 0
            else:
                new_state = 2
        state_list.append(new_state)
        pre_state = new_state
    return state_list
print(simulate_states1(20))

def simulate_states2(states, pi, A, sumofstates):
    states_list = list(states)
    start_state = np.random.choice(states_list, p = pi.ravel())
    state_list = [start_state]
    pre_state = start_state
    for i in range(sumofstates-1):
        next_state = np.random.choice(states_list, p = A[pre_state].ravel())
        state_list.append(next_state)
        pre_state = next_state
    return state_list
print(simulate_states2(states, pi, A, 20))

def chain_prob(pi, A, state_list):
    prob = pi[state_list[0]]
    for i in range(len(state_list)-1):
        prob *= A[state_list[i], state_list[i+1]]
    return prob
print(chain_prob(pi, A, [1,2,1,0,1,0,1]))

observations = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
B = np.array([[1/36, 2/36, 3/36, 4/36, 5/36, 6/36, 5/36, 4/36, 3/36, 2/36, 1/36],
              [1/24, 2/24, 3/24, 4/24, 4/24, 4/24, 3/24, 2/24, 1/24, 0,    0   ],
              [1/16, 2/16, 3/16, 4/16, 3/16, 2/16, 1/16, 0,    0,    0,    0   ]])

def simulate_observ1(state_list):
    observ_list = []
    for i in range(len(state_list)):
        if state_list[i]==0:
            observ_list.append(random.randint(1,6) + random.randint(1,6)) 
        elif state_list[i]==2:
            observ_list.append(random.randint(1,4) + random.randint(1,4))
        else:
            observ_list.append(random.randint(1,4) + random.randint(1,6))
    return observ_list
print(simulate_observ1([1,2,1,0,1,0,1]))

def simulate_observ2(B, state_list):
    observ_list = []
    for i in range(len(state_list)):
        observ = np.random.choice(list(observations), p = B[state_list[i]].ravel())
        observ_list.append(observ)
    return observ_list
print(simulate_observ2(B, [1,2,1,0,1,0,1]))

# 前向算法
def forward(pi, A, B, observ_list):
    N = A.shape[0] # 状态总数
    T = len(observ_list) # 列表长度
    table = np.zeros((N+1,T)) # 前向算法计算过程示例中的表
    table[0:N, 0] = pi * B[:, observ_list[0]-2] # 表中时刻1列的值，也就是前向算法的初始值
    for i in range(N):
        table[N, 0] += table[i, 0]
    for t in range(1,T):
        for i in range(N):
            table[i, t] = np.dot(table[0:N, t-1], A[:, i]) * B[i, observ_list[t]-2]
            table[N, t] += table[i, t]
    return table
print(forward(pi, A, B, [6,2,7,10,3,8,8]))

# 维特比算法
def viterbi(pi, A, B, observ_list):
    N = A.shape[0]
    T = len(observ_list)
    delta = np.zeros((N, T))
    psi = np.zeros((N, T-1), dtype=int)
    delta[:, 0] = pi * B[:,observ_list[0]-2]
    for t in range(1, T):
        for n in range(N):
            delta_t = delta[:, t-1] * A[:, n] * B[n, observ_list[t]-2]
            delta[n, t] = np.max(delta_t)
            psi[n, t-1] = np.argmax(delta_t)
    pre_state = np.argmax(delta[:, T-1])
    state_list = [pre_state]
    for t in range(T-2, -1, -1):
        pre_state = psi[pre_state, t]
        state_list.append(pre_state)
    return delta, state_list
delta, state_list = viterbi(pi, A, B, [6,2,7])
print(delta)
print(state_list)

