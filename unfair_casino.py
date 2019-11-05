import numpy as np
import pandas as pd  # только для оформления рамок в матрице
from viterbi import viterbii
from hmmlearn import hmm
from forward_and_back_algorithm import forward, backward, likelihood, posterior_prob, pbwd
import matplotlib.pyplot as plt
from baum_welch import baum_post, baum_welch_trans, baum_welch_emmis


h = 2  # number of hidden states
paa = 0.95
pab = 0.05
pba = 0.1
pbb = 0.9
#L = int(input('Enter a length of a sequence: '))
sequence = []
L = 300
visible = 6   # number of visible states
dices = ['Fair Dice', 'Loaded Dice']
dices_start = np.array([0.5, 0.5])
# transition
transition_matrix = np.array([[paa, pab], [pba, pbb]])
transition_matrix1 = pd.DataFrame(transition_matrix, columns=dices, index=dices)
print('\nTRANSITION MATRIX', transition_matrix1, sep='\n ')

# emission
events = ['1', '2', '3', '4', '5', '6']
emission_matrix = np.array(([1/6, 1/6, 1/6, 1/6, 1/6, 1/6], [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]))
emission_matrix1 = pd.DataFrame(emission_matrix, columns=events, index=dices)
print('\nEMISSION MATRIX', emission_matrix1, sep='\n ')


# randomize and sequence
counter_for_dices = []
count0 = np.random.random()
if 0 <= count0 < dices_start[0]:
    b = 0
else:
    b = 1
counter_for_dices.append(b)

#   improve our randomizer:

out_count0 = np.random.random()

total = 0
for j in range(visible):
    total += emission_matrix[b][j]
    if out_count0 <= total:
        sequence.append([events[j], j])
        break


#   next steps
for i in range(1, L):
    count = np.random.random()
    if 0 <= count < transition_matrix[b][0]:
        b = 0
    else:
        b = 1
    counter_for_dices.append(b)

    out_count = np.random.random()
    totaln = 0
    for j in range(visible):
        totaln += emission_matrix[b][j]
        if out_count <= totaln:
            sequence.append([events[j], j])
            break


print('\n VISIBLE OUTCOMES:\n', sequence)
print('\n INVISIBLE OUTCOMES(our prediction):\n', counter_for_dices, '\n')


# viterbi algorithm
hid = viterbii(sequence, dices_start, emission_matrix, transition_matrix, h, L)[0]
hid_seq = np.argmax(hid, axis=1)  # искомая последовательность скрытых состояний
print('\n VITERBI ALGORITHM\n', hid_seq, '\n')

ind = np.arange(L)
p1 = plt.bar(ind, counter_for_dices)
p2 = plt.bar(ind, hid_seq, bottom=counter_for_dices)
plt.title('Графическа оценка алгоритма Витерби')
plt.legend((p1[0], p2[0]), ('OUR', 'VA'))
plt.show()


pr = 0
for i in range(L):
    if hid_seq[i] == counter_for_dices[i]:
        pr += 1

x = 100 * pr / L
print(x, '%', '- процент совпадений')




# from hmmlearn we get:
model = hmm.MultinomialHMM(n_components=h)
model.startprob_ = dices_start
model.transmat_ = transition_matrix
model.emissionprob_ = emission_matrix
X = model.sample(L)
print('\n FROM HMM LEARN:\n', X[1])  # последовательность скрытых состояний
states = [0, 1]

print('\nFORWARD ALGORITHM:\n', forward(sequence, dices_start, transition_matrix, emission_matrix, h, L), '\n')

print('\nBACKWARD ALGORITHM:\n', backward(sequence, transition_matrix, emission_matrix, h, L), '\n')

print('\nP(X), forward:\n', likelihood(sequence, dices_start, transition_matrix, emission_matrix, h, L), '\n')

print('P(X), backward:\n', pbwd(sequence, transition_matrix, emission_matrix, h, L))

print('\nPOSTERIOR PROBABILITY:\n', posterior_prob(sequence, dices_start, transition_matrix, emission_matrix, h, L), '\n')


for i in range(L):
    X[0][i] = sequence[i][1]

print('\nPOSTERIOR PROBABILITY FROM HMMLEARN:\n', model.predict_proba(X[0]))

print('\nPOSTERIOR PROBABILITY(BW):\n', baum_post(sequence, dices_start, transition_matrix, emission_matrix, h, L), '\n')

print('\n INVISIBLE OUTCOMES(our prediction):\n', counter_for_dices, '\n')
print('\n VITERBI ALGORITHM\n', hid_seq, '\n')
print('\n FROM HMM LEARN:\n', X[1])

P = posterior_prob(sequence, dices_start, transition_matrix, emission_matrix, h, L)
print(P)
g1 = plt.plot(ind, counter_for_dices, linestyle='--')
g2 = plt.plot(ind, P[:, 1])
plt.show()



#   find answer
trans = []
emiss = []

threshhold = 0.000000001

anew = baum_welch_trans(sequence, dices_start, transition_matrix, emission_matrix, h, L)
trans.append(anew.tolist())
bnew = baum_welch_emmis(sequence, dices_start, transition_matrix, emission_matrix, h, visible, L)
emiss.append(bnew.tolist())
anew = baum_welch_trans(sequence, dices_start, transition_matrix, emission_matrix, h, L)
trans.append(anew.tolist())
bnew = baum_welch_emmis(sequence, dices_start, transition_matrix, emission_matrix, h, visible,  L)
emiss.append(bnew.tolist())

it = 1
while abs(trans[it][0][0] - trans[it - 1][0][0]) > threshhold:
    anew = baum_welch_trans(sequence, dices_start, transition_matrix, emission_matrix, h, L)
    trans.append(anew.tolist())
    bnew = baum_welch_emmis(sequence, dices_start, transition_matrix, emission_matrix, h, visible, L)
    emiss.append(bnew.tolist())
    it += 1

print('\nNumber of iteration: ', it)


new_trans = trans[-1]
new_emiss = emiss[-1]
print('\nNEW TRANSITION:\n', new_trans)
print('\nNEW EMISSION:\n', new_emiss)


true_hid = viterbii(sequence, dices_start, new_emiss, new_trans, h, L)[0]
true_hid_states = np.argmax(true_hid, axis=1)
print('\nOUR HIDDEN:\n', counter_for_dices, '\n')
print('\nBW HIDDEN:\n', true_hid_states, '\n')


# our prediction vs viterbi
ind = np.arange(L)
p1 = plt.bar(ind, counter_for_dices)
p2 = plt.bar(ind, true_hid_states, bottom=counter_for_dices)
plt.title('Графическа оценка алгоритма Витерби')
plt.legend((p1[0], p2[0]), ('OUR', 'VA'))
plt.show()


pr = 0
for i in range(L):
    if true_hid_states[i] == counter_for_dices[i]:
        pr += 1

x = 100 * pr / L
print(x, '%', '- процент совпадений')


posterior = baum_post(sequence, dices_start, transition_matrix, emission_matrix, h, L)

plt.figure(figsize=(15, 4))
plt.xlabel('Length of the sequence', fontsize=20)
plt.ylabel('$P(x)$', fontsize=20)
plt.title('Posterior probability', fontsize = 20)
plt.plot(range(L), posterior[:, 1], color='black')
plt.plot(range(L), true_hid_states, color='blue', linestyle='dotted')
plt.show()
