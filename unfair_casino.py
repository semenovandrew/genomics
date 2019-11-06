import numpy as np
import pandas as pd  # только для оформления рамок в матрице
from viterbi import viterbii
from hmmlearn import hmm
from forward_and_back_algorithm import forward, backward, likelihood, posterior_prob, pbwd
import matplotlib.pyplot as plt
from baum_welch import baum_post, baum_welch_trans, baum_welch_emmis, gamma
import time

h = 2  # number of hidden states
paa = 0.95
pab = 0.05
pba = 0.1
pbb = 0.9
#  L = int(input('Enter a length of a sequence: '))
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
emmit = np.array(([1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6], [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]))
emmit1 = pd.DataFrame(emmit, columns=events, index=dices)
print('\nEMISSION MATRIX', emmit1, sep='\n ')


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
kok = 0
for j in range(visible):
    total += emmit[b][j]
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
    koko = 0
    for j in range(visible):
        totaln += emmit[b][j]
        if out_count <= totaln:
            sequence.append([events[j], j])
            break


print('\n VISIBLE OUTCOMES:\n', sequence)
print('\n INVISIBLE OUTCOMES(our prediction):\n', counter_for_dices, '\n')

'''
# viterbi algorithm
hid = viterbii(sequence, dices_start, emmit, transition_matrix, h, L)[0]
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
model.emissionprob_ = emmit
X = model.sample(L)
print('\n FROM HMM LEARN:\n', X[1])  # последовательность скрытых состояний
states = [0, 1]

print('\nFORWARD ALGORITHM:\n', forward(sequence, dices_start, transition_matrix, emmit, h, L), '\n')

print('\nBACKWARD ALGORITHM:\n', backward(sequence, transition_matrix, emmit, h, L), '\n')

print('\nP(X), forward:\n', likelihood(sequence, dices_start, transition_matrix, emmit, h, L), '\n')

print('P(X), backward:\n', pbwd(sequence, transition_matrix, emmit, h, L))

print('\nPOSTERIOR PROBABILITY:\n', posterior_prob(sequence, dices_start, transition_matrix, emmit, h, L), '\n')


for i in range(L):
    X[0][i] = sequence[i][1]

print('\nPOSTERIOR PROBABILITY FROM HMMLEARN:\n', model.predict_proba(X[0]))

print('\nPOSTERIOR PROBABILITY(BW):\n', baum_post(sequence, dices_start, transition_matrix, emmit, h, L), '\n')

print('\n INVISIBLE OUTCOMES(our prediction):\n', counter_for_dices, '\n')
print('\n VITERBI ALGORITHM\n', hid_seq, '\n')
print('\n FROM HMM LEARN:\n', X[1])

P = posterior_prob(sequence, dices_start, transition_matrix, emmit, h, L)
g1 = plt.plot(ind, counter_for_dices, linestyle='--')
g2 = plt.plot(ind, P[:, 0])
plt.show()


'''
#   find answer
trans = []
emiss = []

threshhold = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001


start_time = time.time()

anew = baum_welch_trans(sequence, dices_start, transition_matrix, emmit, h, L)
trans.append(anew.tolist())
bnew = baum_welch_emmis(sequence, dices_start, transition_matrix, emmit, h, visible, L)
emiss.append(bnew.tolist())
anew = baum_welch_trans(sequence, dices_start, transition_matrix, emmit, h, L)
trans.append(anew.tolist())
bnew = baum_welch_emmis(sequence, dices_start, transition_matrix, emmit, h, visible, L)
emiss.append(bnew.tolist())


it = 1
time_all = 0
while abs(emiss[it][0][0] - emiss[it - 1][0][0]) > threshhold and \
        abs(emiss[it][1][0] - emiss[it - 1][1][0]) > threshhold and \
        abs(emiss[it][0][2] - emiss[it - 1][0][2]) > threshhold and \
        abs(emiss[it][0][4] - emiss[it - 1][0][4]) > threshhold and \
        abs(emiss[it][0][5] - emiss[it - 1][0][5]) > threshhold and \
        abs(emiss[it][1][1] - emiss[it - 1][1][1]) > threshhold and \
        abs(emiss[it][1][2] - emiss[it - 1][1][2]) > threshhold and \
        abs(emiss[it][1][3] - emiss[it - 1][1][3]) > threshhold and \
        abs(emiss[it][1][4] - emiss[it - 1][1][4]) > threshhold and \
        abs(emiss[it][1][5] - emiss[it - 1][1][5]) > threshhold and \
        abs(trans[it][0][0] - trans[it - 1][0][0]) > threshhold and \
        abs(trans[it][0][1] - trans[it - 1][0][1]) > threshhold and \
        abs(trans[it][1][0] - trans[it - 1][1][0]) > threshhold and \
        abs(trans[it][1][1] - trans[it - 1][1][1]) > threshhold:

    anew = baum_welch_trans(sequence, dices_start, transition_matrix, emmit, h, L)
    trans.append(anew.tolist())
    bn1 = baum_welch_emmis(sequence, dices_start, transition_matrix, emmit, h, visible, L)
    emiss.append(bn1.tolist())
    print(transition_matrix)
    print(bn1)
    it += 1
    time_all = (time.time() - start_time) / 60


print('\nNumber of iteration: ', it)
print('\nTime: ', time_all, 'minutes')


new_trans = trans[-1]
new_emiss = emiss[-1]
print('\nNEW TRANSITION:\n', new_trans)
print('\nNEW EMISSION:\n', new_emiss)

jop = np.argmax(gamma(sequence, dices_start, transition_matrix, emmit, h, L), axis=1)

pr = 0
for i in range(L):
    if jop[i] == counter_for_dices[i]:
        pr += 1

x = 100 * pr / L
print('\n', x, '%', '- процент совпадений')



print('\nOUR HIDDEN:\n', counter_for_dices, '\n')
print('\nBW HIDDEN:\n', jop, '\n')


# our prediction vs viterbi
ind = np.arange(L)
p1 = plt.bar(ind, counter_for_dices)
p2 = plt.bar(ind, jop, bottom=counter_for_dices)
plt.title('Графическа оценка алгоритма Витерби')
plt.legend((p1[0], p2[0]), ('OUR', 'VA'))
plt.show()


posterior = baum_post(sequence, dices_start, transition_matrix, emmit, h, L)

plt.figure(figsize=(15, 4))
plt.xlabel('Length of the sequence', fontsize=20)
plt.ylabel('$P(x)$', fontsize=20)
plt.title('Posterior probability', fontsize = 20)
plt.plot(range(L), posterior[:, 0], color='black')
plt.plot(range(L), jop, color='blue', linestyle='dotted')
plt.show()

