import numpy as np
import pandas as pd  # только для оформления рамок в матрице
from viterbi import viterbii
from hmmlearn import hmm
from forward_and_back_algorithm import forward, backward, likelihood, posterior_prob, pbwd
import matplotlib.pyplot as plt
from baum_welch import baum_post, baum_welch

m = 2
paa1 = np.random.random()
pab1 = np.random.random()
pabc = 1 / (paa1 + pab1)
paa = pabc * paa1
pab = pabc * pab1
pba1 = np.random.random()
pbb1 = np.random.random()
pbac = 1 / (pba1 + pbb1)
pba = pba1 * pbac
pbb = pbb1 * pbac
paa = 0.3
pab = 0.7
pba = 0.6
pbb = 0.4
#L = int(input('Enter a length of a sequence: '))
L = 5
sequence = []
boxes = ['Box 1', 'Box 2']
boxes_prob = np.array([0.5, 0.5])
# transition
transition_matrix = np.array([[paa, pab], [pba, pbb]])
transition_matrix1 = pd.DataFrame(transition_matrix, columns=boxes, index=boxes)
print('\nTRANSITION MATRIX', transition_matrix1, sep='\n ')

# emission
x11 = np.random.random()
x12 = np.random.random()
x13 = np.random.random()
aa = 1 / (x11 + x12 + x13)
a11 = x11 * aa
a12 = x12 * aa
a13 = x13 * aa
x21 = np.random.random()
x22 = np.random.random()
x23 = np.random.random()
aa1 = 1 / (x21 + x22 + x23)
a21 = x21 * aa1
a22 = x22 * aa1
a23 = x23 * aa1
a11 = 0.2
a12 = 0.3
a13 = 0.5
a21 = 0.6
a22 = 0.1
a23 = 0.3
events = np.array(['Red', 'Green', 'Blue'])
nevents = 3
emission_matrix = np.array([[a11, a12, a13], [a21, a22, a23]])
emission_matrix1 = pd.DataFrame(emission_matrix, columns=events, index=boxes)
print('\nEMISSION MATRIX', emission_matrix1, sep='\n ')

# randomize and sequence
counter_for_boxes = []
count0 = np.random.random()
if 0 <= count0 <= boxes_prob[0]:
    b = 0
else:
    b = 1
counter_for_boxes.append(b)
check = np.random.random()
if 0 <= check <= emission_matrix[b, 0]:
    sequence.append([events[0], 0])
elif emission_matrix[b, 0] < check <= emission_matrix[b, 1]:
    sequence.append([events[1], 1])
else:
    sequence.append([events[2], 2])

# f for current box
for i in range(1, L):
    count = np.random.random()
    if 0 <= count < transition_matrix[b, 0]:
        b = 0
    elif transition_matrix[b, 0] <= count <= 1:
        b = 1
    counter_for_boxes.append(b)
    check1 = np.random.random()
    if 0 <= check1 <= emission_matrix[b, 0]:
        sequence.append([events[0], 0])
    elif emission_matrix[b, 0] < check1 <= emission_matrix[b, 1]:
        sequence.append([events[1], 1])
    else:
        sequence.append([events[2], 2])

print('\n VISIBLE OUTCOMES:\n', sequence)
print('\n INVISIBLE OUTCOMES(our prediction):\n', counter_for_boxes, '\n')


# viterbi algorithm
hid = viterbii(sequence, boxes_prob, emission_matrix, transition_matrix, m, L)[0]
hid_seq = np.argmax(hid, axis=1)  # искомая последовательность скрытых состояний
print('\n VITERBI ALGORITHM\n', hid_seq, '\n')


# from hmmlearn we get:
model = hmm.MultinomialHMM(n_components=m)
model.startprob_ = boxes_prob
model.transmat_ = transition_matrix
model.emissionprob_ = emission_matrix
X = model.sample(L)
print('\n FROM HMM LEARN:\n', X[1])  # последовательность скрытых состояний
states = [0, 1]


print('\nFORWARD ALGORITHM:\n', forward(sequence, boxes_prob, transition_matrix, emission_matrix, m, L), '\n')

print('\nBACKWARD ALGORITHM:\n', backward(sequence, transition_matrix, emission_matrix, m, L), '\n')

print('\nP(X), forward:\n', likelihood(sequence, boxes_prob, transition_matrix, emission_matrix, m, L), '\n')

print('P(X), backward:\n', pbwd(sequence, transition_matrix, emission_matrix, m, L))

print('\nPOSTERIOR PROBABILITY:\n', posterior_prob(sequence, boxes_prob, transition_matrix, emission_matrix, m, L), '\n')

for i in range(L):
    X[0][i] = sequence[i][1]

print('\nPOSTERIOR PROBABILITY FROM HMMLEARN:\n', model.predict_proba(X[0]))

print('\nPOSTERIOR PROBABILITY(BW):\n', baum_post(sequence, boxes_prob, transition_matrix, emission_matrix, m, L), '\n')

pi = posterior_prob(sequence, boxes_prob, transition_matrix, emission_matrix, m, L)
fandb_hidden = np.argmax(pi, axis=1)

print('\n INVISIBLE OUTCOMES(our prediction):\n', counter_for_boxes, '\n')
print('\n VITERBI ALGORITHM\n', hid_seq, '\n')
print('\n FROM HMM LEARN:\n', X[1])
print('\n HIDDEN SEQUENCE FROM FORWARD AND BACKWARD ALGORITHMS:\n', fandb_hidden, '\n')


#   baum welch training

#bwhidden = np.argmax(bwpost, axis=1)
#print('\n HIDDEN SEQUENCE FROM BAUM-WELCH ALGORITHM:\n', bwhidden, '\n')

# our prediction vs viterbi
ind = np.arange(L)
p1 = plt.bar(ind, counter_for_boxes)
p2 = plt.bar(ind, hid_seq, bottom=counter_for_boxes)
p3 = plt.bar(ind, X[1], bottom=hid_seq)
plt.title('Графическа оценка алгоритма Витерби')
plt.legend((p1[0], p2[0], p3[0]), ('OUR', 'VA', 'HMM', 'BW'))
#plt.show()
v = np.array([0] * L)
for i in range(L):
    v[i] = sequence[i][1]
