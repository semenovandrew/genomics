import numpy as np
import pandas as pd  # только для оформления рамок в матрице
from viterbi import viterbii
from hmmlearn import hmm
from forward_and_back_algorithm import forward, backward, likelihood, posterior_prob, pbwd
import matplotlib.pyplot as plt
from baum_welch import baum_post, baum_welch_trans, baum_welch_emmis


m = 6
paa = 0.95
pab = 0.05
pba = 0.1
pbb = 0.9
#L = int(input('Enter a length of a sequence: '))
sequence = []
L = 100
dices = ['Fair Dice', 'Loaded Dice']
dices_start = np.array([0.5, 0.5])
# transition
transition_matrix = np.array([[paa, pab], [pba, pbb]])
transition_matrix1 = pd.DataFrame(transition_matrix, columns=dices, index=dices)
print('\nTRANSITION MATRIX', transition_matrix1, sep='\n ')

# emission
events = np.array(['1', '2', '3', '4', '5', '6'])
emission_matrix = np.array(([1/6, 1/6, 1/6, 1/6, 1/6, 1/6], [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]))
emission_matrix1 = pd.DataFrame(emission_matrix, columns=events, index=dices)
print('\nEMISSION MATRIX', emission_matrix1, sep='\n ')


# randomize and sequence
counter_for_dices = []
count0 = np.random.random()
if 0 <= count0 <= dices_start[0]:
    b = 0
else:
    b = 1
counter_for_dices.append(b)
print(b)

#   improve our randomizer:

out_count0 = np.random.random()
out_count0 = 0.69
b = 1
print(out_count0)

total = 0
g = 0
for i in range(m):
    while out_count0 > total:
        total += emission_matrix[b][i]
        if out_count0 < total:
            if g >= int(1) and emission_matrix[b][g - 1] < out_count0 <= total:
                sequence.append([events[g], g])
            else:
                sequence.append([events[i], i])
        g += 1


print(sequence)


print('\n VISIBLE OUTCOMES:\n', sequence)
print('\n INVISIBLE OUTCOMES(our prediction):\n', counter_for_dices, '\n')
