from matplotlib.pyplot import axis
import numpy as np

cdata = np.array([
    [87.9209, 7.28955, 0.92849, 0.92598, 272.59184, 2.21299, 6.93105, 0.1354],
    [59.14201, 6.38073, 0.95273, 0.9502, 250.79649, 6.84505, 5.96232, 0.11008]
])

n = cdata.shape[0]

# normalizing the weight matrix
P = cdata / np.sum(cdata, axis=0)
# calculating the entropy vector
e = (-1) * np.sum(P * np.log(P), axis=0) / np.log(n)
# calculating the degree of diversification
d = 1-e
# calculating the weights
w = d / np.sum(d)

# CALCULATED THE WEIGHT MATRIX -----------------------------------------------
# CALCULATING THE RANK -------------------------------------------------------
y = cdata / np.sqrt(np.sum(cdata**2, axis=0))
v = w * y

vpos = []
vneg = []

criteria_map = {
}

stringCriterias = ['mse', 'mae', 'r2', 'adjr2','aic', 'pp', 'meop', 'theil',]
for i in range(8):
    cr = stringCriterias[i]
    criteria_map[cr] = v[:, i]

# for i in criteria_map:
#     print(i, criteria_map[i])

maximizer = ['r2', 'adjr2']
minimizer = ['mse', 'mae', 'pp', 'meop', 'aic', 'theil']


# print()

for criteria in criteria_map:
    if criteria in maximizer:
        best = np.amax(criteria_map[criteria])
        worst = np.amin(criteria_map[criteria])
        
        vpos.append(best)
        vneg.append(worst)
    if criteria in minimizer:
        best = np.amin(criteria_map[criteria])
        worst = np.amax(criteria_map[criteria])
        
        vpos.append(best)
        vneg.append(worst)

# converting ideal best and ideal worst data to usable form
vpos = np.array(vpos)
vneg = np.array(vneg)

print(np.sqrt(np.sum((v - vpos)**2, axis=1)))
print()
print(np.sqrt(np.sum((v - vneg)**2, axis=1)))



input()