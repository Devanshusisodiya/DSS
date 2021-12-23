import numpy as np

# criterias are entered as follows
# mse, mae, r2, adj r2, pp, meop
data = [
    ['go',      88.26, 7.65, 0.928, 0.925, 2.33, 7.52],
    ['delayed', 135.6, 10.02, 0.889, 0.885, 6.39, 9.85],
    ['inflection', 89.95, 7.37, 0.928, 0.924, 2.23, 7.24],
    ['yamada ray', 198.17, 10.59, 0.84, 0.83, 6.30, 10.41]
]

# getting just the numerical values
cdata = []
for modelData in data:
    cdata.append(modelData[1:])
cdata = np.array(cdata)

n = cdata.shape[0] # number of models
m = cdata.shape[1] # number of criterias

# SO FAR SO GOOD
# CALCULATING THE WEIGHT MATRIX ----------------------------------------------

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
    'mse': v[:, 0],
    'mae': v[:, 1],
    'r2': v[:, 2],
    'adjr2': v[:, 3],
    'pp': v[:, 4],
    'meop': v[:, 5]
}
maximizer = ['r2', 'adjr2']
minimizer = ['mse', 'mae', 'pp', 'meop']

for criteria in criteria_map:
    if criteria in maximizer:
        best = max(criteria_map[criteria])
        worst = min(criteria_map[criteria])
        vpos.append(best)
        vneg.append(worst)
    if criteria in minimizer:
        best = min(criteria_map[criteria])
        worst = max(criteria_map[criteria])
        vpos.append(best)
        vneg.append(worst)

# converting ideal best and ideal worst data to usable form
vpos = np.array(vpos)
vneg = np.array(vneg)

spos = np.sqrt(np.sum( (v-vpos)**2 , axis=1))
sneg = np.sqrt(np.sum( (v-vneg)**2 , axis=1))

# final relative closeness results
c = sneg / (spos + sneg)

# ranking the models
ranked = {}
rankArr = np.array([0 for _ in range(n)])
initialRank = 1

for _ in range(n):
    # populating rank list
    index = np.argmin(c)
    ranked[index] = initialRank
    # discarding the considered index
    c[index] = np.inf
    initialRank += 1

for i in range(n):
    rankArr[i] = ranked[i]

rankArr = np.array([[i] for i in rankArr])
# CALCULATED THE RANKS -----------------------------------------------

modelsWithRank = []
criteriasWithRank = np.append(cdata, rankArr, 1)

for i in range(len(criteriasWithRank)):
    row = list(criteriasWithRank[i])
    row.insert(0, data[i][0])
    modelsWithRank.append(row)

print(modelsWithRank)