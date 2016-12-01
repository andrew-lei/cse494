#! /usr/bin/env python

import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import random
from scipy.spatial.distance import pdist, squareform
import numpy as np

# read, extract labels
data = pd.read_csv('wine.data.txt', header=None)
labels = data[0]
del data[0]

# normalise data
means = data.mean()
stdevs = data.std()
z = (data - means) / stdevs
pca = PCA(n_components = 2, random_state = 10)
pca.fit(z)

l = len(data)

# linking: min for single-link, max for complete
def heirclust(data, length, nclust, linking):
    distances = squareform(pdist(data, 'euclidean'))
    # make finding the min easier
    np.fill_diagonal(distances, 10.**10)
    heirclusters = [set([i]) for i in xrange(length)]

    for j in xrange(length - nclust):
        minpair = np.unravel_index(distances.argmin(), distances.shape)
        for i in xrange(length):
            distances[i][min(minpair)] = linking(distances[i][min(minpair)], distances[i][max(minpair)])
            distances[min(minpair)][i] = distances[i][min(minpair)]

        heirclusters[min(minpair)] = heirclusters[min(minpair)].union(heirclusters[max(minpair)])
        del heirclusters[max(minpair)]

        distances[min(minpair)][min(minpair)] = 10.**10
        distances = np.delete(distances, (max(minpair)), axis = 0)
        distances = np.delete(distances, (max(minpair)), axis = 1)
        length -= 1

    return heirclusters, distances

hc = pd.DataFrame(pca.transform(z))
heirclusters, distances = heirclust(z, l, 4, min)
hc['cluster'] = pd.DataFrame([i for j in xrange(len(data)) for i,x in enumerate(heirclusters) if j in x])

ax = hc[hc['cluster'] == 0].plot(x = 0, y = 1, color = 'DarkBlue', kind = 'scatter')
hc[hc['cluster'] == 1].plot(x = 0, y = 1, color = 'Green', kind = 'scatter', ax = ax)
hc[hc['cluster'] == 2].plot(x = 0, y = 1, color = 'Red', kind = 'scatter', ax = ax)
hc[hc['cluster'] == 3].plot(x = 0, y = 1, color = 'LightBlue', kind = 'scatter', ax = ax)

plt.savefig('HW6single4.png')

minpair = np.unravel_index(distances.argmin(), distances.shape)
for i in xrange(4):
    distances[i][min(minpair)] = min(distances[i][min(minpair)], distances[i][max(minpair)])
    distances[min(minpair)][i] = distances[i][min(minpair)]

heirclusters[min(minpair)] = heirclusters[min(minpair)].union(heirclusters[max(minpair)])
del heirclusters[max(minpair)]

hc['cluster'] = pd.DataFrame([i for j in xrange(len(data)) for i,x in enumerate(heirclusters) if j in x])

ax = hc[hc['cluster'] == 0].plot(x = 0, y = 1, color = 'Blue', kind = 'scatter')
hc[hc['cluster'] == 1].plot(x = 0, y = 1, color = 'Green', kind = 'scatter', ax = ax)
hc[hc['cluster'] == 2].plot(x = 0, y = 1, color = 'Red', kind = 'scatter', ax = ax)

plt.savefig('HW6single3.png')

heirclusters, distances = heirclust(z, l, 4, max)
hc['cluster'] = pd.DataFrame([i for j in xrange(len(data)) for i,x in enumerate(heirclusters) if j in x])

ax = hc[hc['cluster'] == 0].plot(x = 0, y = 1, color = 'DarkBlue', kind = 'scatter')
hc[hc['cluster'] == 1].plot(x = 0, y = 1, color = 'Red', kind = 'scatter', ax = ax)
hc[hc['cluster'] == 2].plot(x = 0, y = 1, color = 'LightBlue', kind = 'scatter', ax = ax)
hc[hc['cluster'] == 3].plot(x = 0, y = 1, color = 'Green', kind = 'scatter', ax = ax)

plt.savefig('HW6complete4.png')

minpair = np.unravel_index(distances.argmin(), distances.shape)
for i in xrange(4):
    distances[i][min(minpair)] = max(distances[i][min(minpair)], distances[i][max(minpair)])
    distances[min(minpair)][i] = distances[i][min(minpair)]

heirclusters[min(minpair)] = heirclusters[min(minpair)].union(heirclusters[max(minpair)])
del heirclusters[max(minpair)]

hc['cluster'] = pd.DataFrame([i for j in xrange(len(data)) for i,x in enumerate(heirclusters) if j in x])

ax = hc[hc['cluster'] == 0].plot(x = 0, y = 1, color = 'Blue', kind = 'scatter')
hc[hc['cluster'] == 1].plot(x = 0, y = 1, color = 'Red', kind = 'scatter', ax = ax)
hc[hc['cluster'] == 2].plot(x = 0, y = 1, color = 'Green', kind = 'scatter', ax = ax)

plt.savefig('HW6complete3.png')
