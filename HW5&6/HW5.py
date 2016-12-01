#! /usr/bin/env python

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from operator import itemgetter
import random

# select random points
def some(x, n):
    return x.ix[random.sample(x.index, n)]

data = pd.read_csv('wine.data.txt', header=None)
labels = data[0]
del data[0]

# normalise data
means = data.mean()
stdevs = data.std()
z = (data - means) / stdevs

# principal component analysis
pca = PCA(n_components = 2, random_state = 10)
pca.fit(z)
reduced = pd.DataFrame(pca.transform(z))

# data file
writefile = open('data.txt', 'w')

# write PCA data
writefile.write('components\n' + str(pca.components_) + '\n')
writefile.write('explained variance\n' + str(pca.explained_variance_ratio_) + '\n')

# actual results
truecluster = pd.DataFrame(pca.transform(z))
truecluster['cluster'] = labels
ax = truecluster[truecluster['cluster'] == 1].plot(x = 0, y = 1, color = 'Red', kind = 'scatter')
truecluster[truecluster['cluster'] == 2].plot(x = 0, y = 1, color = 'Green', kind = 'scatter', ax = ax)
truecluster[truecluster['cluster'] == 3].plot(x = 0, y = 1, color = 'Blue', kind = 'scatter', ax = ax)

plt.savefig('hw5true.png')
plt.close()

# kmeans algorithm
def kmeans(data, centroids, n):
    for j in xrange(n):
        # find distance between each centroid and point
        # then square and sum
        distances = pd.DataFrame([((data - centroids.iloc[i])*(data - centroids.iloc[i])).sum(axis = 1) for i in xrange(3)]).transpose()

        # set cluster attribute to centroid with minimum distance from point
        data['cluster'] = distances.idxmin(axis = 1)

        # new clusters as average
        centroids = pd.DataFrame([data[data['cluster'] == i].drop('cluster', axis = 1).mean() for i in xrange(3)])

    # find bss
    # sum squares of centroid locations
    # then sum by weight
    bss = ((centroids**2).sum(axis=1) * [len(data[data['cluster'] == i]) for i in xrange(3)]).sum()

    clusterinf = data['cluster']
    del data['cluster']
    # centroids, cluster data, and BSS
    return centroids, clusterinf, bss

# note that z is centred around zero in all coordinates since mean was subtracted
# SSE is then just each element of z squared, and summed
# with mean zero, this is just the sum of variances
# but variance was already normalized to 1 for each feature
# so SSE = 13

# initial centroids

# first run
# three random points, run K-means 500 times
first = some(z, 3)
clust = kmeans(z, first, 500)

# set cluster attribute
reduced['cluster'] = clust[1]

# plot
ax = reduced[reduced['cluster'] == 0].plot(x = 0, y = 1, color = 'Red', kind = 'scatter')
reduced[reduced['cluster'] == 1].plot(x = 0, y = 1, color = 'Green', kind = 'scatter', ax = ax)
reduced[reduced['cluster'] == 2].plot(x = 0, y = 1, color = 'Blue', kind = 'scatter', ax = ax)

plt.savefig('hw5pt2a.png')
plt.close()

indices = []
# this finds the corresponding indices for the true cluster vs reduced by finding the most matches
indices = [max(
    enumerate([((reduced['cluster']+j-i == (truecluster['cluster'] - 1)) &
        (reduced['cluster']+j-i == j)).sum() for j in xrange(3)]),
    key=itemgetter(1))[0] for i in xrange(3)]


# check matches
reduced['fails'] = (map(lambda i: indices[i]+1, reduced['cluster']) == truecluster['cluster'])
# if non-matches, these are non-empty.
a = reduced[(reduced['fails'] == 0) & (reduced['cluster'] == 0)]
b = reduced[(reduced['fails'] == 0) & (reduced['cluster'] == 1)]
c = reduced[(reduced['fails'] == 0) & (reduced['cluster'] == 2)]

# plot
if len(a)>0:
    ax = a.plot(x = 0, y = 1, color = 'Red', kind = 'scatter')
    if len(b)>0: b.plot(x = 0, y = 1, color = 'Green', kind = 'scatter', ax=ax)
    if len(c)>0: c.plot(x = 0, y = 1, color = 'Blue', kind = 'scatter', ax=ax)
elif len(b)>0:
    ax = b.plot(x = 0, y = 1, color = 'Green', kind = 'scatter')
    if len(c)>0: c.plot(x = 0, y = 1, color = 'Blue', kind = 'scatter', ax=ax)
elif len(c)>0: c.plot(x = 0, y = 1, color = 'Blue', kind = 'scatter')

plt.savefig('hw5pt2afail.png')
plt.close()

# best of 10
inits = []
largest = 0
writefile.write('\npart 2 kmeans bss\n')
for i in xrange(10):
    centroids = some(z, 3)
    inits += [centroids]
    clust = kmeans(z, centroids, 500)

    ax = reduced[reduced['cluster'] == 0].plot(x = 0, y = 1, color = 'Red', kind = 'scatter')
    reduced[reduced['cluster'] == 1].plot(x = 0, y = 1, color = 'Green', kind = 'scatter', ax = ax)
    reduced[reduced['cluster'] == 2].plot(x = 0, y = 1, color = 'Blue', kind = 'scatter', ax = ax)

    plt.savefig('hw5pt2no'+str(i)+'.png')
    plt.close()
    writefile.write('trial no ' + str(i) + ': ' + str(clust[2]) + '\n')

    if clust[2] >= largest:
        largest = clust[2]
        best = clust[1]

reduced['cluster'] = best

ax = reduced[reduced['cluster'] == 0].plot(x = 0, y = 1, color = 'Red', kind = 'scatter')
reduced[reduced['cluster'] == 1].plot(x = 0, y = 1, color = 'Green', kind = 'scatter', ax = ax)
reduced[reduced['cluster'] == 2].plot(x = 0, y = 1, color = 'Blue', kind = 'scatter', ax = ax)

plt.savefig('hw5pt2b.png')
plt.close()

indices = []
# this finds the corresponding indices for the true cluster vs reduced by finding the most matches
indices = [max(
    enumerate([((reduced['cluster']+j-i == (truecluster['cluster'] - 1)) &
        (reduced['cluster']+j-i == j)).sum() for j in xrange(3)]),
    key=itemgetter(1))[0] for i in xrange(3)]


# check matches
reduced['fails'] = (map(lambda i: indices[i]+1, reduced['cluster']) == truecluster['cluster'])
# if non-matches, these are non-empty.
a = reduced[(reduced['fails'] == 0) & (reduced['cluster'] == 0)]
b = reduced[(reduced['fails'] == 0) & (reduced['cluster'] == 1)]
c = reduced[(reduced['fails'] == 0) & (reduced['cluster'] == 2)]

# plot
if len(a)>0:
    ax = a.plot(x = 0, y = 1, color = 'Red', kind = 'scatter')
    if len(b)>0: b.plot(x = 0, y = 1, color = 'Green', kind = 'scatter', ax=ax)
    if len(c)>0: c.plot(x = 0, y = 1, color = 'Blue', kind = 'scatter', ax=ax)
elif len(b)>0:
    ax = b.plot(x = 0, y = 1, color = 'Green', kind = 'scatter')
    if len(c)>0: c.plot(x = 0, y = 1, color = 'Blue', kind = 'scatter', ax=ax)
elif len(c)>0: c.plot(x = 0, y = 1, color = 'Blue', kind = 'scatter')

plt.savefig('hw5pt2bfail.png')
plt.close()

writefile.write('\npart 3 kmeans bss\n')
for j in xrange(10):
    skk = KMeans(n_clusters = 3, init = inits[j], max_iter = 500).fit(z)
    reduced['cluster'] = pd.DataFrame(skk.predict(z))

    # plot
    ax = reduced[reduced['cluster'] == 0].plot(x = 0, y = 1, color = 'Red', kind = 'scatter')
    reduced[reduced['cluster'] == 1].plot(x = 0, y = 1, color = 'Green', kind = 'scatter', ax = ax)
    reduced[reduced['cluster'] == 2].plot(x = 0, y = 1, color = 'Blue', kind = 'scatter', ax = ax)

    # find bss
    # sum squares of centroid locations
    # then sum by weight
    bss = ((pd.DataFrame(skk.cluster_centers_)**2).sum(axis=1) * [len(reduced[reduced['cluster'] == i]) for i in xrange(3)]).sum()

    plt.savefig('hw5pt3no'+str(j)+'.png')
    plt.close()
    writefile.write('trial no ' + str(j) + ': ' + str(bss) + '\n')

writefile.close()
