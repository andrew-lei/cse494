#! /usr/bin/env python

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as sm

# Read the data
data = pd.read_csv('Wholesale customers data.csv')
totals = []
means = []
stderrs = []

# Compute average spending by category
for i in range(1, 4):
    # Get data from region i (1, 2, or 3) for spending on Fresh, Milk, &c.
    regdata = data[data.Region == i][['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']]
    totals += [regdata.sum().as_matrix()]
    means += [regdata.mean().as_matrix()]
    # Compute standard error of mean
    stderrs += [ regdata.std().as_matrix() / np.sqrt(len(regdata)) ]


ind = np.arange(6)  # The x locations for the groups; 0, 1, 2, ..., 5
width = 0.2       # The width of the bars
fig, ax = plt.subplots()
fig.set_size_inches(18.5, 10.5) # Default size is too small

# Bars for bar graph - using total
rects = [ax.bar(ind, totals[0], width, color='r'),
    ax.bar(ind + width, totals[1], width, color='y'),
    ax.bar(ind + 2*width, totals[2], width, color='b')]

# Labels, titles, ticks, &c.
ax.set_xlabel('Product type')
ax.set_ylabel('Spending')
ax.set_title('Total spending by product and region')
ax.set_xticks(ind + 1.5 * width)
ax.set_xticklabels(['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen'])
ax.legend((rects[0][0], rects[1][0], rects[2][0]), ('Lisbon', 'Oporto', 'Other'))

# Save and clear for next image
plt.savefig('part2a.png')
plt.cla()

# Bars for bar graph - using mean
rects = [ax.bar(ind, means[0], width, color='r', yerr=stderrs[0]),
    ax.bar(ind + width, means[1], width, color='y', yerr=stderrs[1]),
    ax.bar(ind + 2*width, means[2], width, color='b', yerr=stderrs[2])]

# Labels, titles, ticks, &c.
ax.set_xlabel('Product type')
ax.set_ylabel('Spending')
ax.set_title('Average spending by product and region')
ax.set_xticks(ind + 1.5 * width)
ax.set_xticklabels(['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen'])
ax.legend((rects[0][0], rects[1][0], rects[2][0]), ('Lisbon', 'Oporto', 'Other'))

# Save and clear for part 3
plt.savefig('part2b.png')
plt.cla()

# Ordinary least squares fit for Grocery (independent) and Detergents_Paper(dependent)
fit = sm.ols(formula="Detergents_Paper ~ Grocery", data=data).fit()
ind = np.arange(min(data['Grocery']), max(data['Grocery']) )

print fit.summary()
print fit.pvalues # Print separately because very close to zero

# Plot points and best-fit
ax.scatter(data['Grocery'], data['Detergents_Paper'])
ax.plot(ind, fit.params[0] + ind*fit.params[1])

# Labels and title
ax.set_xlabel('Grocery spending')
ax.set_ylabel('Detergent and paper product spending')
ax.set_title('Spending on detergents and paper products vs. grocery spending')

# Save
plt.savefig('part3.png')
