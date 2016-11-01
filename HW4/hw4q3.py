#! /usr/bin/env python

from senti_classifier import senti_classifier

tweet_file = open('tweets.csv', 'r')
assignments_file = open('10assignments.csv', 'r')

tweets = tweet_file.read().split('\n')
assignments = [map(int, line.split(',')) for line in assignments_file.read().split('\n')]

tweet_file.close()
assignments_file.close()

# First list gets scores split into topics
# Second list just gets all the scores for a file
sentiments = [ [[], []] for i in range(10) ]
sentiments2 = []

for assign in assignments:
    # Figure out the scores of each tweet
    scores = senti_classifier.polarity_scores([ tweets[assign[0]] ])

    # Add to the sentiment associated with the tweet's topic
    sentiments[assign[1]][0] += [scores[0]]
    sentiments[assign[1]][1] += [scores[1]]

    # Add to a list of sentiments in general
    sentiments2 += [scores]

# Write sentiments to file
sentiments_file = open('sentiments.csv', 'w')
sentiments_file.write('\n'.join(['{},{},{}'.format(i, x[0], x[1]) for (i, x) in enumerate(sentiments2)]))
sentiments_file.close()

# Plot histogram of sentiments
import matplotlib.pyplot as plt
num_bins = 20
for topic in enumerate(sentiments):
    plt.hist(topic[1][0], num_bins, facecolor='green')
    plt.xlabel('Positive Sentiment')
    plt.ylabel('Frequency')
    plt.title(r'Distribution of Positive Sentiment')
    plt.savefig(str(topic[0]) + 'pos.png')
    plt.clf()

    plt.hist(topic[1][1], num_bins, facecolor='green')
    plt.xlabel('Negative Sentiment')
    plt.ylabel('Frequency')
    plt.title(r'Distribution of Negative Sentiment')
    plt.savefig(str(topic[0]) + 'neg.png')
    plt.clf()
