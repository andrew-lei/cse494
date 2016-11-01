#! /usr/bin/env python

import nltk
import stop_words
import gensim
import re
#data cleaning: Tokenizing, Stopping, Stemming
#Tokenizing
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

tweet_file = open('tweets.csv', 'r')

# Twitter url pretty common; filter out
# Technically this is stopping, but got to get this done before tokenizing;
# Tokenizing will split a url into separate list values, making it much more
# Difficult to stop
# Last line is blank, so use [:-1]
tweet_list = re.sub(r'https t co ..........', ' ', tweet_file.read()).split('\n')[:-1]
tweet_file.close()
tweets = []

for t in tweet_list:
    # Make lowercase and tokenize
    t_lower = t.lower()
    tokens = tokenizer.tokenize(t_lower)
    tweets.append(tokens)


#create English stop words list
from stop_words import get_stop_words
en_stop = get_stop_words('en')
#remove stop words from tokens
tokens = []
tweet_remove = ['rt','https','co']
for t in tweets:
    token = [i for i in t if not i in en_stop]
    token = [i for i in token if not i in tweet_remove]
    #you may also want to remove 'rt' etc.
    tokens.append(token)

# print tokens

#stemming
from nltk.stem.porter import PorterStemmer
p_stemmer = PorterStemmer()
texts = []
for t in tokens:
    text = [p_stemmer.stem(i) for i in t]
    text = [i for i in text if not len(i) <= 1]
    texts.append(text)

##Constructing a document-term matrix
from gensim import corpora, models
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
#doc2bow converts text into bag-of-words. Corpus is a list of vectors.
#use genism's LdaModel function
ntops = 10

# Create an LdaModel with ntops topics
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = ntops, id2word = dictionary, passes = 20)

# Write the topics to a file
# Topics are in a list of form [(index, topic)]
topics_file = open(str(ntops) + 'topics.csv', 'w')
topics_file.write('\n'.join( ['{},{}'.format(*i) for i in ldamodel.print_topics(num_topics = ntops, num_words = 4)] ))
topics_file.close()

from operator import itemgetter

# Find topic distribution of each tweet; choose the maximum
# Then write to a file
assigned_topics = [max(x, key = itemgetter(1))[0] for x in ldamodel.get_document_topics(corpus)]
assignments_file = open(str(ntops) + 'assignments.csv', 'w')
assignments_file.write('\n'.join(['{},{}'.format(i,x) for (i,x) in enumerate(assigned_topics)]))
assignments_file.close()

from collections import Counter
# Number in each topic
print Counter(assigned_topics)
