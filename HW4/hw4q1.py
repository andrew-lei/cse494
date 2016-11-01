#! /usr/bin/env python

import tweepy
import re

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Collect 5000 tweets
tweet_num = 5000
tweet_file = open('tweets2.csv', 'a')

# Create a stream
class MyStreamListener(tweepy.StreamListener):
    def __init__(self, api=None):
        super(MyStreamListener, self).__init__()
        self.num_tweets = 0

    def on_error(self, status_code):
        if status_code == 420:
            return False

    def on_status(self, status):
        print self.num_tweets, status.text

        # Replace non-word characters with spaces
        words_only = re.sub(r"[^\w\s]", ' ', status.text)

        # Any amount and any kind of whitespace changed to one single space
        # This is necessary to save in a file delimited with endlines
        # Because some tweets have multiple lines
        only_spaces = re.sub(r"\s+", ' ', words_only)
        tweet_file.write(only_spaces + '\n')
        self.num_tweets += 1

        # Close when goal is met
        if self.num_tweets >= tweet_num:
            tweet_file.close()
            return False
        else:
            return True


myStreamListener = MyStreamListener()
myStream = tweepy.Stream(api.auth, myStreamListener)
myStream.filter(track=['Harambe'], async = True)
