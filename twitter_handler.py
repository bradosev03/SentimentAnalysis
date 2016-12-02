#!/usr/bin/python
## -*- coding: utf-8 -*-
"""
Author: Brandon Radosevich
Date: December 2, 2016

This module is used training a classifier for Twitter Sentiment Analysis.

Attributes:

Todo:
    * TODOS
"""
from tweepy import Stream
import argparse
from collections import deque
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
from twitter_classifer import Twitter_Classifier
import time
import termcolor
from termcolor import colored, cprint

class twitter_handler(object):

    def __init__(self,topics,model):
        auth = self.initOAuth()
        self.streamTwitter(auth,topics,model)

    def initOAuth(self):
        """initializes OAuthentication Process with the given Keys.
        Args:
            none
        Returns:
            auth: (OAuth) OAuth for use in twitter streaming session.
        """
        ckey= "JyEkixVSXO6Vs8URH0VpLNrVH"
        csecret= "aBhcPXJbxiUs8HpXfdADwdqvtVfOapHw2fuYl7Na4GGtFmU584"
        atoken= "792108839814299648-55D1XYMRvpWOH9gHnrvcj3YpdUJwoMX"
        asecret= "S8Ha0btFjOmOY4NWUhkwf8ZQMpFx2zwkVJsqhUXxMvC5j"
        auth = OAuthHandler(ckey, csecret)
        auth.set_access_token(atoken, asecret)
        return auth

    def streamTwitter(self,auth,topics,model):
        """Streams twitter feed with a filter for the given topics.
        Args:
            auth: (OAuth) OAuth session key for authentication
            topics: (list) A list of topics to filter the tweet with.
        Returns:
            None
        """
        classifier = Twitter_Classifier(model)
        count = 0
        pos_count = 0
        neg_count = 0
        mainListener = listener(classifier=classifier,pos_count=pos_count,neg_count=neg_count)
        twitterStream = Stream(auth, mainListener)
        tweet = twitterStream.filter(track=topics)

class listener(StreamListener):

    def __init__(self,classifier,pos_count,neg_count):
        self.classifier = classifier
        self.pos_count = pos_count
        self.neg_count = neg_count

    def on_data(self, data):
        all_data = json.loads(data)
        tweet = all_data["text"]
        username = all_data["user"]["screen_name"]
        classification = self.classifier.classifyTweet(tweet)
        if classification == "positive":
            self.pos_count = self.pos_count + 1
        if classification == "negative":
            self.neg_count = self.neg_count + 1
        print '###############'
        print(time.strftime('%a %H:%M:%S')),
        print ' | ' ,classification, ' | ', tweet
        print '###############'

    def on_timeout(self):
        print('Timeout...')
        return True # To continue listening

    def on_error(self, status):
        print status

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Twitter Sentiment Training")
    parser.add_argument("-f","--file", dest="model", help="Filename of Classifier ", metavar="FILE",required=True)
    args = parser.parse_args()
    #Change Topics in Below For Twitter Stream
    topics = ["Machine Learning","Data Analysis", "Python"]
    th = twitter_handler(topics,args.model)
