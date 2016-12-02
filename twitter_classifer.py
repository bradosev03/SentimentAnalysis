
#!/usr/bin/python
## -*- coding: utf-8 -*-
"""
Author: Brandon Radosevich
Date: December 2, 2016

This module is used training a classifier for Twitter Sentiment Analysis on a live twitter feed

Attributes:

Todo:

"""
#import statements
import nltk
import random
from nltk.corpus import movie_reviews as mr
from nltk.corpus import twitter_samples
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from string import punctuation
from nltk.stem.porter import *
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
from collections import defaultdict, Counter
from nltk.stem.snowball import SnowballStemmer


class Twitter_Classifier(object):

    def __init__(self,model):
        self.classifier = self.unpackModel(model)

    def classifyTweet(self,tweet):
        """Classifies given tweet
        Args:
            tweet: (str) a tweet to classify
        Returns:
            classifcation: (str) either positive or negative
        """
        tokenized_tweet = self.word_feats(tweet)
        classification = self.classifier.classify(tokenized_tweet)
        return classification

    def word_feats(self, words):
        """Extract Features from tweet
        Args:
            words: (str) sentence to extract tokens from.
        Returns:
            dict: tokens from tweet
        """
        stopset = list(set(stopwords.words('english')))
        return dict([(word, True) for word in words.split() if word not in stopset])

    def unpackModel(self,model):
        """Unpacks a given classifer from a pickle
        Args:
            model: (str) filename of model for unpacking
        Returns:
            classifier: (Bayesain,DecisionTree,SVC,LinearSVC) for use in classifying data
        """
        print 'Unpacking Model: %s' %(model,)
        f = open(model,'rb')
        classifier = pickle.load(f)
        f.close()
        return classifier

if __name__ == '__main__':
    Tweet = Twitter_Classifier('linearsvc_model.pickle')
