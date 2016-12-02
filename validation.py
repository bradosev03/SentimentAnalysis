#!/usr/bin/python
## -*- coding: utf-8 -*-
"""
Author: Brandon Radosevich
Date: December 2, 2016

This module is used testing a given classifier for Twitter Sentiment Analysis.

Attributes:

Todo:

"""
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
import argparse

class Validate_Model(object):

    def __init__(self, model):
        classifier = self.unpackModel(model)
        self.validate(classifier)

    def train_feats(self,words):
        """Extract Features from tweet
        Args:
            words: (str) sentence to extract tokens from.
        Returns:
            dict: tokens from tweet
        """
        stopset = list(set(stopwords.words('english')))
        return dict([(word, True) for word in words if word not in stopset])

    def validate(self,classifier):
        """Test the accuracy of a given classifier against a test dataset with labels.
        Args:
            classifier: (Bayesain,DecisionTree,SVC,LinearSVC) for use in classifying data
        Returns:
            None
        """
        tweets =  twitter_samples.fileids()
        pos_tweets = twitter_samples.tokenized(tweets[1])
        neg_tweets = twitter_samples.tokenized(tweets[0])
        pos_testing = pos_tweets[(len(pos_tweets)*7/8):]
        neg_testing = neg_tweets[(len(neg_tweets)*7/8):]
        pos_test  = [(self.train_feats(f), 'positive') for f in pos_testing ]
        neg_test = [(self.train_feats(f), 'negative') for f in neg_testing ]
        testfeats = pos_test + neg_test
        print("Classifier accuracy percent:",(nltk.classify.accuracy(classifier, testfeats))*100)

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
    parser = argparse.ArgumentParser("Twitter Sentiment Testing")
    parser.add_argument("-f","--file", dest="model", help="Filename of Classifier ", metavar="FILE",required=True)
    args = parser.parse_args()
    validation = Validate_Model(args.model)
