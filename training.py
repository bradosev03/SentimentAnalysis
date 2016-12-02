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
import nltk
import argparse
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


class TwitterTraining(object):

    def __init__(self,filename,classifier_type):
        print '[+] Beginning Feature Extraction'
        train_feats = self.parseTweets()
        print '[+] Beginning Classification Using: %s Classifer' % (classifier_type,)
        classifier = self.classify(train_feats, classifier_type)
        self.saveClassifier(classifier, filename)

    def extract_feats(self,words):
        """Extract Features from tweet
        Args:
            words: (str) sentence to extract tokens from.
        Returns:
            dict: tokens from tweet
        """
        stopset = list(set(stopwords.words('english')))
        return dict([(word, True) for word in words if word not in stopset])

    def parseTweets(self):
        """Parses tweets and extracts features from it
        Args:
            none
        Returns:
            extract_feats: list of words with frequency of each word occuring in both positive and negative classes.
        """
        tweets =  twitter_samples.fileids()
        pos_tweets = twitter_samples.tokenized(tweets[1])
        neg_tweets = twitter_samples.tokenized(tweets[0])
        pos_training = pos_tweets[:(len(pos_tweets)*7/8)]
        neg_training = neg_tweets[:(len(pos_tweets)*7/8)]
        pos_feats = [(self.extract_feats(f), 'positive') for f in pos_training ]
        neg_feats = [(self.extract_feats(f), 'negative') for f in neg_training ]
        train_feats = pos_feats + neg_feats
        print '[-] Feature Extraction Finished'
        return train_feats

    def classify(self,features,classifier_type):
        """Read Directed Graph from File
        Args:
            features: (list) of positve and negative class with count of each word.
            classifier_type: (str)
        Returns:
            classifer:(Bayesain,DecisionTree,SVC,LinearSVC) Classifcation using one of the following methods.
        """
        if classifier_type == "Bayesian":
            classifier = nltk.NaiveBayesClassifier.train(features)
            return classifier
        elif classifier_type == "DecisionTree":
            classifier = nltk.classify.DecisionTreeClassifier.train(features, verbose=True)
            return classifier
        elif classifier_type == "SVC":
            classifier = SklearnClassifier(SVC())
            classifier.train(features)
            return classifier
        elif classifier_type == "LinearSVC":
            classifier = SklearnClassifier(LinearSVC())
            classifier.train(features)
            return classifier
        else:
            print 'Classifier %s Type Is Not Available'
        print '[-] Classification Finished'

    def saveClassifier(self, classifier, filename):
        """Save the classifier to a pickle file.
        Args:
            classifer:(Bayesain,DecisionTree,SVC,LinearSVC) Classifcation using one of the following methods.
            filename: (str) filename to save to
         Returns:
            Nothing
        """
        print 'Opening File: %s ' %(filename,)
        new_model = open(filename,"wb")
        pickle.dump(classifier, new_model)
        new_model.close()
        print 'Closing File: %s ' %(filename,)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Twitter Sentiment Training")
    parser.add_argument("-f","--file", dest="filename", help="Filename to save Classifier to", metavar="FILE",required=True)
    parser.add_argument("-t","--type", dest="classifier_type", help="Type of Classifier To Use", metavar="FILE",required=True)
    args = parser.parse_args()
    training = TwitterTraining(args.filename, args.classifier_type)
