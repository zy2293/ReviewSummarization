#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 10:15:37 2018

@author: zihanye

Goal: preprocessing methods
"""


import pandas as pd
from langdetect import detect
import nltk
from inflection import singularize
import spacy

# Global Variables
NNLOGIC = ['NN', 'NNS' ,'NNP','NNPS']
JJLOGIC = ['JJ' ,'JJR' ,'JJS']
nlp = spacy.load('en_core_web_sm')


def preprocess(path, language):
    df = pd.read_csv(path)
    df = df.rename(columns={'stars_x': 'stars', 'stars_y': 'rating', 'text': 'review'})
    df["lan"] = df["review"].apply(lambda x: detect(x))
    df1 = df[df["lan"] == language]
    return df1




def addTagReview(review):

    re = nltk.sent_tokenize(review)
    tagReview = []  # (token, tag) after removing stopwords, punctuations

    for sentence in re:
        tagSent = []

        sent = nlp(sentence)

        tagged = [((token.text).lower(), token.tag_) for token in sent if token.is_alpha and ~token.is_stop]

        for (word, tag) in tagged:
            if tag in NNLOGIC:
                word = singularize(word)

            tagSent.append((word, tag))
        tagReview.append(tagSent)

    return tagReview


def addRefineReview(review):

    re = nltk.sent_tokenize(review)
    refineReview = []  # refined Reviews without tagging after

    for sentence in re:

        refineSent = []

        sent = nlp(sentence)

        tagged = [((token.text).lower(), token.tag_) for token in sent if token.is_alpha and ~token.is_stop]

        for (word, tag) in tagged:
            if tag in NNLOGIC:
                word = singularize(word)
            refineSent.append(word)

        refineReview.append(refineSent)

    return refineReview


def addRawReview(review):

    re = nltk.sent_tokenize(review)
    rawReview = []  # tagged word without removing stopwords, punctuations

    for sentence in re:

        rawSent = []

        sent = nlp(sentence)
        raw = [((w.text).lower(), w.tag_) for w in sent]

        for (w, tag) in raw:
            if tag in NNLOGIC:
                w = singularize(w)

            rawSent.append((w, tag))

        rawReview.append(rawSent)

    return rawReview


