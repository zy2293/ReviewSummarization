#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 12:34:28 2018

@author: zihanye

Goal: generate <feature, opinion, score> for each business_id
"""


import numpy as np
import pandas as pd
import csv
import os
import nltk
import lang
import itertools

from ast import literal_eval
from nltk.corpus import sentiwordnet as swn
from inflection import singularize



# Global Variables
NNLOGIC = ['NN', 'NNS' ,'NNP','NNPS']
JJLOGIC = ['JJ' ,'JJR' ,'JJS']


def extractOpinion(review):
    allpairs = set()

    for tagged in review:

        sentLength = len(tagged)  # sentence length

        n = 3  # search in a context window six words
        if sentLength > (n - 1):
            word_tag_pairs = nltk.ngrams(tagged, n)

            for word_tag in word_tag_pairs:
                nn = [word for (word, tag) in word_tag if tag in NNLOGIC]
                jj = [word for (word, tag) in word_tag if tag in JJLOGIC]

                [allpairs.update(list(itertools.product(nn, jj)))]
        else:

            nn = [word for (word, tag) in tagged if tag in NNLOGIC]
            jj = [word for (word, tag) in tagged if tag in JJLOGIC]

            [allpairs.update(list(itertools.product(nn, jj)))]
            # allpairs.update(pairs)

    return allpairs



def generateList(df):
    opinionList = {}
    for (name, group) in df.groupby("business_id"):
        nn_jj = set()
        for review in group["tagReview"]:
            nn_jj.update(extractOpinion(review))
        opinionList[name] = nn_jj
    return opinionList


def generatePair(opinionList, featureList):
    pair = {}
    for (key, group) in featureList.groupby("business_id"):
        opinion = opinionList[key]
        pair[key] = []
        feature = list(group["featureList"])[0]
        for f in feature:
            attribute = [jj for (nn, jj) in opinion if singularize(nn)== f]
            if attribute:
                result = [(f, attr) for attr in attribute]
                pair[key].extend(result)
    return pair


def searchPair(feature, opinion, review):
    for re in review:
        if feature in re and opinion in re:
            return True
    return False


def scoreSentiment(word):
    ss = None
    score = 0
    word = word.lower()
    if list(swn.senti_synsets(word)):
        ## all input opinion words are adjectives
        adj = list(swn.senti_synsets(word, 'a'))
        s = list(swn.senti_synsets(word, 's'))

        if adj:
            ss = adj[0]
            score = ss.pos_score() - ss.neg_score()
        elif s:
            ss = s[0]
            score = ss.pos_score() - ss.neg_score()
    else:
        score = 0

    return score

def generateSentiScore(pairs, df):
    pairScore = {}

    for (key, group) in df.groupby("business_id"):

        pairScore[key] = []
        p = pairs[key]  # extract all <feature, pair> pairs

        for (feature, opinion) in p:
            sentiScore = scoreSentiment(opinion)
            logic = group["rating"][group["refineReview"].apply(lambda re: searchPair(feature, opinion, re))]

            score = np.mean(logic) * sentiScore

            if score != 0 and ~np.isnan(score):
                out = (feature, opinion, score)
                pairScore[key].append(out)

    return pairScore


def execute(df, featureList):
    featureList['featureList'] = featureList['featureList'].apply(literal_eval)
    opinionList = generateList(df)
    pairs = generatePair(opinionList, featureList)
    sentiScore = generateSentiScore(pairs, df)

    return sentiScore



def main():

    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    rel_path = "data/sbj_reviews.csv"
    path = os.path.join(script_dir, rel_path)
    language = "en"
    df = lang.preprocess(path, language)
    df["tagReview"] = df["review"].apply(lambda re: lang.addTagReview(re))
    df["refineReview"] = df["review"].apply(lambda re: lang.addRefineReview(re))

    # import feature list
    featureInput = os.path.join(script_dir, "output/sbj_features.csv")
    featureList = pd.read_csv(featureInput)

    sentiScore= execute(df, featureList)

    fields = ["business_id", "feature", "opinion", "score"]
    output = os.path.join(script_dir, "output/outputPairs.csv")
    with open(output, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fields)
        writer.writeheader()
        for (res, pairs) in sentiScore.items():
            for item in pairs:
                writer.writerow({'business_id': res, 'feature': item[0], 'opinion': item[1], 'score': item[2]})


if __name__ == "__main__":
    main()