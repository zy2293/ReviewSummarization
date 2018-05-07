#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Fri 20 7:58:31 2018

@author: zihanye

Goal: use dependency parsing to generate <feature, opinion, score> for each business_id
"""

import numpy as np
import spacy
import lang
import nltk
import csv
from inflection import singularize
import os
from nltk.corpus import sentiwordnet as swn

nlp = spacy.load('en_core_web_sm')

# Global Variables
NNLOGIC = ['NN', 'NNS' ,'NNP','NNPS']
JJLOGIC = ['JJ' ,'JJR' ,'JJS']


def dependentParse(review):
    re = nltk.sent_tokenize(review)
    pairs = set()

    for sentence in re:

        sent = nlp(sentence)
        for token in sent:
            if token.dep_ == "amod" and token.head.pos_ == "NOUN":
                opinion = (token.text).lower()
                feature = singularize((token.head.text).lower())

                pairs.add((feature, opinion))

    return pairs


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


def generatePairs(df):
    pairScore = {}
    for (key, group) in df.groupby("business_id"):
        pairsList = set()
        pairScore[key] = []

        for review in group["review"]:
            resultPairs = dependentParse(review)
            for item in resultPairs:
                pairsList.add(item)

        # pairsList = set(pairsList)

        for (feature, opinion) in pairsList:

            sentiScore = scoreSentiment(opinion)
            logic = group["rating"][group["refineReview"].apply(lambda re: searchPair(feature, opinion, re))]

            score = np.mean(logic) * sentiScore

            if score != 0 and ~np.isnan(score):
                out = (feature, opinion, score)
                pairScore[key].append(out)

    return pairScore

def main():

    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    rel_path = "data/sbj_reviews.csv"
    path = os.path.join(script_dir, rel_path)
    language = "en"
    df = lang.preprocess(path, language)
    df["refineReview"] = df["review"].apply(lambda re: lang.addRefineReview(re))

    sentiPair = generatePairs(df) # generate pairs

    # output to csv files
    fields = ["business_id", "feature", "opinion", "score"]

    output = os.path.join(script_dir, "output/dependencyParsePairs.csv")
    with open(output, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fields)
        writer.writeheader()
        for (res, pairs) in sentiPair.items():
            for item in pairs:
                writer.writerow({'business_id': res, 'feature': item[0], 'opinion': item[1], 'score': item[2]})


if __name__ == "__main__":
    main()



