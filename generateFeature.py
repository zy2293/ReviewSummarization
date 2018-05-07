#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 12:34:28 2018

@author: zihanye

Goal: generate frequent feature list for each restaurant/business_id
"""


import lang
import numpy as np
from nltk.corpus import stopwords
import string
import nltk
import csv
import os
from inflection import singularize
from collections import Counter
import itertools


NNLOGIC = ['NN', 'NNS' ,'NNP','NNPS']

# input: restaurant reviews
# output: dict: key (Transaction ID), value (NN (including duplicates))
def initTID(reviews):
    featureList = {}
    TID = 0
    for review in reviews:
        fe = []
        for tagged in review:
            fe += [word for (word, tag) in tagged if (tag in NNLOGIC)]

        featureList[TID] = fe
        TID += 1
    return featureList, TID


def initItemSet(featureList):
    items = []
    [items.extend(values) for (key, values) in featureList.items()]
    itemset = Counter(items)
    return itemset


def minMaxScale(itemset):
    count = []
    [count.append(itemset[c]) for c in itemset]
    scaleRange = max(count) - min(count)
    for (key, value) in itemset.items():
        itemset[key] = value/scaleRange
    return itemset


# itemset: itemset: Counter
# support: user-defined support, item frequency, int
# return: candidateset: Counter
def prune(itemset, support):
    if itemset == Counter():
        return Counter()

    candidateset = Counter()
    for (key, value) in itemset.items():
        if value > support:
            candidateset[key] = value
    return candidateset




# initialized frequent items list
def apriori(L, featureList, support, TIDnum):
    k = 2  # combinations pair
    # support = 0.1
    L_k = Counter()
    prev = []
    post = []
    [prev.append(c) for c in L]
    prevLength = len(list(set(prev)))
    postLength = len(post)

    while L != Counter() and (postLength != prevLength):
        pair = Counter()
        prev = []
        if k < 3:
            [prev.append(c) for c in L]
            L = list(set(prev))
        else:
            [prev.extend([*c]) for c in L]
            L = list(set(prev))

        prevLength = len(L)

        candidate = itertools.combinations(L, r=k)  # dtype: itertools.combinations
        for (key, values) in featureList.items():
            featureCount = Counter(values)
            for c in candidate:  # (k-tuple)
                freq = np.prod([featureCount[c[i]] for i in range(k)])
                freq = freq / TIDnum

                if freq > 0:
                    pair += Counter({c: freq})

        L = prune(pair, support)
        post = []
        [post.extend([*c]) for c in L]
        postLength = len(list(set(post)))
        L_k += L
        k += 1  # increment k pairs combination

    final = []
    [final.extend([*c]) for c in L_k]
    frequentFeature = list(set(final))
    return frequentFeature


def generateFeatures(df):
    frequentFeature = {}
    for (name, group) in df.groupby("business_id"):
        featureList, TIDnum = initTID(group["tagReview"])
        itemset = initItemSet(featureList)
        support = 0.001
        L = prune(itemset, support)
        features = apriori(L, featureList, support, TIDnum)
        frequentFeature[name] = features
    return frequentFeature


def main():

    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    rel_path = "data/sbj_reviews.csv"
    path = os.path.join(script_dir, rel_path)
    language = "en"
    df = lang.preprocess(path, language)
    df["tagReview"] = df["review"].apply(lambda re: lang.addTagReview(re))
    df["refineReview"] = df["review"].apply(lambda re: lang.addRefineReview(re))

    frequentFeature = generateFeatures(df)


    fields = ["business_id", "featureList"]
    output = os.path.join(script_dir, "output/sbj_features_1.csv")
    with open(output, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fields)
        writer.writeheader()
        for (name, pair) in frequentFeature.items():
            writer.writerow({'business_id': name, 'featureList': pair})

if __name__ == "__main__":
    main()