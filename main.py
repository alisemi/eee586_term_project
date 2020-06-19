import sqlite3
from sqlite3 import Error
from tqdm import tqdm
import igraph as ig
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import ngrams
import pickle
import language_check
import math
import statistics
import fileinput
import os
import re
import numpy as np
import string
from profanity_check import predict, predict_prob
from shutil import copyfile
from sklearn.cluster import KMeans
from nltk import FreqDist
import collections
from sklearn import metrics
import matplotlib.pyplot as plt
import utils
import feature_engineering
import graph_engineering

dataset_filename = "../reddit-comments-may-2015/CasualConversations_sub.db"
dataset_filename_pkl = "../reddit-comments-may-2015/CasualConversations_sub.pkl"
acronyms_filename = "./list_acronym.txt"
recursive_weigh_factor = 1/2
base_weight = 1
total_comments = 234694

# Features except ngrams
feature_file_list = ["features/feature_acronym.pkl","features/feature_emoji.pkl","features/feature_grammar_check.pkl","features/feature_profanity.pkl","features/feature_punct.pkl","features/feature_sentence_length.pkl","features/feature_uppercase.pkl","features/feature_zipf.pkl"]
feature_graph_list = ["networks/feature_acronym_graph.txt","networks/feature_emoji_graph.txt","networks/feature_grammar_check_graph.txt","networks/feature_profanity_graph.txt","networks/feature_punct_graph.txt","networks/feature_sentence_length_graph.txt","networks/feature_uppercase_graph.txt","networks/feature_zipf_graph.txt"]

#filenames of features used in k-means clustering
feature_filenames_cluster = ["features/feature_acronym.pkl", "features/feature_emoji.pkl", 
                        "features/feature_profanity.pkl", "features/feature_punct.pkl", \
                         "features/feature_zipf.pkl" ]
num_authors_effective = 6334


# reads feature files and generates clusters based on k-means clustering
def feature_to_cluster(feature_filenames_cluster, num_clusters):
    points = np.ndarray((num_authors_effective, len(feature_filenames_cluster))) # rows for authors, columns for features
    print(points.shape)
    for j in range(len(feature_filenames_cluster)):
        feature_data = utils.load_feature(feature_filenames_cluster[j])
        feature_list = list(feature_data.items())
        authors = list(feature_data.keys())
        feature_expon = []
        for i, val in enumerate(feature_list):
            feature_expon.append((val[0], 10**(val[1]))) # features are converted to exponential scale for better resolution in distance
        amin, amax = min(feature_expon,key=lambda tup: tup[1] ), max(feature_expon, key=lambda tup: tup[1])
        feature_normed = []
        for i, val in enumerate(feature_expon):
            if amax[1] == amin[1]: # prevents division by zero
                feature_normed.append((val[0], val[1]))
            else:
                feature_normed.append((val[0], (val[1]-amin[1]) / (amax[1]-amin[1])))
        for i in tqdm(range(len(authors))):
            points.itemset((i, j), feature_normed[i][1])
    points[np.isfinite(points) == False] = 1 # for safety
    points[np.isnan(points) == True] = 0 # for safety
    kmeans = KMeans(n_clusters = num_clusters)
    kmeans = kmeans.fit(points)
    labels = kmeans.predict(points)
    clusters = {}
    for i in tqdm(range(len(authors))):
       clusters[authors[i]] = (labels[i],)
    kmeans_file = open('kmeans_clusters.pkl', 'wb')
    pickle.dump(clusters, kmeans_file)
    kmeans_file.close()
    return clusters


if __name__ == '__main__':
    print("This file contains a collection of functions intended to be used in Spyder environment for the project of the course EEE 586: Statistical Foundations of Natural Language Processing in Bilkent University, Spring 2020.")