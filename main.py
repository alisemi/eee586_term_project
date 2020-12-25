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


# Features except ngrams
#feature_file_list = ["features/feature_acronym.pkl","features/feature_emoji.pkl","features/feature_grammar_check.pkl","features/feature_profanity.pkl","features/feature_punct.pkl","features/feature_sentence_length.pkl","features/feature_uppercase.pkl","features/feature_zipf.pkl"]
feature_graph_list = ["networks/TipOfMyTongue_sub/feature_ngrams_unigram_overlap_graph.txt",
                      "networks/TipOfMyTongue_sub/feature_ngrams_bigram_overlap_graph.txt",
                      "networks/TipOfMyTongue_sub/feature_ngrams_trigram_overlap_graph.txt",
                      "networks/TipOfMyTongue_sub_feature_acronym_graph.txt",
                      "networks/TipOfMyTongue_sub_feature_emoji_graph.txt",
                      "networks/TipOfMyTongue_sub_feature_grammar_check_graph.txt",
                      "networks/TipOfMyTongue_sub_feature_profanity_graph.txt",
                      "networks/TipOfMyTongue_sub_feature_punct_graph.txt",
                      "networks/TipOfMyTongue_sub_feature_sentence_length_graph.txt",
                      "networks/TipOfMyTongue_sub_feature_uppercase_graph.txt",
                      "networks/TipOfMyTongue_sub_feature_zipf_graph.txt",
                      "networks/ngram_merged_graph.txt",
                      "networks/style_merged_graph.txt"]

#filenames of features used in k-means clustering
'''
feature_filenames_cluster = ["features/feature_acronym.pkl", "features/feature_emoji.pkl", 
                        "features/feature_profanity.pkl", "features/feature_punct.pkl", \
                         "features/feature_zipf.pkl" ]
'''

# reads feature files and generates clusters based on k-means clustering
def feature_to_cluster(feature_filenames_cluster, num_clusters):    
    num_authors_effective = len(utils.load_feature(feature_filenames_cluster[0]))
    points = np.ndarray((num_authors_effective, len(feature_filenames_cluster))) # rows for authors, columns for features
    for j in range(len(feature_filenames_cluster)):
        feature_data = utils.load_feature(feature_filenames_cluster[j])
        feature_list = list(feature_data.items())
        authors = list(feature_data.keys())
        feature_expon = []
        for i, val in enumerate(feature_list):
            #feature_expon.append((val[0], 10**(val[1]))) # features are converted to exponential scale for better resolution in distance
            feature_expon.append((val[0], val[1]))
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
    
    clusters = {}
    kmeans = KMeans(n_clusters = num_clusters[0])
    kmeans = kmeans.fit(points)
    labels = kmeans.predict(points)     
    for i in tqdm(range(len(authors))):
       clusters[authors[i]] = (labels[i],)
    
    for clusterSize in num_clusters[1:]:     
        kmeans = KMeans(n_clusters = clusterSize)
        kmeans = kmeans.fit(points)
        labels = kmeans.predict(points)     
        for i in tqdm(range(len(authors))):
           clusters[authors[i]] += (labels[i],)
           
    kmeans_file = open('kmeans_clusters.pkl', 'wb')
    pickle.dump(clusters, kmeans_file)
    kmeans_file.close()
    return clusters


def new_experiment(dataset_filename, network_filename):
    dataset_filename = '../reddit-comments-may-2015/TipOfMyTongue_sub.db'
    network_filename = 'TipOfMyTongue_sub_network_Dec_2020.txt'
    graph_engineering.db_to_graph(dataset_filename, network_filename, parenting=False)
    
    print('Community detection...')
    topological_community,used_authors,numClusters = graph_engineering.community_detection(network_filename)
    print('Used authors : ' + str(used_authors))

    # Feature Extraction
    print('Feature Extraction....')
    feature_file_list = feature_engineering.extract_features(dataset_filename,used_authors)

    # Cluster communities
    # TODO different community detection algorithms
    # TODO number of clusters based on how many communities
    print('Cluster communities...')
    clusters = []
    clusters.append(feature_to_cluster(feature_file_list[1:8], numClusters))
    
    # Evaluation
    print('Evaluations.....')
    cluster_names = ['K-means']
    for i, cluster in enumerate(clusters):
        evaluations = utils.evaluate_cluster_to_community(topological_community, cluster, 5)
        utils.plot_results(evaluations,cluster_names[i], "results/" + os.path.basename(dataset_filename)[:-3] + "_result_" + cluster_names[i].replace(" ", "_") + ".png")


if __name__ == '__main__':
    print("This file contains a collection of functions intended to be used in Spyder environment for the project of the course EEE 586: Statistical Foundations of Natural Language Processing in Bilkent University, Spring 2020.")
    utils.table_name = "TipOfMyTongue_sub"
    #graph_engineering.db_to_graph("../reddit-comments-may-2015/TipOfMyTongue_sub.db","TipOfMyTongue_sub_network_Dec_2020.txt",False)
    new_experiment('../reddit-comments-may-2015/TipOfMyTongue_sub.db', 'TipOfMyTongue_sub_network_Dec_2020.txt')
    
    