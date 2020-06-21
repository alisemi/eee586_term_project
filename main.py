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
    print(points.shape)
    for j in range(len(feature_filenames_cluster)):
        print(feature_filenames_cluster[j])
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


def das_experiment(dataset_filename, network_filename):
    # Non-linguistic Graph Construction
    print('Non-linguistic Graph Construction...')
    graph_engineering.db_to_graph(dataset_filename, network_filename)   
    
    # Feature Extraction
    print('Feature Extraction....')
    feature_file_list = feature_engineering.extract_features(dataset_filename)
    
    # Linguistic Graph Construction
    print('Linguistic Graph Construction...')
    graph_files = []
    graph_files.extend(graph_engineering.overlapping_ngrams(feature_file_list[0]))
    graph_files.extend(graph_engineering.create_graphs_from_features(feature_file_list[1:]))
    
    # Merge Graphs
    print('Merging the desired graphs..(You should write down the code for that)')  
    graph_files.append(graph_engineering.merge_graphs(graph_files[0:3],[0.33,0.33,0.33], 'networks/TipOfMyTongue_sub_ngram_merged_graph.txt' ))
    graph_files.append(graph_engineering.merge_graphs(graph_files[4:11],[1,1,1,1,1,1,1],'networks/TipOfMyTongue_sub_style_merged_graph.txt')) 
    
    # Communities
    print('Communitiy Detection....')
    ground_truth_community = graph_engineering.community_detection(network_filename)
    communities = []
    
    for graph_file in graph_files:
        current_communitiy = graph_engineering.community_detection(graph_file)
        communities.append(current_communitiy)
        
    # Cluster communities
    clusters = []
    #clusters.append(feature_to_cluster(feature_file_list[1:8], 15, utils.num_authors_effective))
    
    # Evaluation
    print('Evaluations.....')
    community_names = ['Unigram Feature Only', 'Bigram Feature Only', 'Trigram Feature Only',
                       'Acronym Feature Only', 'Zipf\'s Law Feature Only',
                       'Uppercase Feature Only', 'Emoji Feature Only',
                       'Punctuation Feature Only','Profanity Feature Only',
                       'Grammar Feature Only', 'Sentence Length Feature Only',
                       'Ngram Features', 'Style Features']
    for i,community in enumerate(communities):
        evaluations = utils.evaluate(ground_truth_community, community, 4)
        utils.plot_results(evaluations,community_names[i], "results/result_" + community_names[i].replace(" ", "_") + ".png")
    
    cluster_names = ['K-means']
    for i, cluster in enumerate(clusters):
        evaluations = utils.evaluate_cluster_to_community(ground_truth_community, cluster, 4)
        utils.plot_results(evaluations,cluster_names[i], "results/result_" + cluster_names[i].replace(" ", "_") + ".png")



if __name__ == '__main__':
    print("This file contains a collection of functions intended to be used in Spyder environment for the project of the course EEE 586: Statistical Foundations of Natural Language Processing in Bilkent University, Spring 2020.")
    utils.table_name = "TipOfMyTongue_sub"
    das_experiment('../reddit-comments-may-2015/TipOfMyTongue_sub.db', 'TipOfMyTongue.txt')