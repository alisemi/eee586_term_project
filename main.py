#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
from shutil import copyfile

dataset_filename = "../reddit-comments-may-2015/CasualConversations_sub.db"
recursive_weigh_factor = 1/2
base_weight = 1
total_comments = 234694


# merge graph files
def merge_graphs(file_list, scalar_list, output_filename):
    if os.path.exists(output_filename):
        os.remove(output_filename) 
    for i in range(len(file_list)):
        copyfile(file_list[i], "temp_graph.txt")
        for line in fileinput.input("temp_graph.txt", inplace=1):
            line_components = line.split()
            new_weight = float(line_components[2])*scalar_list[i]
            if new_weight > 0.0001:
                print(line_components[0] + " " + line_components[1] + " " + str(new_weight))
        os.system("cat temp_graph.txt >> " + output_filename)
        g = ig.Graph.Read_Ncol(output_filename,directed=False)
        g.simplify(combine_edges='sum')
        g.write_ncol(output_filename)
    os.remove("temp_graph.txt")
    normalize_graph(output_filename)


def normalize_graph(graph_filename):
    graph_file = open(graph_filename)
    line = graph_file.readline() 
    max_val = float(line.split()[2])
    min_val = 0.0
    while True:     
        # Get next line from file 
        line = graph_file.readline() 
        # if line is empty 
        # end of file is reached 
        if not line: 
            break
        
        weight = float(line.split()[2])
        '''
        if weight < min_val:
            min_val = weight
        '''
        if weight > max_val:
            max_val = weight
            
    graph_file.close()
        
    for line in fileinput.input(graph_filename, inplace=1):
        line_components = line.split()
        new_weight = (float(line_components[2])-min_val)/(max_val-min_val)
        print(line_components[0] + " " + line_components[1] + " " + str(new_weight))   
    
    
def feature_to_graph(feature_file):
    graph_file = open(feature_file[0:-4] + "_graph.txt", "w")
    feature = load_feature(feature_file)
    feature_list = list(feature.items())
    feature_list.sort(key=lambda tup: tup[1])
    
    # Setting the log scale
    feature_logged = []
    # Get the smallest value after 0
    for val in feature_list:
        if val[1] > 0:
            smallest = val
            break
    for i, val in enumerate(feature_list):
        if val[1] > 0:
            feature_logged.append((val[0], math.log10(val[1])))
        else:
            feature_logged.append((val[0], math.log10(smallest[1])))
    
    # Normalizing the feature list
    amin, amax = min(feature_logged,key=lambda tup: tup[1] ), max(feature_logged, key=lambda tup: tup[1])
    feature_normed = []
    for i, val in enumerate(feature_logged):
        feature_normed.append((val[0], (val[1]-amin[1]) / (amax[1]-amin[1])))
        
    std_deviation = statistics.stdev(list(map(lambda x: x[1], feature_normed)))
    
    for i in tqdm(range(len(feature_normed))):
        edges = []
        for j in range(i+1,len(feature_normed)):
            distance = feature_normed[j][1] - feature_normed[i][1]
            if distance < std_deviation: # Similartiy should be at least 1-standart_deviation
                similarity = 1-distance
                edges.append((feature_normed[j][0],similarity))
            else: # Otherwise it is considered as zero, as list is sorted, no need to look at the rest
                break
        # Write the edges to file
        for edge in edges:    
            print(feature_normed[i][0] + " " + edge[0] + " " + str(edge[1]), file=graph_file) 
    

# Gives the grammar_mistake/sentence for each author
def feature_grammar_check(dataset_filename):
    tool = language_check.LanguageTool('en-US')
    conn = connect_db_in_memory(dataset_filename)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT author FROM data")
    distinct_authors = cur.fetchall() # Fetchall returns a tuple ("author_name",)
    target = ('[deleted]',)
    distinct_authors.remove(target)
    comments_sum = 0
    author_grammars = {}
    for author in tqdm(distinct_authors):
        cur.execute("SELECT body FROM data WHERE author=?", author)
        comments = cur.fetchall() #rows holds ids of all comments made by the author
        if len(comments) > 3:
            comments = list(map(lambda x: x[0], comments)) 
            single_text = ''.join(comments)
            sentences = nltk.sent_tokenize(single_text)
            if len(sentences) > 0: # Just a precaution
                matches = tool.check(single_text)
                author_grammars[author[0]] = len(matches)/len(sentences)
        
        comments_sum += len(comments)
        print("\n" + str((comments_sum/total_comments)*100) + "% of total main comments are processed.")
    
    ngrams_file = open('feature_grammar_check.pkl', 'wb')
    pickle.dump(author_grammars, ngrams_file)                      
    ngrams_file.close()

class NgramSets:
    pass

def overlapping_ngrams():
    author_ngrams = load_feature('feature_ngrams.pkl')
    unigram_file = open("unigram_overlap_network.txt", "w")
    bigram_file = open("bigram_overlap_network.txt", "w")
    trigram_file = open("trigram_overlap_network.txt", "w")
    authors = list(author_ngrams.keys())
    for i in tqdm(range(len(authors))):
        source_ngrams = author_ngrams[authors[i]]
        for j in range(i+1,len(authors)):
            target_ngrams = author_ngrams[authors[j]]
            # Unigrams
            if len(source_ngrams.unigrams) > 0:
                unigram_ratio = len(source_ngrams.unigrams.intersection(target_ngrams.unigrams))/len(source_ngrams.unigrams)
                if unigram_ratio >= 0.15: # Threshold
                    print(authors[i] + " " + authors[j] + " " + str(unigram_ratio), file=unigram_file)
            # Bigrams
            if len(source_ngrams.bigrams) > 0:
                bigram_ratio = len(source_ngrams.bigrams.intersection(target_ngrams.bigrams))/len(source_ngrams.bigrams)
                if bigram_ratio >= 0.05:
                    print(authors[i] + " " + authors[j] + " " + str(bigram_ratio), file=bigram_file)
            # Trigrams
            if len(source_ngrams.trigrams) > 0:
                trigram_ratio = len(source_ngrams.trigrams.intersection(target_ngrams.trigrams))/len(source_ngrams.trigrams)
                if trigram_ratio > 0.01:
                    print(authors[i] + " " + authors[j] + " " + str(trigram_ratio), file=trigram_file) 
    unigram_file.close()
    bigram_file.close()
    trigram_file.close()       

def load_feature(file_name):
    dbfile = open(file_name, 'rb')      
    db = pickle.load(dbfile) 
    dbfile.close() 
    return db

# Note that all ngrams of all authors are kept in memory currently
def feature_ngrams(dataset_filename):
    conn = connect_db_in_memory(dataset_filename)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT author FROM data")
    distinct_authors = cur.fetchall() # Fetchall returns a tuple ("author_name",)
    target = ('[deleted]',)
    distinct_authors.remove(target)
    comments_sum = 0
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words('english'))
    author_ngrams = {}
    for author in tqdm(distinct_authors):
        cur.execute("SELECT body FROM data WHERE author=?", author)
        comments = cur.fetchall() #rows holds ids of all comments made by the author
        unigrams_set = set()
        bigrams_set = set()
        trigrams_set = set()
        if len(comments) > 3:
            for comment in comments:
                sentences = nltk.sent_tokenize(comment[0])
                for sentence in sentences:
                    tokens = tokenizer.tokenize(sentence)
                    filtered_sentence = [w for w in tokens if not w in stop_words]
                    bigrams = ngrams(filtered_sentence, 2)
                    trigrams = ngrams(filtered_sentence,3)
                    for unigram in filtered_sentence: #Tokens are alredy unigram
                        unigrams_set.add(unigram)
                    for bigram in bigrams:
                        bigrams_set.add(bigram)
                    for trigram in trigrams:
                        trigrams_set.add(trigram)
             # Now we have a set of ngrams of each author
            author_ngram_sets = NgramSets()
            author_ngram_sets.unigrams = unigrams_set
            author_ngram_sets.bigrams = bigrams_set
            author_ngram_sets.trigrams = trigrams_set
            author_ngrams[author[0]] = author_ngram_sets

        comments_sum += len(comments)
        print("\n" + str((comments_sum/total_comments)*100) + "% of total main comments are processed.")
    ngrams_file = open('feature_ngrams.pkl', 'wb')
    pickle.dump(author_ngrams, ngrams_file)                      
    ngrams_file.close() 
    
    
def draw_graph_clusters(cluster_obj, output_filename):
    visual_style = dict()
    visual_style["bbox"] = (700, 600)
    visual_style["vertex_label"] = cluster_obj.graph.vs["name"]
    ig.plot(cluster_obj,output_filename,mark_groups = True,**visual_style)


# Used for a bug in igraph, floating point weights in file are problematic
def scale_graph(graph_filename, scale):
    for line in fileinput.input(graph_filename, inplace=1):
        line_components = line.split()
        new_weight = int(float(line_components[2])*1000)
        if new_weight > 0:       
            print(line_components[0] + " " + line_components[1] + " " + str(new_weight))   
    

def community_detection(graph_file):
    print("Reading the graph file...")
    normalize_graph(graph_file)
    scale_graph(graph_file,1000) # Related to a bug in igraph
    g = ig.Graph.Read_Ncol(graph_file,directed=False)
    g.simplify(combine_edges='sum')
    
    print("Running Community Detection Algorithms...")
    communities = {}
        
    # Walktrap Method, time O(mn^2) and space O(n^2) in the worst case
    '''
    dendogram = g.community_walktrap(weights=g.es["weight"], steps = 4)
    clusters = dendogram.as_clustering()
    for i in range(len(clusters.membership)):
        communities[clusters.graph.vs[i]["name"]] = (clusters.membership[i],)
    '''
    
    # Fast Greedy, greedy optimization of modularity,
    # n vertices and m edges is O(mdlogn) where d is the depth of the 
    # dendrogram describing the community structure
    dendogram = g.community_fastgreedy(weights=g.es["weight"])
    clusters = dendogram.as_clustering()
    for i in range(len(clusters.membership)):
        communities[clusters.graph.vs[i]["name"]] = (clusters.membership[i],)
        
    # Leiden, TODO parameters
    clusters = dendogram = g.community_leiden(weights=g.es["weight"])
    #clusters = dendogram.as_clustering()
    for i in range(len(clusters.membership)):
        communities[clusters.graph.vs[i]["name"]] += (clusters.membership[i],)    
    
    # Infomap method, space constraint
    '''
    clusters = g.community_infomap(edge_weights=g.es["weight"])
    #clusters = dendogram.as_clustering()
    for i in range(len(clusters.membership)):
        communities[clusters.graph.vs[i]["name"]] += (clusters.membership[i],)
    '''
    
    # Label Propogation
    clusters = g.community_label_propagation(weights=g.es["weight"])
    #clusters = dendogram.as_clustering()
    for i in range(len(clusters.membership)):
        communities[clusters.graph.vs[i]["name"]] += (clusters.membership[i],)
    
    # Newman's eigenvector
    clusters = g.community_leading_eigenvector(weights=g.es["weight"])
    #clusters = dendogram.as_clustering()
    for i in range(len(clusters.membership)):
        communities[clusters.graph.vs[i]["name"]] += (clusters.membership[i],)
        
    # Multi level clustering algorithm
    clusters = g.community_multilevel(weights=g.es["weight"])
    for i in range(len(clusters.membership)):
        communities[clusters.graph.vs[i]["name"]] += (clusters.membership[i],)
    
      
    return communities

def recursive_parenting(cur, edges, comment_id, weight):
    # Get the author and parent id of the comment
    cur.execute("SELECT author,parent_id FROM data WHERE name=?", comment_id)
    parent_comment = cur.fetchall() 
    if weight*recursive_weigh_factor<0.05: # For efficieny 
        return
    if parent_comment: # There is a row for the parent comment in DB
        # All comment names(id) are unique, there should be only one result
        parent_comment = parent_comment[0]
        # Now, parent_comment[0] holds parent author, parent_comments[1] holds parent id
        edges[parent_comment[0]] = edges.get(parent_comment[0], 0) + weight
        # Recursive call to look parent of parents
        recursive_parenting(cur,edges,(parent_comment[1],),weight*recursive_weigh_factor)
        

# cur is cursor object from the database connection
def generate_graph_data(cur, graph_file_name):
    graph_file = open(graph_file_name, "w")
    cur.execute("SELECT DISTINCT author FROM data")
    distinct_authors = cur.fetchall() # Fetchall returns a tuple ("author_name",)
    target = ('[deleted]',)
    distinct_authors.remove(target)
    comments_sum = 0
    for author in tqdm(distinct_authors):
        cur.execute("SELECT name,parent_id FROM data WHERE author=?", author)
        comments = cur.fetchall() #rows holds ids of all comments made by the author
        if len(comments) > 3:     
            edges = {} # Edges for the current author
            for comment in comments:
                # Recursively look parent author of the commment and parent of parents to make connections
                recursive_parenting(cur,edges,(comment[1],),base_weight)
                # TODO how about authors making comment onto the same comment, does it needed?
            #All edges belonging to the current author are added, now append it to the file
            edges.pop(author[0], None) #delete the edge of author to itself(sub comment of their own comments)
            edges.pop("[deleted]", None) #delete the edge of author to the [deleted] authors
            for key,val in edges.items():
                print(author[0] + " " + key + " " + str(val), file=graph_file)
        comments_sum += len(comments)
        print("\n" + str((comments_sum/total_comments)*100) + "% of total main comments are processed.")
    
    graph_file.close()
    

def connect_db_in_memory(dataset_filename):
    # Dataset is an sqlite Databse
    conn = None
    try:
        conn = sqlite3.connect(dataset_filename)
        print(sqlite3.version)
        # Processed the database file in memory
        dest = sqlite3.connect(':memory:')
        conn.backup(dest)
        return dest
    except Error as e:
        print(e)
        return None
    
def db_to_graph(dataset_filename):
    conn = connect_db_in_memory(dataset_filename)
    cur = conn.cursor()
    generate_graph_data(cur, "reddit_casualconversation_network.txt") 
    if conn:
        conn.close()

if __name__ == '__main__':
    print("lölölöl")
