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

dataset_filename = "../reddit-comments-may-2015/CasualConversations_sub.db"
recursive_weigh_factor = 1/2
base_weight = 1
total_comments = 234694

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
    ngrams_file = open('feature_ngrams.pkl', 'ab')
    pickle.dump(author_ngrams, ngrams_file)                      
    ngrams_file.close() 
    

def community_detection(graph_file):
    g = ig.Graph.Read_Ncol(graph_file)
    g_undirected = g.as_undirected(combine_edges="sum")
    dendrogram = g_undirected.community_fastgreedy()
    
    clustering=dendrogram.as_clustering()
    membership=clustering.membership
    return membership

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
    overlapping_ngrams()
