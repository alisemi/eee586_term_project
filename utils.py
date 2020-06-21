# -*- coding: utf-8 -*-

import pickle
from tqdm import tqdm
import sqlite3
from sqlite3 import Error
import collections
from sklearn import metrics
import matplotlib.pyplot as plt

table_name = 'data'

def reorganize_dataset(dataset_filename):
    conn = connect_db_in_memory(dataset_filename)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT author FROM " + table_name)
    distinct_authors = cur.fetchall() # Fetchall returns a tuple ("author_name",)
    target = ('[deleted]',)
    distinct_authors.remove(target)
    new_data = {}
    for author in tqdm(distinct_authors):
        cur.execute("SELECT body FROM " + table_name + " WHERE author=?", author)
        comments = cur.fetchall() #rows holds ids of all comments made by the author
        if len(comments) > 3:
            comments = list(map(lambda x: x[0], comments)) 
            new_data[author] = comments
    
    new_dataset_name = dataset_filename[:-3] + '.pkl'
    pickle_file = open(new_dataset_name,'wb')
    pickle.dump(new_data, pickle_file)
    pickle_file.close()
    return new_dataset_name
    
    
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
    
# Evalutaes the clusters based on a number of metrics, communities1 is the ground truth
# Refer to https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation
def evaluate(communities1, communities2, number_of_algorithms):
    if not communities2:
        return None
    c1 = communities1.copy()
    c2 = communities2.copy()
    
    # Making sure they have the same size
    c1_keys = list(c1.keys())
    c2_keys = list(c2.keys())
    
    for key in c1_keys:
        if not key in c2:
            del c1[key]
    for key in c2_keys:
        if not key in c1:
            del c2[key]
            
    od1 = collections.OrderedDict(sorted(c1.items()))
    od2 = collections.OrderedDict(sorted(c2.items()))
    evaluations = []
        
    for i in range(0,number_of_algorithms):
        cluster1_list = list(map(lambda x: x[i], od1.values()))
        cluster2_list = list(map(lambda x: x[i], od2.values()))
        ari = metrics.adjusted_rand_score(cluster1_list, cluster2_list)
        ami = metrics.adjusted_mutual_info_score(cluster1_list, cluster2_list) 
        v_score = metrics.v_measure_score(cluster1_list, cluster2_list)
        fmc = metrics.fowlkes_mallows_score(cluster1_list, cluster2_list)
        evaluations.append((ari,ami,v_score,fmc))
    return evaluations


# Get rand index for each community algoritms
def rand_index(communities1, communities2, number_of_algorithms):
    rand_index_values = []
    all_keys = set(communities1.keys()) | set(communities2.keys())
    all_keys = list(all_keys)
    n_choose_2 = (len(all_keys) * (len(all_keys)-1))/2
    for k in range(0,number_of_algorithms):
        true_positive_negative = 0
        for i in range(0,len(all_keys)):
            source_cluster1 = -1
            source_cluster2 = -1
            if all_keys[i] in communities1:
                    source_cluster1 = communities1[all_keys[i]][k]           
            if all_keys[i] in communities2:
                    source_cluster2 = communities2[all_keys[i]][k]
            
            for j in range(i+1,len(all_keys)):
                same1 = False
                same2 = False
                target_cluster1 = -1
                target_cluster2 = -1
                if all_keys[j] in communities1:
                    target_cluster1 = communities1[all_keys[j]][k]
                if all_keys[j] in communities2:
                    target_cluster2 = communities2[all_keys[j]][k]
                        
                if source_cluster1 == target_cluster1:
                    same1 = True
                if source_cluster2 == target_cluster2:
                    same2 = True 
                
                if same1 == same2: # They are either in same clusters or both in different clusters
                    true_positive_negative += 1
        rand_index_values.append(true_positive_negative/n_choose_2)
    return rand_index_values    
    
    
def load_feature(file_name):
    dbfile = open(file_name, 'rb')      
    db = pickle.load(dbfile) 
    dbfile.close() 
    return db


# For evaluations othher than single clustering algorithms on linguistic data, such as k-means
def evaluate_cluster_to_community(communities,clusters,number_of_algorithms):
    c1 = communities.copy()
    c2 = clusters.copy()
    
    # Making sure they have the same size
    c1_keys = list(c1.keys())
    c2_keys = list(c2.keys())
    
    for key in c1_keys:
        if not key in c2:
            del c1[key]
    for key in c2_keys:
        if not key in c1:
            del c2[key]
            
    od1 = collections.OrderedDict(sorted(c1.items()))
    od2 = collections.OrderedDict(sorted(c2.items()))
    evaluations = []
        
    for i in range(0,number_of_algorithms):
        cluster1_list = list(map(lambda x: x[i], od1.values()))
        cluster2_list = list(map(lambda x: x[0], od2.values()))
        ari = metrics.adjusted_rand_score(cluster1_list, cluster2_list)
        ami = metrics.adjusted_mutual_info_score(cluster1_list, cluster2_list) 
        v_score = metrics.v_measure_score(cluster1_list, cluster2_list)
        fmc = metrics.fowlkes_mallows_score(cluster1_list, cluster2_list)
        evaluations.append((ari,ami,v_score,fmc))
    return evaluations

# Evaluations are list of results
def plot_results(evaluations, cluster_name,file_name):
    #x= ["Adjusted Rand Index", "Adjusted Mutual Information", "V-measure","Fowlkes-Mallows Index"]
    x= ["ARI", "AMI", "V-measure","FMI"]
    community_algorithms = ["Fast Greedy","Leiden","Label Propogation",'''"Newman's EigenVector"''',"Multi-level Clustering"]
    
    if not evaluations:
        return
    for i in range(len(evaluations)):
        plt.plot(x,list(evaluations[i]),label = community_algorithms[i])
    
    plt.title("Evaluations for " + cluster_name) 
    # naming the x axis 
    plt.xlabel("Evaluation Metrics") 
    # naming the y axis 
    plt.ylabel("Score") 
    # show a legend on the plot 
    plt.legend() 
    # Save to a file
    plt.savefig(file_name)
    # function to show the plot 
    plt.show()
    