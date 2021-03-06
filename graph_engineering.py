#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 19:28:13 2020

@author: ali
"""
import os
from tqdm import tqdm
import igraph as ig
from shutil import copyfile
import fileinput
import utils
import math
import statistics
import itertools

recursive_weigh_factor = 1/2
base_weight = 1

# merge graph files
def merge_graphs(file_list, scalar_list, output_filename):
    if os.path.exists(output_filename):
        os.remove(output_filename) 
    for i in tqdm(range(len(file_list))):
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
    return output_filename


def normalize_graph(graph_filename):
    graph_file = open(graph_filename)
    line = graph_file.readline() 
    if not line: # Graph is empty, no need to normalize
        return
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
    
    
def create_graphs_from_features(feature_file_list):
    graph_files = []
    for file_name in tqdm(feature_file_list):
        graph_files.append(feature_to_graph(file_name))
    return graph_files
   
    
def feature_to_graph(feature_file):
    graph_filename = "networks/" + os.path.basename(feature_file)[:-4] + "_graph.txt"
    graph_file = open(graph_filename, "w")
    feature = utils.load_feature(feature_file)
    feature_list = list(feature.items())
    feature_list.sort(key=lambda tup: tup[1])
    distance_adjustment = 0.01
    
    # Setting the log scale
    feature_logged = []
    # Get the smallest value after 0
    smallest = 0
    for val in feature_list:
        if val[1] > 0:
            smallest = val
            break
    if smallest == 0: # It means that every author has 0 value
        return graph_filename # Then return empty graph
        
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
            if distance < std_deviation*distance_adjustment: # Similartiy should be at least 1-standart_deviation
                similarity = 1-distance
                edges.append((feature_normed[j][0],similarity))
            else: # Otherwise it is considered as zero, as list is sorted, no need to look at the rest
                break
        # Write the edges to file
        for edge in edges:    
            print(feature_normed[i][0] + " " + edge[0] + " " + str(edge[1]), file=graph_file) 
    return graph_filename
    

def overlapping_ngrams(ngram_feature_filename):
    author_ngrams = utils.load_feature(ngram_feature_filename)
    unigram_filename = "networks/" + os.path.basename(ngram_feature_filename)[:-4] + "_unigram_overlap_graph.txt"
    bigram_filename = "networks/" + os.path.basename(ngram_feature_filename)[:-4] + "_bigram_overlap_graph.txt"
    trigram_filename = "networks/" + os.path.basename(ngram_feature_filename)[:-4] + "_trigram_overlap_graph.txt" 
    unigram_file = open(unigram_filename, "w")
    bigram_file = open(bigram_filename, "w")
    trigram_file = open(trigram_filename, "w")
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
    return [unigram_filename, bigram_filename, trigram_filename]
             
    
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
    


def remove_unused_authors(graph_filename, output_filename, cur):
    cur.execute("SELECT DISTINCT author FROM " + utils.table_name)
    distinct_authors = cur.fetchall() # Fetchall returns a tuple ("author_name",)
    target = ('[deleted]',)
    distinct_authors.remove(target)
    our_authors = []
    for author in tqdm(distinct_authors):
        cur.execute("SELECT body FROM " + utils.table_name + " WHERE author=?", author)
        comments = cur.fetchall() #rows holds ids of all comments made by the author
        if len(comments) > 3:
            our_authors.append(author)
    our_authors = list(map(lambda x: x[0], our_authors))
    
    input_file = open(graph_filename,'r')
    output_file = open(output_filename, 'w')
    
    while True:     
        # Get next line from file 
        line = input_file.readline() 
        # if line is empty 
        # end of file is reached 
        if not line: 
            break
        line_components = line.split()
        
        if line_components[0] in our_authors and line_components[1] in our_authors:
            print(line[:-1],file=output_file)
            
    input_file.close()
    output_file.close()
    

def community_detection(graph_file):
    print("Reading the graph file...")  
    g = ig.Graph.Read_Ncol(graph_file,directed=False)
    communities = {}
    if not g: # If this is an empty graph
        return communities
    g.simplify(combine_edges='sum')
    g = g.components().giant()
    
    numClusters = []
    
    print("Running Community Detection Algorithms...")    
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
    print("Fast Greedy is running...")
    dendogram = g.community_fastgreedy(weights=g.es["weight"])
    clusters = dendogram.as_clustering()
    numClusters.append(len(clusters))
    for i in range(len(clusters.membership)):
        communities[clusters.graph.vs[i]["name"]] = (clusters.membership[i],)
    print("Fast Greedy is done...")
        
    print("Leiden is running...")
    # Leiden, TODO parameters
    clusters = g.community_leiden(weights=g.es["weight"], n_iterations=4)
    numClusters.append(len(clusters))
    #clusters = dendogram.as_clustering()
    for i in range(len(clusters.membership)):
        communities[clusters.graph.vs[i]["name"]] += (clusters.membership[i],) 
    print("Leiden is done...")
    
    # Infomap method, space constraint
    '''
    clusters = g.community_infomap(edge_weights=g.es["weight"])
    #clusters = dendogram.as_clustering()
    for i in range(len(clusters.membership)):
        communities[clusters.graph.vs[i]["name"]] += (clusters.membership[i],)
    '''
    
    # Label Propogation
    print("Label Propogation is running...")
    clusters = g.community_label_propagation(weights=g.es["weight"])
    numClusters.append(len(clusters))
    #clusters = dendogram.as_clustering()
    for i in range(len(clusters.membership)):
        communities[clusters.graph.vs[i]["name"]] += (clusters.membership[i],)
    print("Label Propogation is done...")
    
    # Newman's eigenvector
    
    print("Newman's EigenVector is running...")
    clusters = g.community_leading_eigenvector(weights=g.es["weight"])
    numClusters.append(len(clusters))
    #clusters = dendogram.as_clustering()
    for i in range(len(clusters.membership)):
        communities[clusters.graph.vs[i]["name"]] += (clusters.membership[i],)
    print("Newman's EigenVector is done...")
    
        
    print("Multi level clustering is running...")
    # Multi level clustering algorithm
    clusters = g.community_multilevel(weights=g.es["weight"])
    numClusters.append(len(clusters))
    for i in range(len(clusters.membership)):
        communities[clusters.graph.vs[i]["name"]] += (clusters.membership[i],)
    print("Multi level clustering is done...")
      
    return communities, g.vs["name"],numClusters

def recursive_parenting(cur, edges, comment_id, weight):
    # Get the author and parent id of the comment
    cur.execute("SELECT author,parent_id FROM " + utils.table_name + " WHERE name=?", comment_id)
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
        

# No parenting, more sql
def generate_graph_data_sql(cur,graph_file_name):
   graph_file = open(graph_file_name, "w")
   comment_count = 20
   #sql = 'SELECT author FROM ' + utils.table_name + ' GROUP BY author HAVING COUNT(*) > ' + str(comment_count) + " AND author IS NOT '[deleted]'"
   #sql2 = 'SELECT name,parent_id FROM ' + utils.table_name + ' WHERE author IN (' + sql + ')'
   #cur.execute(sql2)
   sql6 = ("WITH our_authors AS (SELECT author FROM " + utils.table_name + " GROUP BY author HAVING COUNT(*) > " + 
           str(comment_count) + " AND author IS NOT '[deleted]') SELECT replier.author, host.author FROM " +
           utils.table_name + " host INNER JOIN " + utils.table_name + " replier ON replier.parent_id = host.name " 
           "WHERE replier.author IN our_authors AND host.author IN our_authors")
   cur.execute(sql6)
   connections = cur.fetchall()
   authors = [(key, [num for _, num in value]) for key, value in itertools.groupby(connections, lambda x: x[0])]
   for author in authors:
       edges = {}
       for neighbor in author[1]:
           edges[neighbor] = edges.get(neighbor, 0) + 1
       for key,val in edges.items():
           print(author[0] + " " + key + " " + str(val), file=graph_file)
   graph_file.close()
   normalize_graph(graph_file_name)
   scale_graph(graph_file_name,1000) # Related to a bug in igraph 

# cur is cursor object from the database connection
def generate_graph_data(cur, graph_file_name, parenting):
    graph_file = open(graph_file_name, "w")
    cur.execute("SELECT DISTINCT author FROM " + utils.table_name)
    distinct_authors = cur.fetchall() # Fetchall returns a tuple ("author_name",)
    target = ('[deleted]',)
    distinct_authors.remove(target)
    for author in tqdm(distinct_authors):
        cur.execute("SELECT name,parent_id FROM " + utils.table_name + " WHERE author=?", author)
        comments = cur.fetchall() #rows holds ids of all comments made by the author
        if len(comments) > 100:     
            edges = {} # Edges for the current author
            for comment in comments:
                if parenting:
                    # Recursively look parent author of the commment and parent of parents to make connections
                    recursive_parenting(cur,edges,(comment[1],),base_weight)
                else:
                    cur.execute("SELECT author,parent_id FROM " + utils.table_name + " WHERE name=?", (comment[1],))
                    parent_comment = cur.fetchall()
                    if parent_comment:
                        parent_comment = parent_comment[0]
                        edges[parent_comment[0]] = edges.get(parent_comment[0], 0) + 1
                # TODO how about authors making comment onto the same comment, does it needed?
            #All edges belonging to the current author are added, now append it to the file
            edges.pop(author[0], None) #delete the edge of author to itself(sub comment of their own comments)
            edges.pop("[deleted]", None) #delete the edge of author to the [deleted] authors
            for key,val in edges.items():
                print(author[0] + " " + key + " " + str(val), file=graph_file)
    graph_file.close()
    remove_unused_authors(graph_file_name, graph_file_name[:-4] + "_processed.txt", cur )
    os.remove(graph_file_name)
    os.rename(graph_file_name[:-4] + "_processed.txt", graph_file_name)
    normalize_graph(graph_file_name)
    scale_graph(graph_file_name,1000) # Related to a bug in igraph      
    
def db_to_graph(dataset_filename, graph_filename, parenting=True):
    if not os.path.isfile(graph_filename):
        conn = utils.connect_db_in_memory(dataset_filename)
        cur = conn.cursor()
        #generate_graph_data(cur, graph_filename, False) 
        generate_graph_data_sql(cur, graph_filename) 
        if conn:
            conn.close()