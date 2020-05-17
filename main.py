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

dataset_filename = "../reddit-comments-may-2015/CasualConversations_sub.db"
acronyms_filename = "./list_acronym.txt"
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
    

# Gives the average sentence length for each author
def feature_sentence_length(dataset_filename):
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
                total_sentence_length = sum(map(lambda x: len(x), sentences))
                author_grammars[author[0]] = total_sentence_length/len(sentences)
        
        comments_sum += len(comments)
        print("\n" + str((comments_sum/total_comments)*100) + "% of total main comments are processed.")
    
    ngrams_file = open('feature_sentence_length.pkl', 'wb')
    pickle.dump(author_grammars, ngrams_file)                      
    ngrams_file.close()


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
                    sentence = sentence.lower()
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
    

# Evalutaes the clusters based on a number of metrics, communities1 is the ground truth
# Refer to https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation
def evaluate(communities1, communities2, number_of_algorithms):
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
    

def remove_unused_authors(graph_filename, output_filename):
    conn = connect_db_in_memory(dataset_filename)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT author FROM data")
    distinct_authors = cur.fetchall() # Fetchall returns a tuple ("author_name",)
    target = ('[deleted]',)
    distinct_authors.remove(target)
    our_authors = []
    for author in tqdm(distinct_authors):
        cur.execute("SELECT body FROM data WHERE author=?", author)
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
    print("Fast Greedy is running...")
    dendogram = g.community_fastgreedy(weights=g.es["weight"])
    clusters = dendogram.as_clustering()
    for i in range(len(clusters.membership)):
        communities[clusters.graph.vs[i]["name"]] = (clusters.membership[i],)
    print("Fast Greedy is done...")
        
    print("Leiden is running...")
    # Leiden, TODO parameters
    clusters = dendogram = g.community_leiden(weights=g.es["weight"], n_iterations=4)
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
    #clusters = dendogram.as_clustering()
    for i in range(len(clusters.membership)):
        communities[clusters.graph.vs[i]["name"]] += (clusters.membership[i],)
    print("Label Propogation is done...")
    
    # Newman's eigenvector
    print("Newman's EigenVector is running...")
    clusters = g.community_leading_eigenvector(weights=g.es["weight"])
    #clusters = dendogram.as_clustering()
    for i in range(len(clusters.membership)):
        communities[clusters.graph.vs[i]["name"]] += (clusters.membership[i],)
    print("Newman's EigenVector is done...")
        
    print("Multi level clustering is running...")
    # Multi level clustering algorithm
    clusters = g.community_multilevel(weights=g.es["weight"])
    for i in range(len(clusters.membership)):
        communities[clusters.graph.vs[i]["name"]] += (clusters.membership[i],)
    print("Multi level clustering is done...")
      
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

def feature_profanity(dataset_filename):
    conn = connect_db_in_memory(dataset_filename)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT author FROM data")
    distinct_authors = cur.fetchall() # fetchall returns a tuple ("author_name",)
    target = ('[deleted]',)
    distinct_authors.remove(target)
    comments_sum = 0;
    author_profanity = {}
    for author in tqdm(distinct_authors):
        cur.execute("SELECT body FROM data WHERE author=?", author)
        comments = cur.fetchall() # rows hold ids of all comments made by the author
        if (len(comments) < 4):
            continue
        comments = list(map(lambda x: x[0], comments)) 
        single_text = ''.join(comments)
        profanity_rate = predict_prob([single_text])
        author_profanity[author[0]] = profanity_rate[0]
        comments_sum += len(comments)
        print("\n" + str((comments_sum / total_comments)*100) + "% of total main comments are processed.")
    profanity_file = open('feature_profanity.pkl', 'wb')
    pickle.dump(author_profanity, profanity_file)
    profanity_file.close()
           
def feature_punct(dataset_filename):
    conn = connect_db_in_memory(dataset_filename)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT author FROM data")
    distinct_authors = cur.fetchall() # fetchall returns a tuple ("author_name",)
    target = ('[deleted]',)
    distinct_authors.remove(target)
    comments_sum = 0
    author_punct = {}
    for author in tqdm(distinct_authors):
        cur.execute("SELECT body FROM data WHERE author=?", author)
        comments = cur.fetchall() # rows hold ids of all comments made by the author
        if (len(comments) < 4):
            continue
        punct_count = 0
        character_count = 0
        for comment in comments:
            for character in comment[0]:
                if character in string.punctuation:
                    punct_count += 1
                character_count += len(comment[0])
        punct_rate = punct_count / character_count
        author_punct[author[0]] = punct_rate
        comments_sum += len(comments)
        print("\n" + str((comments_sum / total_comments)*100) + "% of total main comments are processed.")
    punct_file = open('feature_punct.pkl', 'wb')
    pickle.dump(author_punct, punct_file)
    punct_file.close()

def feature_emoji(dataset_filename):
    conn = connect_db_in_memory(dataset_filename)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT author FROM data")
    distinct_authors = cur.fetchall() # fetchall returns a tuple ("author_name",)
    target = ('[deleted]',)
    distinct_authors.remove(target)
    comments_sum = 0
    author_emoji = {}
    for author in tqdm(distinct_authors):
        cur.execute("SELECT body FROM data WHERE author=?", author)
        comments = cur.fetchall() # rows hold ids of all comments made by the author
        if (len(comments) < 4):
            continue
        emoji_count = 0;
        character_count = 0;
        for comment in comments:
            emoji_count += len(re.findall(r'(?::|;|=|x)(?:-)?(?:\)|\(D|P|S)',comment[0]))
            character_count += len(comment[0])
        emoji_rate = emoji_count / character_count
        author_emoji[author[0]] = emoji_rate
        comments_sum += len(comments)
        print("\n" + str((comments_sum / total_comments)*100) + "% of total main comments are processed.")
    emoji_file = open('feature_emoji.pkl', 'wb')
    pickle.dump(author_emoji, emoji_file)
    emoji_file.close()
    
def feature_uppercase(dataset_filename):
    conn = connect_db_in_memory(dataset_filename)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT author FROM data")
    distinct_authors = cur.fetchall() # fetchall returns a tuple ("author_name",)
    target =('[deleted]',)
    distinct_authors.remove(target)
    comments_sum = 0
    author_uppercase = {}
    for author in tqdm(distinct_authors):
        cur.execute("SELECT body FROM data WHERE author=?", author)
        comments = cur.fetchall() # rows hold ids of all comments made by the author
        if (len(comments) < 4):
            continue
        uppercase_count = 0
        character_count = 0
        for comment in comments:
            for character in comment[0]:
                if (character.isupper()):
                    uppercase_count += 1
                character_count += len(comment[0])
        uppercase_rate = uppercase_count / character_count
        author_uppercase[author[0]] = uppercase_rate
        comments_sum += len(comments)
        print("\n" + str((comments_sum / total_comments)*100) + "% of total main comments are processed.")
    uppercase_file = open('feature_uppercase.pkl', 'wb')
    pickle.dump(author_uppercase, uppercase_file)
    uppercase_file.close()

def feature_zipf(dataset_filename):
    conn = connect_db_in_memory(dataset_filename)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT author FROM data")
    distinct_authors = cur.fetchall() # fetchall returns a tuple ("author_name",)
    target = ('[deleted]',)
    distinct_authors.remove(target)
    comments_sum = 0
    author_zipf = {}
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words('english'))
    for author in tqdm(distinct_authors):
        cur.execute("SELECT body FROM data WHERE author=?", author)
        comments = cur.fetchall() # rows hold ids of all comments made by the author
        if (len(comments) < 4):
            continue
        fd = FreqDist()
        for comment in comments:
            sentences = nltk.sent_tokenize(comment[0])
            for sentence in sentences:
                sentence = sentence.lower()
                tokens = tokenizer.tokenize(sentence)
                filtered_sentence = [w for w in tokens if not w in stop_words]
                for word in filtered_sentence:
                    fd[word] += 1
        ranks = np.array([])
        freqs = np.array([])
        for rank, word in enumerate(fd):
            ranks = np.append(ranks, rank + 1)
            freqs = np.append(freqs, fd[word])
        slope = linefit_slope(np.log(ranks), np.log(freqs))
        author_zipf[author[0]] = slope
        comments_sum += len(comments)
        print("\n" + str((comments_sum / total_comments)*100) + "% of total main comments are processed.")
    zipf_file = open('feature_zipf.pkl', 'wb')
    pickle.dump(author_zipf, zipf_file)
    zipf_file.close()
    
def feature_acronym(dataset_filename, acronyms_filename):
    conn = connect_db_in_memory(dataset_filename)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT author FROM data")
    distinct_authors = cur.fetchall()
    target = ('[deleted]',)
    distinct_authors.remove(target)
    comments_sum = 0
    acronym_count = 0
    character_count = 0
    tokenizer = RegexpTokenizer(r'\w+')
    author_acronym = {}
    acronyms = open(acronyms_filename).read().splitlines()
    for author in tqdm(distinct_authors):
        cur.execute("SELECT body FROM data WHERE author=?", author)
        comments = cur.fetchall()
        if (len(comments) > 3):
            for comment in comments:
                sentences = nltk.sent_tokenize(comment[0])
                for sentence in sentences:
                    sentence = sentence.lower()
                    tokens = tokenizer.tokenize(sentence)
                    if "tl&dr" in sentence or "tl;dr" in sentence:
                        acronym_count += 1
                    for token in tokens:
                        if token in acronyms:
                            acronym_count += 1
                character_count += len(comment[0])
            acronym_rate = acronym_count / character_count
            author_acronym[author[0]] = acronym_rate
            comments_sum += len(comments)
            print("\n" + str((comments_sum / total_comments)*100) + "% of total main comments are processed.")
    acronym_file = open('feature_acronym.pkl', 'wb')
    pickle.dump(author_acronym, acronym_file)
    acronym_file.close()
    

# calculates the slope of the best-fitting line
def linefit_slope(x, y):
    slope = (((np.mean(x) * np.mean(y)) - np.mean(x*y)) / ((np.mean(x)**2) - np.mean(x**2)))
    return slope

# reads a feature file and generate clusters based on k-means clustering
def feature_to_cluster(feature_filename, num_clusters):
    data = load_feature(feature_filename)
    authors = list(data.keys())
    points = np.array([])
    for i in tqdm(range(len(authors))):
        points = np.append(points, data[authors[i]])
    points = points.reshape(-1, 1) # because the original data is 1D, reshaping is needed
    kmeans = KMeans(n_clusters = num_clusters)
    kmeans = kmeans.fit(points)
    labels = kmeans.predict(points)
    clusters = {}
    for i in tqdm(range(len(authors))):
        clusters[authors[i]] = (labels[i],)
    return clusters

if __name__ == '__main__':
    print("lölölöl")
    print("lelele")
