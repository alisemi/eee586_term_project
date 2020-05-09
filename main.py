#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sqlite3
from sqlite3 import Error
from tqdm import tqdm

dataset_filename = "../reddit-comments-may-2015/CasualConversations_sub.db"
recursive_weigh_factor = 1/2
base_weight = 1
total_comments = 234694

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

if __name__ == '__main__':
    conn = connect_db_in_memory(dataset_filename)
    cur = conn.cursor()
    generate_graph_data(cur, "reddit_casualconversation_network.txt") 
    if conn:
        conn.close()