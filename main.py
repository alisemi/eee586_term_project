#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

dataset_filename = "RC_2008-12"


if __name__ == '__main__':
    dataset_file = open(dataset_filename, "r")
    comments = [] # list of dict objects
    while True: 
        # Get next line from file 
        line = dataset_file.readline() 
        # if line is empty 
        # end of file is reached 
        if not line: 
            break
        comments.append(json.loads(line))
        
    dataset_file.close() 