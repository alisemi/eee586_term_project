#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pickle
from tqdm import tqdm
import language_check
from profanity_check import predict, predict_prob
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import ngrams
from nltk import FreqDist
import utils
import numpy as np
import string
import re
import os

# Gives the average sentence length for each author
def feature_sentence_length(dataset_filename_pkl):
    feature_filename = 'features/' + os.path.basename(dataset_filename_pkl)[:-4] + '_feature_sentence_length.pkl'
    if not os.path.isfile(feature_filename):
        author_data = utils.load_feature(dataset_filename_pkl)
        author_grammars = {}
        for author in tqdm(author_data):
            single_text = ''.join(author_data[author])
            sentences = nltk.sent_tokenize(single_text)
            if len(sentences) > 0: # Just a precaution
                total_sentence_length = sum(map(lambda x: len(x), sentences))
                author_grammars[author] = total_sentence_length/len(sentences)        
        ngrams_file = open(feature_filename, 'wb')
        pickle.dump(author_grammars, ngrams_file)                      
        ngrams_file.close()
    return feature_filename
    

# Gives the grammar_mistake/sentence for each author
def feature_grammar_check(dataset_filename_pkl):
    feature_filename = 'features/' + os.path.basename(dataset_filename_pkl)[:-4] + '_feature_grammar_check.pkl'
    if not os.path.isfile(feature_filename):
        tool = language_check.LanguageTool('en-US')
        author_data = utils.load_feature(dataset_filename_pkl)
        author_grammars = {}
        for author in tqdm(author_data):
            single_text = ''.join(author_data[author])
            sentences = nltk.sent_tokenize(single_text)
            if len(sentences) > 0: # Just a precaution
                matches = tool.check(single_text)
                author_grammars[author] = len(matches)/len(sentences)
                
        ngrams_file = open(feature_filename, 'wb')
        pickle.dump(author_grammars, ngrams_file)                      
        ngrams_file.close()
    return feature_filename
    
 
class NgramSets:
    pass

# Note that all ngrams of all authors are kept in memory currently
def feature_ngrams(dataset_filename_pkl):
    feature_filename = 'features/' + os.path.basename(dataset_filename_pkl)[:-4] + '_feature_ngrams.pkl'
    if not os.path.isfile(feature_filename):
        author_data = utils.load_feature(dataset_filename_pkl)
        tokenizer = RegexpTokenizer(r'\w+')
        stop_words = set(stopwords.words('english'))
        author_ngrams = {}
        for author in tqdm(author_data):
            comments = author_data[author]
            unigrams_set = set()
            bigrams_set = set()
            trigrams_set = set()
            for comment in comments:
                sentences = nltk.sent_tokenize(comment)
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
            author_ngrams[author] = author_ngram_sets
            
        ngrams_file = open(feature_filename, 'wb')
        pickle.dump(author_ngrams, ngrams_file)                      
        ngrams_file.close()
    return feature_filename
    
def feature_profanity(dataset_filename_pkl):
    feature_filename = 'features/' + os.path.basename(dataset_filename_pkl)[:-4] + '_feature_profanity.pkl'
    if not os.path.isfile(feature_filename):
        author_data = utils.load_feature(dataset_filename_pkl)
        author_profanity = {}
        for author in tqdm(author_data):
            single_text = ''.join(author_data[author])
            profanity_rate = predict_prob([single_text])
            author_profanity[author] = profanity_rate[0]
        
        profanity_file = open(feature_filename, 'wb')
        pickle.dump(author_profanity, profanity_file)
        profanity_file.close()
    return feature_filename
           
def feature_punct(dataset_filename_pkl):
    feature_filename = 'features/' + os.path.basename(dataset_filename_pkl)[:-4] + '_feature_punct.pkl'
    if not os.path.isfile(feature_filename):
        author_data = utils.load_feature(dataset_filename_pkl)
        author_punct = {}
        for author in tqdm(author_data):
            comments = author_data[author]
            punct_count = 0
            character_count = 0
            for comment in comments:
                for character in comment:
                    if character in string.punctuation:
                        punct_count += 1
                    character_count += len(comment)
            punct_rate = punct_count / character_count
            author_punct[author] = punct_rate
    
        punct_file = open(feature_filename, 'wb')
        pickle.dump(author_punct, punct_file)
        punct_file.close()
    return feature_filename

def feature_emoji(dataset_filename_pkl):
    feature_filename = 'features/' + os.path.basename(dataset_filename_pkl)[:-4] + '_feature_emoji.pkl'
    if not os.path.isfile(feature_filename):
        author_data = utils.load_feature(dataset_filename_pkl)
        author_emoji = {}
        for author in tqdm(author_data):
            comments = author_data[author]
            emoji_count = 0;
            character_count = 0;
            for comment in comments:
                emoji_count += len(re.findall(r'(?::|;|=|x)(?:-)?(?:\)|\(D|P|S)',comment))
                character_count += len(comment)
            emoji_rate = emoji_count / character_count
            author_emoji[author] = emoji_rate
            
        emoji_file = open(feature_filename, 'wb')
        pickle.dump(author_emoji, emoji_file)
        emoji_file.close()
    return feature_filename
    
def feature_uppercase(dataset_filename_pkl):
    feature_filename = 'features/' + os.path.basename(dataset_filename_pkl)[:-4] + '_feature_uppercase.pkl'
    if not os.path.isfile(feature_filename):
        author_data = utils.load_feature(dataset_filename_pkl)
        author_uppercase = {}
        for author in tqdm(author_data):
            comments = author_data[author]
            uppercase_count = 0
            character_count = 0
            for comment in comments:
                for character in comment:
                    if (character.isupper()):
                        uppercase_count += 1
                    character_count += len(comment)
            uppercase_rate = uppercase_count / character_count
            author_uppercase[author] = uppercase_rate
            
        uppercase_file = open(feature_filename, 'wb')
        pickle.dump(author_uppercase, uppercase_file)
        uppercase_file.close()
    return feature_filename

def feature_zipf(dataset_filename_pkl):
    feature_filename = 'features/' + os.path.basename(dataset_filename_pkl)[:-4] + '_feature_zipf.pkl'
    if not os.path.isfile(feature_filename):
        author_data = utils.load_feature(dataset_filename_pkl)
        author_zipf = {}
        tokenizer = RegexpTokenizer(r'\w+')
        stop_words = set(stopwords.words('english'))
        for author in tqdm(author_data):
            comments = comments = author_data[author]
            fd = FreqDist()
            for comment in comments:
                sentences = nltk.sent_tokenize(comment)
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
            author_zipf[author] = slope
            
        zipf_file = open(feature_filename, 'wb')
        pickle.dump(author_zipf, zipf_file)
        zipf_file.close()
    return feature_filename
    
def feature_acronym(dataset_filename_pkl, acronyms_filename):
    feature_filename = 'features/' + os.path.basename(dataset_filename_pkl)[:-4] + '_feature_acronym.pkl'
    if not os.path.isfile(feature_filename):
        author_data = utils.load_feature(dataset_filename_pkl)
        acronym_count = 0
        character_count = 0
        tokenizer = RegexpTokenizer(r'\w+')
        author_acronym = {}
        acronyms = open(acronyms_filename).read().splitlines()
        for author in tqdm(author_data):
            comments = author_data[author]
            for comment in comments:
                sentences = nltk.sent_tokenize(comment)
                for sentence in sentences:
                    sentence = sentence.lower()
                    tokens = tokenizer.tokenize(sentence)
                    if "tl&dr" in sentence or "tl;dr" in sentence:
                        acronym_count += 1
                    for token in tokens:
                        if token in acronyms:
                            acronym_count += 1
                character_count += len(comment)
            acronym_rate = acronym_count / character_count
            author_acronym[author] = acronym_rate
            
        acronym_file = open(feature_filename, 'wb')
        pickle.dump(author_acronym, acronym_file)
        acronym_file.close()
    return feature_filename
    

# calculates the slope of the best-fitting line
def linefit_slope(x, y):
    slope = (((np.mean(x) * np.mean(y)) - np.mean(x*y)) / ((np.mean(x)**2) - np.mean(x**2)))
    return slope

# First one is ngrams, it is kind of special for graph construction
def extract_features(dataset_filename, used_authors):
    # Generate pickle file for faster construction
    dataset_filename_pkl = utils.reorganize_dataset(dataset_filename,used_authors)
    feature_filenames = []
    feature_filenames.append(feature_ngrams(dataset_filename_pkl))
    feature_filenames.append(feature_acronym(dataset_filename_pkl, 'list_acronym.txt'))
    feature_filenames.append(feature_uppercase(dataset_filename_pkl))
    feature_filenames.append(feature_emoji(dataset_filename_pkl))
    feature_filenames.append(feature_punct(dataset_filename_pkl))
    feature_filenames.append(feature_profanity(dataset_filename_pkl))
    feature_filenames.append(feature_grammar_check(dataset_filename_pkl))
    feature_filenames.append(feature_sentence_length(dataset_filename_pkl))
    feature_filenames.append(feature_zipf(dataset_filename_pkl))
    return feature_filenames
    