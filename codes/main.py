import argparse
import os
import pprint

import nltk
import util
import numpy as np
import pandas as pd
from VectorSpace import VectorSpace

def cal_tfidf_ed(vectorSpace, files, query):
    print("TF-IDF Weighting + Euclidean Distance")

    # calculate
    scores = vectorSpace.search([query], "euc")

    # Indices of N smallest elements in list 
    indices = np.argpartition(scores,5)[:5]

    # save as (index, value)
    d = {}
    for i in indices:
        d[i] = scores[i]

    # sort dict by value instead of key (smallest)
    sd = sorted(d.items(), key=lambda item: item[1])

    sorted_indices = [s[0] for s in sd]
    sorted_scores = [s[1] for s in sd]

    # find docid
    docid = []
    for index in sorted_indices:
        x = files[index]
        docid.append(os.path.splitext(x)[0])

    round_score = [round(score, 6) for score in sorted_scores]

    d = {'docID': docid, 'Score': round_score}


    return pd.DataFrame(data=d)

def cal_tfidf_cs(vectorSpace, files, query):
    print("TF-IDF Weighting + Cosine Similarity")

    # calculate
    scores = vectorSpace.search([query], "cos", idf=True)
    # Indices of N largest elements in list 
    indices = np.argpartition(scores,-5)[-5:]

    # save as (index, value)
    d = {}
    for i in indices:
        d[i] = scores[i]

    # sort dict by value instead of key
    sd = sorted(d.items(), key=lambda item: item[1], reverse=True)


    sorted_indices = [s[0] for s in sd]
    sorted_scores = [s[1] for s in sd]

    # find docid
    docid = []
    for index in sorted_indices:
        x = files[index]
        docid.append(os.path.splitext(x)[0])

    round_score = [round(score, 6) for score in sorted_scores]

    d = {'docID': docid, 'Score': round_score}

    return sorted_indices, pd.DataFrame(data=d)

def cal_tf_ed(vectorSpace, files, query):
    print("Term Frequency (TF) Weighting + Euclidean Distance")

    # calculate
    scores = vectorSpace.search([query], "euc")

    # Indices of N smallest elements in list 
    indices = np.argpartition(scores,5)[:5]

    # save as (index, value)
    d = {}
    for i in indices:
        d[i] = scores[i]

    # sort dict by value instead of key (smallest)
    sd = sorted(d.items(), key=lambda item: item[1])

    sorted_indices = [s[0] for s in sd]
    sorted_scores = [s[1] for s in sd]

    # find docid
    docid = []
    for index in sorted_indices:
        x = files[index]
        docid.append(os.path.splitext(x)[0])

    round_score = [round(score, 6) for score in sorted_scores]

    d = {'docID': docid, 'Score': round_score}

    return pd.DataFrame(data=d)

def cal_tf_cs(vectorSpace, files, query):
    print("Term Frequency (TF) Weighting + Cosine Similarity")

    # calculate
    scores = vectorSpace.search([query], "cos")

    # TODO: find top largest N elements, each element represented in form of (index, value), then sort them by value
    # Indices of N largest elements in list 
    indices = np.argpartition(scores,-5)[-5:]

    # save as (index, value)
    d = {}
    for i in indices:
        d[i] = scores[i]

    # sort dict by value instead of key
    sd = sorted(d.items(), key=lambda item: item[1], reverse=True)


    sorted_indices = [s[0] for s in sd]
    sorted_scores = [s[1] for s in sd]

    # find docid
    docid = []
    for index in sorted_indices:
        x = files[index]
        docid.append(os.path.splitext(x)[0])

    round_score = [round(score, 6) for score in sorted_scores]

    d = {'docID': docid, 'Score': round_score}

    return pd.DataFrame(data=d)

def load_data(file_dir):
    files = os.listdir(file_dir)

    documents = []

    # update documents
    for f in files:
        with open("{}/{}".format(file_dir, f)) as file:
            documents.append(file.read())

    vectorSpace = VectorSpace(documents)

    return vectorSpace, files

def cal_fq_tfidf_cs(vectorSpace, files, query):
    # Feedback Query + TF-IDF Weighting + Cosine Similarity
    #
    # step 1
    # step 2
    # step 3
    # step 4

    sorted_indices , _ = cal_tfidf_cs(vectorSpace, files, query)
    # The new query term weighting = [1 * original query + 0.5 * feedback query]
    '''
    For instance, suppose the index vector is 
    ["network" ,"computer" , "share", "ask", "soccer", "song"], 
    the query is "network", and the content of the feedback document is:
    Jimmy shares songs via the computer network.
    Then we will get a new query vector like this:
    1 * [1, 0, 0, 0, 0, 0] + 0.5 * [1, 1, 1, 0, 0, 1] = [1.5, 0.5, 0.5, 0, 0, 0.5]
    '''

    # get ranked first vector
    first_vector = vectorSpace.documentVectors[sorted_indices[0]]

    my_dict = vectorSpace.vectorKeywordIndex

    def get_key(val): 
        for key, value in my_dict.items(): 
             if val == value: 
                 return key 
      
        return "key doesn't exist"

    # map the vector' item into words
    words = []
    for i in range(len(first_vector)):
        if first_vector[i] > 0:
            for j in range(first_vector[i]):
                words.append(get_key(i))

    # do the pos tagging to words
    tagged = nltk.pos_tag(words)

    feedback = [0] * len(vectorSpace.vectorKeywordIndex)

    # find the verb and noun words and transform to feedback vector
    pos = ['NN', 'VB', 'VBP', 'VBD', 'VBG']
    for tup in tagged:
        if tup[1] in pos:
            feedback[my_dict[tup[0]]] += 1


    feedback = np.array(feedback)

    queryVector = vectorSpace.buildQueryVector([query])

    new_query = queryVector + 0.5 * feedback
    new_query = list(new_query)
    


    # feedback rating
    print("Feedback Queries + TF-IDF Weighting + Cosine Similarity feedback")
    scores = [util.cosine(new_query, documentVector) for documentVector in vectorSpace.documentVectors]

    # Indices of N largest elements in list 
    indices = np.argpartition(scores,-5)[-5:]

    # save as (index, value)
    d = {}
    for i in indices:
        d[i] = scores[i]

    # sort dict by value instead of key
    sd = sorted(d.items(), key=lambda item: item[1], reverse=True)

    sorted_indices = [s[0] for s in sd]
    sorted_scores = [s[1] for s in sd]

    # find docid
    docid = []
    for index in sorted_indices:
        x = files[index]
        docid.append(os.path.splitext(x)[0])

    round_score = [round(score, 6) for score in sorted_scores]

    d = {'docID': docid, 'Score': round_score}

    return pd.DataFrame(data=d)

def main():
    # step 1 load data
    # step 2 print result of the calculation using Term Frequency (TF) Weighting + Cosine Similarity
    # step 3 print result of the calculation using Term Frequency (TF) Weighting + Euclidean Distance
    # step 4 print result of the calculation using TF-IDF Weighting + Cosine Similarity
    # step 5 print result of the calculation using TF-IDF Weighting + Euclidean Distance
    # step 6 print result of the calculation using Feedback Query + TF-IDF Weighting + Cosine Similarity
    # python main.py --query <query>
    parser = argparse.ArgumentParser()

    parser.add_argument('--query', dest="query", help="enter query", type=str)

    args = parser.parse_args()

    query = args.query

    vectorSpace, files = load_data("../documents")
    print(cal_tf_cs(vectorSpace, files, query))
    print(cal_tf_ed(vectorSpace, files, query))
    print(cal_tfidf_cs(vectorSpace, files, query)[1])
    print(cal_tfidf_ed(vectorSpace, files, query))
    print(cal_fq_tfidf_cs(vectorSpace, files, query))


if __name__ == "__main__":
    main()

