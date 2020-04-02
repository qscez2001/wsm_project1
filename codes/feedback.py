import os
from VectorSpace import VectorSpace
import pandas as pd
import numpy as np
import nltk
import util

file_path = "../documents"
files = os.listdir(file_path)

documents = []

for f in files:
    with open("../documents/{}".format(f)) as file:
        documents.append(file.read())

vectorSpace = VectorSpace(documents)

# tfidf + cos
print("TF-IDF Weighting + Cosine Similarity")

# calculate
scores = vectorSpace.search(["drill wood sharp"], "cos", idf=True)
# Indices of N largest elements in list 
indices = np.argpartition(scores,-5)[-5:]

# save as (index, value)
d = {}
for i in indices:
    d[i] = scores[i]

# sort dict by value instead of key
sd = sorted(d.items(), key=lambda item: item[1], reverse=True)
# print(sd)

sorted_indices = [s[0] for s in sd]
sorted_scores = [s[1] for s in sd]

# find docid
docid = []
for index in sorted_indices:
    x = files[index]
    docid.append(os.path.splitext(x)[0])

round_score = [round(score, 6) for score in sorted_scores]

d = {'docID': docid, 'Score': round_score}

print(pd.DataFrame(data=d))

# The new query term weighting = [1 * original query + 0.5 * feedback query]
'''
For instance, suppose the index vector is 
["network" ,"computer" , "share", "ask", "soccer", "song"], 
the query is "network", and the content of the feedback document is:
Jimmy shares songs via the computer network.
Then we will get a new query vector like this:
1 * [1, 0, 0, 0, 0, 0] + 0.5 * [1, 1, 1, 0, 0, 1] = [1.5, 0.5, 0.5, 0, 0, 0.5]
'''
# print(sorted_indices[0])
# get ranked first vector
first_vector = vectorSpace.documentVectors[sorted_indices[0]]
# print(first_vector)
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
# print(words)

# do the pos tagging to words
tagged = nltk.pos_tag(words)
# print(tagged)

feedback = [0] * len(vectorSpace.vectorKeywordIndex)

# find the verb and noun words and transform to feedback vector
pos = ['NN', 'VB', 'VBP', 'VBD', 'VBG']
for tup in tagged:
    if tup[1] in pos:
        # print(my_dict[tup[0]])
        feedback[my_dict[tup[0]]] += 1
    
# print(feedback)
feedback = np.array(feedback)

queryVector = vectorSpace.buildQueryVector(["drill wood sharp"])

new_query = queryVector + 0.5 * feedback
new_query = list(new_query)
print(new_query)


# feedback rating
print("TF-IDF Weighting + Cosine Similarity feedback")
scores = [util.cosine(new_query, documentVector) for documentVector in vectorSpace.documentVectors]

# Indices of N largest elements in list 
indices = np.argpartition(scores,-5)[-5:]

# save as (index, value)
d = {}
for i in indices:
    d[i] = scores[i]

# sort dict by value instead of key
sd = sorted(d.items(), key=lambda item: item[1], reverse=True)
# print(sd)

sorted_indices = [s[0] for s in sd]
sorted_scores = [s[1] for s in sd]

# find docid
docid = []
for index in sorted_indices:
    x = files[index]
    docid.append(os.path.splitext(x)[0])

round_score = [round(score, 6) for score in sorted_scores]

d = {'docID': docid, 'Score': round_score}

print(pd.DataFrame(data=d))