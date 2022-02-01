import torch
from sent2vec.vectorizer import Vectorizer
from scipy import spatial

import pickle
sentences = [
    "activity with 189 calorie consumption.",
    "activity with 200 calorie consumption.",
    "activity with 300 calorie consumption."
]
vectorizer = Vectorizer()
vectorizer.bert(sentences)
vectors = vectorizer.vectors
print(vectors)
dist_1 = spatial.distance.cosine(vectors[0], vectors[1])
dist_2 = spatial.distance.cosine(vectors[1], vectors[2])
print('dist_1: {0}, dist_2: {1}'.format(dist_1, dist_2))
print(spatial.distance.cosine([1,2,3,4,5], [1,2,3,4,5]))