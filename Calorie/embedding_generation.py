import torch
from sent2vec.vectorizer import Vectorizer
import pickle
sentences = [
    "A person is swinging a golf club.",
    "A person is walking.",
    "A person is eating.",
    "A person is sitting down.",
    "A person is running.",
    "A person is standing.",
    "A person is climbing stairs.",
    "A person is kicking ball.",
    "A person is punching.",
    "A person is boxing.",
    "A person is jogging.",
    "A person is jumping rope.",
    "A person is doing pushups.",
    "A person is bowling.",
    "A person is playing basketball.",
    "A person is biking",
    "A person is swinging a tennis racket.",
    "A person is climbing rolk.",
    "A person is jumping and catching frisbee.",
    "A person is playing Taichi.",
    "A person is doing sky diving.",
    "A person is walking with a dog.",
    "A person is rowing.",
    "A person is skiing.",
    "A person is playing skate board.",
    "A person is doing jumping jacks.",
    "A person is doing squats.",
    "A person is doing yoga.",
    "A person is doing zumba and dancing.",
    "A person is sleeping.",
    "A person is driving in the car.",
    "A person is shopping.",
    "A person is doing stretching."
]

vectorizer = Vectorizer()
vectorizer.bert(sentences)
vectors = vectorizer.vectors
with open('/home/kpeng/calorie/MUSDL/Calorie/word_embedding_common_sport.pkl', 'wb') as f:
    pickle.dump(vectors, f)