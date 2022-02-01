import torch
from sent2vec.vectorizer import Vectorizer
import pickle
sentences = [
    "A person is standing and drinking water.",
    "A person is standing and eating snacks.",
    "A person is standing and brushing teeth.",
    "A person is standing and brushing hair.",
    "A person is standing and dropping something.",
    "A person is bending down and picking up something.",
    "A person is standing and throwing something using one arm.",
    "A person is sitting down on chair.",
    "A person is standing up from chair.",
    "A person is clapping with two hands.",
    "A person is standing and reading.",
    "A person is standing and writing.",
    "A person is standing and tearing up paper.",
    "A person is standing and wearing jacket.",
    "A person is standing and taking off the jacket.",
    "A person is sitting and wearing a shoe.",
    "A person is sitting and taking off a shoe.",
    "A person is sitting and wearing on glasses.",
    "A person is sitting and taking off glasses.",
    "A person is sitting and putting on a hat.",
    "A person is sitting and taking off a hat.",
    "A person is sitting and cheering up.",
    "A person is sitting and waving hands.",
    "A person is standing and kicking use one leg.",
    "A person is standing and reaching into pocket use one arm.",
    "A person is jumping with one foot.",
    "A person is jumping up.",
    "A person is standing and making a phone call.",
    "A person is standing and playing with phone or tablet.",
    "A person is sitting and typing on a laptop.",
    "A person is standing and pointing with finger.",
    "A person is standing and taking a selfie.",
    "A person is standing and checking time from watch.",
    "A person is standing and rubbing hands.",
    "A person is standing and nodding head.",
    "A person is standing and shaking head.",
    "A person is standing wiping face.",
    "A person is standing and saluting using one arm.",
    "A person is standing and putting the palms together.",
    "A person is standing, saying stop and crossing hands in front.",
    "A person is standing and sneezing or coughing.",
    "A person is standing and staggering.",
    "A person is falling.",
    "A person is standing and touching head.",
    "A person is standing and touching chest.",
    "A person is standing and touching back.",
    "A person is standing and touching neck.",
    "A person is bending down and covered mouth with one hand.",
    "A person is standing and using a fan with hand or paper."
]

vectorizer = Vectorizer()
vectorizer.bert(sentences)
vectors = vectorizer.vectors
with open('ntu_staffs/ntu_word_embedding_2.pkl', 'wb') as f:
    pickle.dump(vectors, f)