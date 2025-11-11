import random
from pathlib import Path
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize
from tensorflow.keras.models import load_model
# from tensorflow.keras.layers import Dense, Activation, Dropout
# from tensorflow.keras.optimizers import SGD


lemmatizer = WordNetLemmatizer()

json_path = Path(__file__).resolve().parent / "intents_mental.json" #

with open(json_path, "r", encoding="utf-8") as f:
    intents = json.load(f)

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')


def clean_up_sentence(sentence):
    
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]

    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)

    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose = 0)[0] 
    ERROR_THRESHOLD = 0.25

    # filter out predictions below threshold, i = index, r = probability
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD] 

    results.sort(key=lambda x: x[1], reverse=True) # sort by highest probabolity (r at index 1)

    return_list = []

    for r in results:
        return_list.append({'intent':classes[r[0]], 'probability': str(r[1])})

    return return_list


def get_response(intents_list, intents_json):
    
    if not intents_list:
        # try a specific fallback intent if you have one
        for tag in ("no-response", "neutral-response", "fallback_unknown"):
            for it in intents_json["intents"]:
                if it["tag"] == tag and it.get("responses"):
                    return random.choice(it["responses"])
        # generic fallback
        return "I'm not sure I understood that. Could you rephrase?"


    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']

    for i in list_of_intents:
        if i['tag'] == tag and i.get('responses'):
            return random.choice(i['responses'])
            
    return "I'm not sure I understood that. Could you rephrase?"


print('Go! Bot is running')

while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)
