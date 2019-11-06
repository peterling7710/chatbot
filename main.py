import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json

from flask import Flask, render_template, url_for, request, jsonify
#, request, redirect, send_from_directory, session

from flask_cors import CORS, cross_origin


app = Flask(__name__)
CORS(app)

app.config["SECRET_KEY"] = b'\xa3\x08\xf6\x18\xe5#\xb4)H\xee\xfe\x81\xb3\xb9Ky'
app.secret_key = b'\xa3\x08\xf6\x18\xe5#\xb4)H\xee\xfe\x81\xb3\xb9Ky'

def model_init(in_dim, out_dim, n_epoch, batch_size, training, output):
    tensorflow.reset_default_graph()


    net = tflearn.input_data(shape=[None, in_dim])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, out_dim, activation="softmax")
    net = tflearn.regression(net)

    model = tflearn.DNN(net)

    try:
        model.load("model.tflearn")
    except:
        model.fit(training, output, n_epoch=n_epoch, batch_size=batch_size, show_metric=True)
        model.save("model.tflearn")

    return model

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

def generate_train_set(data_file):
    with open(data_file) as file:
        data = json.load(file)

    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w.lower()) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)


    training = numpy.array(training)
    output = numpy.array(output)

    return data, labels, words, training, output

@app.route("/chat", methods=["GET","POST"])
@cross_origin()
def chat():

    data, labels, words, training, output = generate_train_set("training/intents.json")

    model = model_init(len(training[0]), len(output[0]), 1000, 8, training, output)

    if request.method == 'GET':
        
        return jsonify(["Hello! How can I help you today?"])

    if request.method == 'POST':
        inp = request.get_json()
        print(inp)
        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']


        return jsonify([ random.choice(responses) ])
         
if __name__ == '__main__':
    app.run(debug=True, port=5001)
