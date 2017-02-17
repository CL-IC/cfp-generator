from __future__ import print_function

import glob

import gensim
import numpy as np
import random

from ftfy import fix_text
from nltk import tokenize, word_tokenize
from nltk.translate.bleu_score import sentence_bleu

import rnn
import nlp
import sys

from keras.utils.visualize_util import plot
import numpy as np
reload(sys)
sys.setdefaultencoding('latin-1')

def load_text(filename):
    text = open(filename).read().lower()
    print('corpus length:', len(text))

    chars = sorted(list(set(text)))
    print('total chars:', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    return text, chars, char_indices, indices_char

def generate_sequences(text, maxlen, step = 3):
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('nb sequences:', len(sentences))
    return sentences, next_chars

def create_dataset(sentences, maxlen, char_indices, next_chars, char_len):
    X = np.zeros((len(sentences), maxlen, char_len), dtype=np.bool)
    y = np.zeros((len(sentences), char_len), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1
    return X, y

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(model, text, maxlen, sentence, char_indices, indices_char, char_len, diversity = 0.5,no_chars=400):
    start_index = random.randint(0, len(text) - maxlen - 1)
    generated = ''
    # sentence = text[start_index: start_index + maxlen]
    generated += sentence

    for i in range(no_chars):
        x = np.zeros((1, maxlen, char_len))
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1.

        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, diversity)
        next_char = indices_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

    return generated

def main():
    files = glob.glob('..\\input\\topic_*.txt')
    corpus, dictionary = nlp.build_corpus_from_topics(files)
    lsi, index = nlp.build_similarity_index(corpus, dictionary)
    for topic in xrange(7,10):
        print(str(topic))
        text, chars, char_indices, indices_char = load_text('..\\input\\topic_'+str(topic)+'.txt')
        char_len = len(chars)
        maxlen = 40
        no_chars = 1000
        model = rnn.build_model(maxlen, char_len)
        rnn.load_model(model,'..\\input\\'+str(topic)+'weights59.h5')
        n_evals = 25
        scores = np.zeros(n_evals)

        for i in xrange(n_evals):
            start_index = random.randint(0, len(text) - maxlen - 1)
            sentence = text[start_index: start_index + maxlen]
            generated = generate_text(model, text, maxlen, sentence, char_indices, indices_char, char_len, no_chars=no_chars)
            # print(generated)
            scores[i] = nlp.get_similarity(generated, topic, dictionary, lsi, index)
        print(str(topic) + ': ' + str(np.average(scores)))

if __name__ == "__main__":

    main()