import glob
from gensim import models

from gensim import corpora
from gensim import similarities
from gensim.corpora import dictionary
import ftfy as ftfy


def build_corpus_from_topics(files):
    texts = []
    for file in files:
        print file
        texts.append(open(file).read().lower().split())
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    return corpus, dictionary

def build_similarity_index(corpus, dictionary, p_num_topic=10):
    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=p_num_topic)
    index = similarities.MatrixSimilarity(lsi[corpus])
    return lsi, index

def get_similarity(text, topic, dictionary, lsi, index):
    vec_bow = dictionary.doc2bow(text.lower().split())
    vec_lsi = lsi[vec_bow]  # convert the query to LSI space
    sims = index[vec_lsi]
    # sims = sorted(enumerate(sims), key=lambda item: -item[1])
    # print(sims)
    return sims[topic]

if __name__ == "__main__":
    files = glob.glob('..\\input\\topic_*.txt')
    corpus, dictionary = build_corpus_from_topics(files)
    lsi, index = build_similarity_index(corpus, dictionary)
    print(get_similarity("valami", lsi, index))