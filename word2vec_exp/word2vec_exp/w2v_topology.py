from gensim.models import word2vec
import logging
import random
from numpy import average, histogram
from time import strftime, gmtime
from collections import Counter
import matplotlib.pyplot as plt
import pylab
from math import acos
from word2vec_exp import logger

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# sentences = word2vec.Text8Corpus('res/text8')
# model = word2vec.Word2Vec(sentences)
#
# model.save('res/text8.model')
#
# model.save_word2vec_format('res/text8.model.bin', binary=True)
model = word2vec.Word2Vec.load_word2vec_format('res/news.bin', binary=True)

def get_similarity(word1,word2):
    return(model.similarity(word1,word2))

def plot_lists(l, min_len=None):
    fig = pylab.figure()
    ax = fig.add_subplot(221)
    l = [x for x in l if len(x) >= min_len]
    for y in l:
        ax.plot(y)
    ax = fig.add_subplot(222)
    for y in l:
        ax.plot(y)
    ax.set_xscale('log')
    ax = fig.add_subplot(223)
    for y in l:
        ax.plot(y)
    ax.set_yscale('log')
    ax = fig.add_subplot(224)
    for y in l:
        ax.plot(y)
    ax.set_yscale('log')
    ax.set_xscale('log')
    pylab.draw()    

if __name__ == '__main__':
    vocabulary = model.vocab.keys()
    dist_lists = []
    hop_list = []
    len_list = []
    attractors = []
    N = len(vocabulary)
    for i in range(1000):
        if i%100 == 0:
            print((i, strftime("%Y-%m-%d %H:%M:%S", gmtime())))
        word_list = []
        prev_word_tuple = ('', 0)
        r = random.randint(0,len(vocabulary))
        original_word = vocabulary[r]
        word_tuple = (original_word,0)
        dist_list = []
#         word_list.append(word_tuple)
        for j in range(100):
            next_word_tuple = model.most_similar(word_tuple[0])[0]
            if prev_word_tuple == next_word_tuple:
                break
            prev_word_tuple = word_tuple
            word_tuple = next_word_tuple
            word_list.append((word_tuple[0], acos(word_tuple[1])))
            dist = model.similarity(original_word, word_tuple[0])
            if dist > 1:
                dist_list.append(0.0)
            else:
                dist_list.append(acos(dist))
        hop_list.append([h for w,h in word_list])
        dist_lists.append(dist_list)
        len_list.append(len(word_list))
        attractors.append(word_tuple[0]) 
        attractors.append(prev_word_tuple[0])
    w2v.logger.logging.info(len_list)
    w2v.logger.logging.info(average(len_list))
    w2v.logger.logging.info(histogram(len_list))
    hist, bins = histogram(len_list)
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, hist, align='center')
    plt.draw()
    w2v.logger.logging.info(Counter(attractors))
    plot_lists(hop_list, 6)
    plot_lists(dist_lists, 6)
    
    plt.show()