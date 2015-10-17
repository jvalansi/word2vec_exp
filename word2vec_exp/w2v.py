from nltk.corpus import brown
import word2vec
import logging
import os
from multiprocessing import Pool
from six import iteritems, itervalues
from numpy.core.fromnumeric import argsort
from utils import clean_name, pos_file, join_files, encode_heb, to_text,\
    multiply_file, to_section_name, remove_pos

class W2V:
    def __init__(self, fname='news.bin'):
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        
        self.fname = fname
        fpath = os.path.join('res', fname)
        if not os.path.exists(fpath):
            self.create_model(clean_name(fname))
        self.model = self.get_model(fpath)
    
    def get_model(self, fpath):
        return word2vec.Word2Vec.load_word2vec_format(fpath, binary=True)
        
    def create_model(self, name, max_news=99, n_proc=1, window=3):
        model = word2vec.Word2Vec(window=window, workers=n_proc)
        if name == 'text8':
            sentences = word2vec.Text8Corpus(os.path.join('res', 'text8'))
            model.train(sentences)
        elif name == 'brown':
        #     sentences = word2vec.BrownCorpus(fpath)
            sentences = brown.sents()
            model.train(sentences)
        elif name.startswith('news'):
            fnames = ['news.en-{:05}-of-00100'.format(i+1) for i in range(max_news)]
            fpaths = [os.path.join('res', 'training-monolingual.tokenized.shuffled', fname) for fname in fnames]
            if name == 'news_pos':
                p = Pool(n_proc)
                p.map(pos_file, [fpath for fpath in fpaths if not os.path.exists(fpath+'.pos')])
#                 [pos_file(fpath) for fpath in fpaths if not os.path.exists(fpath+'.pos')]
                fpaths = [fpath+'.pos' for fpath in fpaths]
            target_fpath = os.path.join('res', name+'.txt')
            join_files(fpaths, target_fpath)
            with open(target_fpath) as fp:
                s = fp.read().lower()
            with open(target_fpath, 'w') as fp:
                fp.write(s)
            sentences = word2vec.LineSentence(target_fpath)
            model.build_vocab(sentences)
            model.train(sentences)
        else:
            fpath = os.path.join('res', name)
            with open(fpath) as fp:
                sentences = fp.readlines()
            sentences = [sentence.lower() for sentence in sentences]
            print(len(sentences))
            model.build_vocab(sentences)
            model.train(sentences)
         
    #     model.save(os.path.join('res',name+'.model'))
        model.save_word2vec_format(os.path.join('res',name+'.bin'), binary=True)  

    def get_similarity(self, word1, word2):
        return(self.model.similarity(word1,word2))

    def get_prediction(self, a,b,c, ok_index, restrict_vocab=30000):
        ok_vocab = dict(sorted(iteritems(self.model.vocab), 
                               key=lambda item: -item[1].count)[:restrict_vocab])
        ok_index = set(v.index for v in itervalues(ok_vocab))

        ignore = set(self.model.vocab[v].index for v in [a, b, c])  # indexes of words to ignore
        positive = [b, c]
        negative = [a]
        for index in argsort(self.model.most_similar(self.model, positive, negative, False))[::-1]:
            if index in ok_index and index not in ignore:
                predicted = self.model.index2word[index]
                break
        return predicted

    def evaluate_model(self, questions_fpath = os.path.join('res', 'questions-words.txt')):
        if clean_name(self.fname).endswith('pos'):
            pos_file(questions_fpath)
            questions_fpath = questions_fpath+'.pos' 
        return self.model.accuracy(questions_fpath)
    
def get_section(model_eval, name):
    for sec in model_eval:
        if sec['section'] == name:
            return sec
    print('not found')
    return None

def compare_section(eval1, eval2, section_name):
    sec1 = get_section(eval1, section_name)
    sec2 = get_section(eval2, section_name)
    sec1 =  {k: [tuple(remove_pos(w) for w in c) for c in v] for k,v in sec1.items()} 
    sec2 =  {k: [tuple(remove_pos(w) for w in c) for c in v] for k,v in sec2.items()} 
    correct1 = [c for c in sec1['correct'] if c not in sec2['OOV']]
    correct2 = [c for c in sec2['correct'] if c not in sec1['OOV']]
    print(len(correct1))
    print(len(correct2))
    missing1 = [c for c in correct2 if c not in correct1]
    missing2 = [c for c in correct1 if c not in correct2]
    print('missing1')
    for m in missing1:
        ind = sec1['incorrect'].index(m)
        print('the correct sequence is: '+str(m)+' but predicted: '+str(sec1['predicted'][ind]))
    print('missing2')
    for m in missing2:
        ind = sec2['incorrect'].index(m)
        print('the correct sequence is: '+str(m)+' but predicted: '+str(sec2['predicted'][ind]))
    return(missing1, missing2)


def main():
    
#     name = 'text8'
#     name = 'brown'
#     name = 'GoogleNews-vectors-negative300'
#     name = 'news_pos.bin'
    name = 'hebtb.sd.final.conll.txt.bin'
    name = 'news.bin'
    pos_name = clean_name(name) +'.pos' + '.bin'
    w2v = W2V(name)
    w2v_pos = W2V(pos_name)

#     print(len(w2v.model.vocab))
#     print(w2v.model.vocab.items()[:10])
#     print(w2v.model.similarity('add_VB','remove_VB'))
#     print(len(model.vocab.keys()))    

#     questions_fpath = os.path.join('res', 'mult',fname)
#     eval1 = w2v.evaluate_model(questions_fpath)
#     eval2 = w2v_pos.evaluate_model(questions_fpath)
#     compare_section(eval1, eval2, to_section_name(fname))
    
if __name__ == '__main__':
    main()