from nltk.corpus import brown
import word2vec
import logging
import os
import nltk
import datetime
from multiprocessing import Pool
from six import iteritems, itervalues
from gensim.utils import smart_open, to_unicode
from itertools import product
import re
from numpy.core.fromnumeric import argsort

def clean_name(name):
    return '.'.join(os.path.splitext(os.path.basename(name))[:-1])

def to_section_name(name):
    return re.sub(r'_', r'-', os.path.basename(name).split('.')[0])
    
def multiply_file(fname):
    with open(fname) as fn:
        lines = fn.readlines()
    new_lines = []
    new_lines.append(': ' + to_section_name(fname) + '\n')
    for mult_line in product(lines, repeat=2):
        if mult_line[0] == mult_line[1]:
            continue
        new_lines.append(" ".join([line.strip() for line in mult_line]) + '\n')
    new_fname = os.path.join('res', 'mult', os.path.basename(fname))
    with open(new_fname, 'w') as fn:
        fn.writelines(new_lines)

def pos_file(fname, out_fname=None, overwrite=False, pos=None):
    print(fname)
    if not out_fname:
        out_fname = fname+'.pos' 
    if os.path.exists(out_fname) and not overwrite:
        return
    with open(fname) as f_:
        lines = f_.readlines()
        print(len(lines))
        tok_lines = [nltk.word_tokenize(sent.decode('utf-8', 'replace')) for sent in lines]
        print(datetime.datetime.now())
        new_lines = []
        for tok_line in tok_lines:
            if (tok_lines.index(tok_line))%1000 == 0:
                print(datetime.datetime.now())
                print(tok_lines.index(tok_line))
            if pos:
                new_lines.append(' '.join(['_'.join([w,p]) for w,p in zip(tok_line, pos)]) + '\n')
            else:
                pos_line = nltk.pos_tag(tok_line)
                new_lines.append(' '.join(['_'.join([w,p]) for w,p in pos_line]) + '\n')
    with open(out_fname,'w') as f:
        f.writelines([line.encode('utf-8') for line in new_lines])

def join_files(fnames, target_fname):
    with open(target_fname, 'w') as f:
        pass
    with open(target_fname, 'a') as f:
        for fname in fnames:
            print(fname)
            with open(fname) as f_:
                f.write(f_.read()) 

def to_text(fpath, pos=False, max_lines=None):
    with open(fpath) as fp:
        data = fp.readlines()[:max_lines]
    sentences = []
    sentence = []
    for line in data:
        if len(line.split()) < 2+3*pos:
            sentences.append(sentence)
            sentence = []
        else:
            word = line.split()[1]
            if pos:
                word += '_'+line.split()[4] 
            sentence.append(unicode(word))
    with open(fpath + '.txt' +'.pos'*pos, 'w') as fp:
        fp.writelines([' '.join(sentence).lower()+'\n' for sentence in sentences])

def encode_heb(fpath):
    with open(fpath) as fp1:
        data = fp1.read()
    data = unicode(data, 'utf-8', 'escape')
    with open(os.path.join('res', 'heb_code')) as fp2:
        heb_code = fp2.readlines()
    heb_code = [unicode(line, 'utf-8', 'escape') for line in heb_code]
    heb_code = {line.split()[0]: line.split()[1] for line in heb_code}
    data = ''.join([heb_code[c] if c in heb_code else c for c in data])
    with open(fpath+'.enc', 'w') as fp:
        fp.write(data.lower())

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

def remove_pos(word):
    return word.split('_')[0]

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
    fpath = os.path.join('res', 'words', 'ambiguous_verbs_heb')
    encode_heb(fpath)
    encode_heb(fpath)

    fpath = os.path.join('res', 'hebtb.sd.final.conll')
    to_text(fpath)
    to_text(fpath, pos=True)
    
#     fname = 'ambiguous_verbs_mixed'
    fname = 'ambiguous_verbs_heb.enc'
#     fname = 'ambiguous_verbs'
    fpath = os.path.join('res', 'words', fname)
    pos_file(fpath, overwrite=True, pos=['VB'.lower(), 'VB-TOINFINITIVE'.lower()])
    multiply_file(fpath)
    multiply_file(fpath+'.pos')
    
#     name = 'text8'
#     name = 'brown'
#     name = 'GoogleNews-vectors-negative300'
#     name = 'news_pos.bin'
    name = 'hebtb.sd.final.conll.txt.bin'
    pos_name = clean_name(name) +'.pos' + '.bin'
    w2v = W2V(name)
    w2v_pos = W2V(pos_name)

#     print(len(w2v.model.vocab))
#     print(w2v.model.vocab.items()[:10])
#     print(w2v.model.similarity('add_VB','remove_VB'))
#     print(len(model.vocab.keys()))    

    questions_fpath = os.path.join('res', 'mult',fname)
    eval1 = w2v.evaluate_model(questions_fpath)
    eval2 = w2v_pos.evaluate_model(questions_fpath)
    compare_section(eval1, eval2, to_section_name(fname))
    
if __name__ == '__main__':
    main()