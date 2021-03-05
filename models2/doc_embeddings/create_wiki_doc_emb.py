import pickle
import random
import numpy as np
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity


def load_word_embeddings(filepath):
    print("Loading word embeddings from", filepath)
    emb = KeyedVectors.load_word2vec_format(fname=filepath, binary=False)
    return emb

languages = ['fi', 'sv']

# create Wiki test data
wiki_file = "/proj/zosa/data/wiki/wikialign_clean_fi-sv_test.pkl"
wiki_data = pickle.load(open(wiki_file, 'rb'))

# open aligned word embeddings
filepaths = {'fi': '/proj/zosa/data/fasttext/wiki.fi.align.vec',
             'sv': '/proj/zosa/data/fasttext/wiki.sv.align.vec'}
word_embeddings = {}
for lang in languages:
    emb = load_word_embeddings(filepaths[lang])
    word_embeddings[lang] = emb

# wiki doc embeddings
doc_embeddings = {'fi':[], 'sv':[]}
for article in wiki_data:
    doc_pair = {}
    for lang in languages:
        tokens = article[lang].lower().split()
        doc_emb = np.array([word_embeddings[lang][w] for w in tokens if w in word_embeddings[lang]])
        if doc_emb.shape[0] > 0:
            doc_emb = np.mean(doc_emb)
            doc_pair[lang] = emb
    if len(doc_pair) == 2:
        for lang in languages:
            doc_embeddings[lang].append(doc_pair[lang])

# sanity check
print("FI doc embeddings:", len(doc_embeddings['fi']))
print("SV doc embeddings:", len(doc_embeddings['sv']))
for i in range(len(doc_embeddings['fi'])):
    doc_fi = doc_embeddings['fi'][i]
    doc_sv = doc_embeddings['sv'][i]
    cos_sim = cosine_similarity(doc_fi, doc_sv)[0]
    print("Doc pair", i, ":", cos_sim)

if len(doc_embeddings['fi']) == len(doc_embeddings['sv']):
    dump_file = wiki_file[:-4]+"_emb.pkl"
    with open(dump_file, 'wb') as f:
        pickle.dump(doc_embeddings, f)
        f.close()



