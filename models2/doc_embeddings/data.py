import pickle
import json
import random
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import KeyedVectors
from collections import Counter

from nltk.corpus import stopwords
stops_sv = list(set(stopwords.words('swedish')))
stops_fi = list(set(stopwords.words('finnish')))
stops_en = list(set(stopwords.words('english')))
stops_de = list(set(stopwords.words('german')))
stops = {'fi': stops_fi, 'sv': stops_sv, 'en': stops_en, 'de': stops_de}

import string
punct = set(string.punctuation)


def load_word_embeddings(path):
    # Load word embeddings
    print("Loading word embeddings from", path)
    emb = KeyedVectors.load_word2vec_format(fname=path, binary=False)
    return emb


def create_document_embedding(tokens, embeddings, lang):
    tf_tokens = dict(Counter(tokens))
    words_in_vocab = [word for word in tf_tokens.keys() if word in embeddings and word not in stops[lang]]
    if len(words_in_vocab) == 0:
        #raise Exception("No matching tokens with the vocabulary were found.")
        return None
    tfs = np.array([tf_tokens[word] for word in words_in_vocab])
    embs = np.array([embeddings[word] for word in words_in_vocab])
    normalized_tfs = tfs / np.linalg.norm(tfs)
    for i in range(len(tfs)):
        embs[i] = embs[i] * tfs[i]
    doc_emb = embs.sum(axis=0)
    doc_emb = doc_emb / np.linalg.norm(tfs)
    return doc_emb


# return vocab of wikipedia train set
def get_wiki_docs_vocab(filepath, min_df=5):
    languages = ['fi', 'sv']
    vocab_dict = {}
    corpus_dict = {}
    data = pickle.load(open(filepath, 'rb'))
    for lang in languages:
        corpus = [pair[lang].lower() for pair in data]
        vectorizer = CountVectorizer(stop_words=stops[lang], min_df=min_df)
        counts = vectorizer.fit_transform(corpus)
        vocab = vectorizer.get_feature_names()
        array_counts = counts.toarray()
        total_counts = array_counts.sum(axis=0)
        vocab_counts = {vocab[i]: total_counts[i] for i in range(len(vocab))
                        if vocab[i] not in punct and vocab[i] not in stops[lang]}
        vocab_dict[lang] = vocab_counts
        corpus_dict[lang] = corpus
    return corpus_dict, vocab_dict


def get_yle_docs_vocab(filepath, max_docs=30000, min_df=8):
    languages = ['fi', 'sv']
    data = json.load(open(filepath))
    vocab_dict = {}
    docs_dict = {'fi':[], 'sv':[]}
    for lang in languages:
        print("Lang:", lang.upper())
        docs = []
        for art in data[lang]:
            text = art['content'].lower()
            docs.append(text)
        print("total articles:", len(docs))
        random.shuffle(docs)
        if len(docs) > max_docs:
            sampled_docs = docs[:max_docs]
        else:
            sampled_docs = docs
        vectorizer = CountVectorizer(stop_words=stops[lang], min_df=min_df)
        counts = vectorizer.fit_transform(sampled_docs)
        vocab = vectorizer.get_feature_names()
        valid_vocab = [w for w in vocab if len(re.findall('[0-9]+', w)) == 0]
        print(lang.upper(), "vocab:", len(valid_vocab))
        vocab_dict[lang] = valid_vocab
        docs_dict[lang] = sampled_docs
    return docs_dict, vocab_dict


def get_yle_aligned(filepath, min_df=5):
    print("Fetching aligned articles from", filepath)
    languages = ['fi', 'sv']
    vocab_dict = {}
    data = json.load(open(filepath, 'r'))
    data = data[:10000] #limit aligned articles to 10k
    unique_docs = {'fi':{}, 'sv':{}}
    for aligned_art in data:
        for lang in languages:
            article_list = aligned_art[lang]
            for article in article_list:
                if article['id'] not in unique_docs[lang]:
                    doc = article['content']
                    art_id = article['id']
                    unique_docs[lang][art_id] = doc
    for lang in languages:
        print(lang.upper(), "articles:", len(unique_docs[lang]))
        docs = [unique_docs[lang][art_id] for art_id in unique_docs[lang]]
        print("Docs:", docs[:2])
        vectorizer = CountVectorizer(stop_words=stops[lang], min_df=min_df)
        counts = vectorizer.fit_transform(docs)
        vocab = vectorizer.get_feature_names()
        print(lang.upper(), "vocab:", len(vocab))
        array_counts = counts.toarray()
        total_counts = array_counts.sum(axis=0)
        vocab_counts = {vocab[i]: total_counts[i] for i in range(len(vocab))
                        if vocab[i] not in punct and vocab[i] not in stops[lang]}
        vocab_dict[lang] = vocab_counts
    return unique_docs, vocab_dict


def get_denews_docs_vocab(filepath, min_df=2):
    print("Getting DENews articles from", filepath)
    languages = ['en', 'de']
    data = json.load(open(filepath, 'r'))
    docs_dict = {'en' :[], 'de': []}
    vocab_dict = {'en': [], 'de': []}
    for lang in languages:
        print("Lang:", lang.upper())
        articles = data[lang]
        docs = [articles[k]['content'].lower() for k in list(articles.keys())]
        docs_dict[lang] = docs
        print("total articles:", len(docs))
        vectorizer = CountVectorizer(stop_words=stops[lang], min_df=min_df)
        counts = vectorizer.fit_transform(docs)
        vocab = vectorizer.get_feature_names()
        valid_vocab = [w for w in vocab if re.match(r'^([\s\d]+)$', w) is None]
        print("valid vocab:", len(valid_vocab))
        vocab_dict[lang] = valid_vocab
    return docs_dict, vocab_dict


def get_denews_docs_paired(filepath):
    print("Getting paired DENews articles from", filepath)
    languages = ['en', 'de']
    data = json.load(open(filepath, 'r'))
    doc_ids = list(data['en'].keys())
    paired_docs = {'en':[], 'de':[]}
    for doc_id in doc_ids:
        if doc_id in data['en'] and doc_id in data['de']:
            for lang in languages:
                text = data[lang][doc_id]['headline'] + data[lang][doc_id]['content']
                paired_docs[lang].append(text.lower())
    # sanity check
    for lang in languages:
        print(lang.upper(), "docs:", len(paired_docs[lang]))
    return paired_docs


def get_vocab_scores(docs, train_vocab, score_type='tf'):
    vectorizer = CountVectorizer()
    if score_type == 'tfidf':
        vectorizer = TfidfVectorizer(use_idf=True)
    vectors = vectorizer.fit_transform(docs)
    vocab = vectorizer.get_feature_names()
    totals = vectors.toarray().sum(axis=0)
    vocab_scores = {}
    for i, word in enumerate(vocab):
        if word in train_vocab:
            vocab_scores[word] = totals[i]
    return vocab_scores

#
# import codecs
# from gensim.models import KeyedVectors
#
# file = open('joint_28_en.txt', 'rt', encoding='utf-8')
# outfile = codecs.open('en.txt', 'w', encoding='utf8')
# outfile.write('200000 300\n')
# for i,line in enumerate(file):
#     parts = line.split(" ")
#     word = parts[0]
#     vec = ' '.join(parts[-300:])
#     correct_line = word + ' ' + vec
#     outfile.write(correct_line)
#     print(i)
#
# outfile.close()
# emb = KeyedVectors.load_word2vec_format(fname='en.txt', binary=False)