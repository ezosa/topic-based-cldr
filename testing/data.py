import numpy as np
import json
from scipy.stats import entropy
from gensim.models import KeyedVectors


def load_word_embeddings(path):
    # Load word embeddings
    print("Loading word embeddings from", path)
    emb = KeyedVectors.load_word2vec_format(fname=path, binary=False)
    return emb


def compute_jsd(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    p /= p.sum()
    q /= q.sum()
    m = (p + q) / 2
    return (entropy(p, m) + entropy(q, m)) / 2


def compute_kld(p, q):
    return sum(p[i] * np.log(p[i]/q[i]) for i in range(len(p)))


def get_denews_paired(json_file):
    languages = ['en', 'de']
    docs = json.load(open(json_file, 'r'))
    art_ids = list(docs['en'].keys())
    paired_docs = {'en': [], 'de': []}
    for art_id in art_ids:
        if art_id in docs['en'] and art_id in docs['de']:
            for lang in languages:
                text = docs[lang][art_id]['content'].lower()
                paired_docs[lang].append(text)
    # sanity check - must have same no. of articles
    print("EN articles:", len(paired_docs['en']))
    print("DE articles:", len(paired_docs['de']))
    return paired_docs


def get_yle_aligned_articles(json_file):
    # construct test dataset
    articles = {'fi': [], 'sv': []}
    article_ids = {'fi': [], 'sv': []}
    # get FI articles with in aligned dataset
    yle_related = json.load(open(json_file, 'r'))
    for related in yle_related:
        fi_art = related['fi'][0]
        fi_id = fi_art['id']
        fi_text = fi_art['headline'] + '. ' + fi_art['content']
        if fi_id not in article_ids['fi']:
            articles['fi'].append(fi_text)
            article_ids['fi'].append(fi_id)
    # get SV articles in aligned dataset
    for related in yle_related:
        for sv_art in related['sv']:
            sv_id = sv_art['id']
            sv_text = sv_art['headline'] + '. ' + sv_art['content']
            if sv_id not in article_ids['sv']:
                articles['sv'].append(sv_text)
                article_ids['sv'].append(sv_id)
    # sanity check
    for lang in articles:
        print(lang, "articles:", len(articles[lang]))
    return articles, article_ids

