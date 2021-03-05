from contextualized_topic_models.models.ctm import ZeroShotTM
from contextualized_topic_models.utils.data_preparation import QuickText
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing
# from contextualized_topic_models.datasets.dataset import CTMDataset
# from gensim.corpora.dictionary import Dictionary
# import nltk
import numpy as np
import pickle
import json

from data import get_yle_aligned_articles, compute_jsd

import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--test_data', default='data/yle/yle_aligned_articles_2018_3_small.json', type=str, help="test articles")
argparser.add_argument('--trained_model', default='results/cldr/ctm.pkl', type=str, help="trained model")
argparser.add_argument('--n_samples', default=50, type=int, help='sampling iterations')
argparser.add_argument('--sparse_vec', default=0, type=int, help='re-normalize doc vectors to make it more sparse')
args = argparser.parse_args()

print("----- Testing ContextualizedTM on Yle data -----")
print("test_data:", args.test_data)
print("trained_model:", args.trained_model)
print("n_samples:", args.n_samples)
print("sparse_vec:", args.sparse_vec)
print("-"*50)

languages = ['fi', 'sv']

# loading trained model
print("Loading trained model")
ctm = pickle.load(open(args.trained_model, 'rb'))

# get test docs
print("Loading test data")
articles, article_ids = get_yle_aligned_articles(args.test_data)


def renormalize_vector(vector, thresh=0.01):
    vector[vector < thresh] = 0
    vector = vector/vector.sum()
    return vector

print("Infer doc-topic distributions per language")
doc_vectors = {'fi':[], 'sv':[]}
doc_ids = {'fi':[], 'sv':[]}
for lang in languages:
    print("-"*5, "Lang:", lang.upper(), "-"*5)
    # pre-process docs
    print("Pre-process docs")
    if lang == 'fi':
        stopwords_lang = 'finnish'
    else:
        stopwords_lang = 'swedish'

    sp = WhiteSpacePreprocessing(articles[lang], stopwords_language=stopwords_lang)
    preprocessed_documents, unpreprocessed_corpus, vocab = sp.preprocess()

    # get BERT embeddings of docs
    print("Get BERT encoding")
    qt = QuickText("distiluse-base-multilingual-cased",
                    text_for_bow=preprocessed_documents,
                    text_for_bert=unpreprocessed_corpus)

    test_dataset = qt.load_dataset()

    thetas = ctm.get_thetas(test_dataset, n_samples=args.n_samples)
    print("thetas:", thetas.shape)
    if args.sparse_vec:
        thetas = [renormalize_vector(vec, thresh=0.01) for vec in thetas]
    doc_vectors[lang] = thetas
    doc_ids[lang] = article_ids[lang]

# for each FI query article, rank similarity to all candidate SV articles
yle_related_ranking = {}
query_vectors = doc_vectors['fi']
cand_vectors = doc_vectors['sv']
query_ids = doc_ids['fi']
cand_ids = doc_ids['sv']

for i, query_vec in enumerate(query_vectors):
    jsd = np.array([compute_jsd(query_vec, candidate_vec) for candidate_vec in cand_vectors])
    ranked_indexes = np.argsort(jsd)
    query_id = query_ids[i]
    ranked_ids = [cand_ids[r] for r in ranked_indexes]
    yle_related_ranking[query_id] = ranked_ids


dump_file = args.trained_model[:-4] + "_rankings.json"
with open(dump_file, 'w') as f:
    json.dump(yle_related_ranking, f)
    f.close()
    print("Saved clustering rankings as", dump_file, "!")

