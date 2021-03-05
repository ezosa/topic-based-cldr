from contextualized_topic_models.models.ctm import ZeroShotTM
from contextualized_topic_models.utils.data_preparation import QuickText
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing
# from contextualized_topic_models.datasets.dataset import CTMDataset
# from gensim.corpora.dictionary import Dictionary
# import nltk
import numpy as np
import pickle

from data import get_denews_paired, compute_jsd

import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--test_data', default='data/denews/1998/parsed_articles.json', type=str, help="")
argparser.add_argument('--trained_model', default='results/cldr/ctm.pkl', type=str, help="trained model")
argparser.add_argument('--n_samples', default=50, type=int, help='sampling iterations')
argparser.add_argument('--sparse_vec', default=0, type=int, help='re-normalize doc vectors to make it more sparse')
args = argparser.parse_args()

print("----- Testing ContextualizedTM on DE-News data -----")
print("test_data:", args.test_data)
print("trained_model:", args.trained_model)
print("n_samples:", args.n_samples)
print("sparse_vec:", args.sparse_vec)
print("-"*50)

languages = ['en', 'de']

# loading trained model
print("Loading trained model")
ctm = pickle.load(open(args.trained_model, 'rb'))

# get test docs
print("Loading test data")
paired_docs = get_denews_paired(args.test_data)


def renormalize_vector(vector, thresh=0.01):
    vector[vector < thresh] = 0
    vector = vector/vector.sum()
    return vector

print("Infer doc-topic distributions per language")
doc_vectors = {'en':[], 'de':[]}
for lang in languages:
    print("Lang:", lang.upper())
    # pre-process docs
    print("Pre-process docs")
    if lang == 'en':
        stopwords_lang = 'english'
    else:
        stopwords_lang = 'german'

    sp = WhiteSpacePreprocessing(paired_docs[lang], stopwords_language=stopwords_lang)
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

# rank DE articles according to similarity to EN query article
ranked_articles = []
en_vectors = doc_vectors['en']
de_vectors = doc_vectors['de']
mean_pair_jsd = []
for i, query_vec in enumerate(en_vectors):
    jsd = np.array([compute_jsd(query_vec, candidate_vec) for candidate_vec in de_vectors])
    ranked_indexes = np.argsort(jsd)
    ranked_articles.append(ranked_indexes)
    jsd_pair = compute_jsd(query_vec, de_vectors[i])
    mean_pair_jsd.append(jsd_pair)

mean_pair_jsd = np.mean(mean_pair_jsd)
print("Mean pair JSD:", mean_pair_jsd)

dump_file = args.trained_model[:-4] + "_" + str(args.n_samples) + "samples_sparse" + str(args.sparse_vec) + "_rankings.pkl"
with open(dump_file, 'wb') as f:
    pickle.dump(ranked_articles, f)
    f.close()
    print("Saved clustering rankings as", dump_file, "!")


