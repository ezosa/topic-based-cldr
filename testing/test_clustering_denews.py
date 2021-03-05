import json
import numpy as np
import pickle
import scipy.stats
from collections import Counter
from numpy import dot
from numpy.linalg import norm
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax

from data import load_word_embeddings, compute_jsd


import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--emb_type', default='fasttext', type=str, help="embedding type: fasttext or bert")
argparser.add_argument('--cluster_file', default='./', type=str, help="clustering results file")
argparser.add_argument('--cluster_method', default='kmeans', type=str, help="clustering method: kmeans, affprop, gmm")
argparser.add_argument('--model_file', default=None, type=str, help="path to trained clustering model")
argparser.add_argument('--doc_rep', default='topic', type=str, help="doc dist computation: doc, token, topic")
argparser.add_argument('--dist_metric', default='jsd', type=str, help="distance metric: jsd, kld or cosine")
argparser.add_argument('--sparse_vec', default=0, type=int, help="make vectors more sparse")
args = argparser.parse_args()

print("-"*5, "Cross-lingual linking of DE-News with Topic clusters", "-"*5)
print("emb_type:", args.emb_type)
print("clustering:", args.cluster_file)
print("cluster_method:", args.cluster_method)
print("doc representation:", args.doc_rep)
print("dist metric:", args.dist_metric)
print("-"*50)


languages = ['en', 'de']
# load clustering results
result_file = args.cluster_file
data = pickle.load(open(result_file, 'rb'))

if args.cluster_method != 'gmm':
    labels = data['labels']
    centers = data['centers']
    valid_vocab = list(data['vocab'].keys())
    n_clusters = centers.shape[0]
else:
    labels = data['labels']
    centers = data['means']
    covar = data['covar']
    valid_vocab = list(data['vocab'].keys())
    n_clusters = centers.shape[0]

print("Clusters/topics:", n_clusters)



# open embeddings
embeddings = {}
for lang in languages:
    if args.emb_type == 'fasttext':
        emb_file = "/proj/zosa/data/fasttext/wiki.multi." + lang + ".vec"
        emb = load_word_embeddings(emb_file)
        embeddings[lang] = emb
    else:
        emb_file = "/proj/zosa/data/denews/1997/parsed_articles_bert_extract_" + lang +".pkl"
        emb = pickle.load(open(emb_file, 'rb'))
        # for word in emb:
        #     emb[word] = emb[word][1:]
        embeddings[lang] = emb


# open test docs
test_file = "/proj/zosa/data/denews/1998/parsed_articles.json"
docs = json.load(open(test_file, 'r'))

art_ids = list(docs['en'].keys())
valid_docs = {'en': [], 'de': []}
for art_id in art_ids:
    if art_id in docs['en'] and art_id in docs['de']:
        for lang in languages:
            text = docs[lang][art_id]['headline'] + " " + docs[lang][art_id]['content']
            tokens = text.lower().split()
            valid_tokens = [word for word in tokens if word in embeddings[lang]]
            valid_docs[lang].append(valid_tokens)

# sanity check - must have same no. of articles
print("EN articles:", len(valid_docs['en']))
print("DE articles:", len(valid_docs['de']))


def renormalize_vector(vector, thresh=0.01):
    vector[vector < thresh] = 0
    vector = vector/vector.sum()
    return vector


def assign_topic_to_word(word_vector):
    dist = [cosine_similarity([word_vector], [centers[k]])[0][0] for k in range(n_clusters)]
    topic = np.argmax(dist)
    return topic


def assign_topic_to_word_vectorized(word_vector):
    #cos_sim = cosine_similarity([q], mat)[0]
    cos_sim = dot(word_vector, centers.T) / (norm(word_vector) * norm(centers, axis=1))
    topic = np.argmax(cos_sim)
    return topic


def get_topics_for_doc_gmm(doc_emb):
    density = np.array([scipy.stats.multivariate_normal(cov=covar[k], mean=centers[k]).logpdf(doc_emb) for k in range(n_clusters)])
    topics = [np.argmax(density.T[i]) for i in range(density.T.shape[0])]
    return topics

# def assign_topic_to_word_with_model(word_vector, model):
#     label = model.predict([word_vector])[0]
#     return label
#
#
# def assign_topic_to_word_from_cluster_labels(word_vector, vocab, labels):
#     label = model.predict([word_vector])[0]
#     return label



# compute distance from word embeddings to cluster centers
doc_vectors = {'en': [], 'de': []}
for lang in languages:
    docs = valid_docs[lang]
    for i, doc in enumerate(docs):
        #print("Doc", i+1, "of", len(docs))
        # if args.doc_rep == 'token':
        #     #print("Compute doc distance by", args.doc_dist)
        #     #doc_dist = [[distance.cosine(embeddings[lang][word], centers[k]) for k in range(n_clusters)] for word in doc]
        #     #doc_dist = np.mean(np.array(doc_dist), axis=0)
        #     doc_sim = [[cosine_similarity([embeddings[lang][word]], [centers[k]])[0][0] for k in range(n_clusters)] for word in doc]
        #     doc_sim = np.mean(np.array(doc_sim), axis=0)
        # elif args.doc_rep == 'doc':
        #     doc_emb = np.array([embeddings[lang][word] for word in doc])
        #     doc_emb = np.mean(doc_emb, axis = 0)
        #     doc_sim = np.array([cosine_similarity([doc_emb], [centers[k]])[0][0] for k in range(n_clusters)])
        # else:
        if args.cluster_method != 'gmm':
            if args.emb_type != 'bert':
                doc_topics = [assign_topic_to_word_vectorized(embeddings[lang][word]) for word in doc]
            else:
                doc_topics = [assign_topic_to_word(embeddings[lang][word]) for word in doc]
        else:
            doc_emb = np.array([embeddings[lang][word] for word in doc])
            doc_topics = get_topics_for_doc_gmm(doc_emb)
        topic_counts = Counter(doc_topics)
        doc_prop = np.array([topic_counts[k] for k in range(n_clusters)])
        doc_sim = doc_prop/doc_prop.sum()
        # if args.dist_metric in ['jsd', 'kld'] and args.doc_rep == 'token':
        #     doc_vec = softmax(doc_sim)
        #     thresh = 1/n_clusters
        #     doc_vec = renormalize_vector(doc_vec, thresh)
        # else:
        doc_vec = doc_sim
        doc_vectors[lang].append(doc_vec)

# rank DE articles according to similarity to EN query article
ranked_articles = []
en_vectors = doc_vectors['en']
de_vectors = doc_vectors['de']
for query_vec in en_vectors:
    if args.dist_metric in ['jsd', 'kld']:
        jsd = np.array([compute_jsd(query_vec, candidate_vec) for candidate_vec in de_vectors])
        ranked_indexes = np.argsort(jsd)
    else:
        sim = np.array([cosine_similarity([query_vec], [candidate_vec])[0][0] for candidate_vec in de_vectors])
        ranked_indexes = np.argsort(-sim)
    ranked_articles.append(ranked_indexes)

# print("Articles rankings:")
# for i, ranks in enumerate(ranked_articles):
#     print("Article", i, ":", ranks)

dump_file = result_file[:-4] + "_" + args.doc_rep + "_" + args.dist_metric + "_rankings.pkl"
with open(dump_file, 'wb') as f:
    pickle.dump(ranked_articles, f)
    f.close()
    print("Saved clustering rankings as", dump_file, "!")








