import pickle
import json
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax
from numpy import dot
from numpy.linalg import norm
import numpy as np
import pickle
from collections import Counter
import scipy.stats

from data import load_word_embeddings, compute_jsd

import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--emb_type', default='bert', type=str, help="embedding type: fasttext or bert")
argparser.add_argument('--cluster_file', default='./', type=str, help="clustering results file")
argparser.add_argument('--cluster_method', default='kmeans', type=str, help="clustering method: kmeans, affprop, gmm")
argparser.add_argument('--doc_rep', default='topic', type=str, help="doc dist computation: doc, token or topic")
argparser.add_argument('--dist_metric', default='jsd', type=str, help="distance metric: jsd, kld or cosine")
args = argparser.parse_args()

print("-"*5, "Cross-lingual linking of Yle articles with Topic clusters", "-"*5)
print("emb_type:", args.emb_type)
print("clustering:", args.cluster_file)
print("cluster_method:", args.cluster_method)
print("doc representation:", args.doc_rep)
print("dist metric:", args.dist_metric)
print("-"*50)


languages = ['fi', 'sv']
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
    valid_vocab = data['vocab']
    n_clusters = centers.shape[0]

print("Clusters/topics:", n_clusters)

# open embeddings
print("Loading word embeddings")
embeddings = {}
for lang in languages:
    print("Lang:", lang.upper())
    if args.emb_type == 'fasttext':
        emb_file = "/proj/zosa/data/fasttext/wiki.multi." + lang + ".vec"
        emb = load_word_embeddings(emb_file)
        embeddings[lang] = emb
    else:
        emb_file = "/proj/zosa/data/yle/yle_articles_2017_bert_extract_" + lang + "_large.pkl"
        emb = pickle.load(open(emb_file, 'rb'))
        embeddings[lang] = emb
        print("Vocab:", len(embeddings[lang]))


# construct test dataset
articles = {'fi': [], 'sv': []}
article_ids = {'fi': [], 'sv': []}

# get FI articles with in aligned dataset
yle_filepath = "/proj/zosa/data/yle/yle_aligned_articles_2018_3_small.json"
yle_related = json.load(open(yle_filepath, 'r'))
for related in yle_related:
    fi_art = related['fi'][0]
    fi_id = fi_art['id']
    fi_text = fi_art['headline'] + ' ' + fi_art['content']
    fi_tokens = fi_text.lower().split()
    fi_text = [word for word in fi_tokens if word in embeddings['fi']]
    if fi_id not in article_ids['fi'] and len(fi_text) > 0:
        articles['fi'].append(fi_text)
        article_ids['fi'].append(fi_id)

# get SV articles in aligned dataset
for related in yle_related:
    for sv_art in related['sv']:
        sv_id = sv_art['id']
        sv_text = sv_art['headline'] + ' ' + sv_art['content']
        sv_tokens = sv_text.lower().split()
        sv_text = [word for word in sv_tokens if word in embeddings['sv']]
        if sv_id not in article_ids['sv'] and len(sv_text) > 0:
            articles['sv'].append(sv_text)
            article_ids['sv'].append(sv_id)


# # get all SV 2018 articles
# yle_filepath = "/proj/zosa/data/yle/yle_articles_2018.json"
# yle_articles = json.load(open(yle_filepath, 'r'))
# yle_sv = yle_articles['sv']
# for art in yle_sv:
#     sv_id = art['id']
#     sv_text = art['headline'] + ' ' + art['content']
#     sv_tokens = sv_text.lower().split()
#     sv_text = [word for word in sv_tokens if word in embeddings['sv']]
#     if sv_id not in article_ids['sv']:
#         articles['sv'].append(sv_text)
#         article_ids['sv'].append(sv_id)

print("FI articles:", len(articles['fi']))
print("SV articles:", len(articles['sv']))


def renormalize_vector(vector, thresh=0.1):
    vector[vector < thresh] = 0
    vector = vector/vector.sum()
    return vector


def assign_topic_to_word(word_vector):
    sims = cosine_similarity([word_vector], centers)
    topic = np.argmax(sims)
    return topic


def assign_topic_to_word_vectorized(word_vector):
    #cos_sim = cosine_similarity([q], mat)[0]
    cos_sim = dot(word_vector, centers.T) / (norm(word_vector) * norm(centers, axis=1))
    topic = np.argmax(cos_sim)
    return topic


def get_topics_for_doc(doc_emb):
    cos_sim = cosine_similarity(doc_emb, centers)
    topics = [np.argmax(cos_sim[i]) for i in range(cos_sim.shape[0])]
    return topics

def get_topics_for_doc_gmm(doc_emb):
    density = np.array([scipy.stats.multivariate_normal(cov=covar[k], mean=centers[k]).logpdf(doc_emb) for k in range(n_clusters)])
    topics = [np.argmax(density.T[i]) for i in range(density.T.shape[0])]
    return topics

# get doc-vector representation of each article
doc_vectors = {'fi': [], 'sv': []}
doc_ids = {'fi':[], 'sv':[]}
for lang in languages:
    docs = articles[lang]
    for i, doc in enumerate(docs):
        print(lang.upper(), "doc", i+1, "of", len(docs))
        # if args.doc_rep == 'token':
        #     doc_sim = [[cosine_similarity([embeddings[lang][word]], [centers[k]])[0][0] for k in range(n_clusters)] for word in doc]
        #     doc_sim = np.mean(np.array(doc_sim), axis=0)
        # elif args.doc_rep == 'doc':
        #     doc_emb = np.array([embeddings[lang][word] for word in doc])
        #     doc_emb = np.mean(doc_emb, axis=0)
        #     doc_sim = np.array([cosine_similarity([doc_emb], [centers[k]])[0][0] for k in range(n_clusters)])
        # else:
        doc_emb_mat = np.array([embeddings[lang][word].astype('float64') for word in doc])
        if args.cluster_method != 'gmm':
            doc_topics = get_topics_for_doc(doc_emb_mat)
        else:
            doc_topics = get_topics_for_doc_gmm(doc_emb_mat)
        topic_counts = Counter(doc_topics)
        doc_prop = np.array([topic_counts[k] for k in range(n_clusters)])
        doc_vec = doc_prop/doc_prop.sum()
        # if args.dist_metric in ['jsd', 'kld'] and args.doc_rep == 'token':
        #     doc_vec = softmax(doc_sim)
        #     # thresh = 1/n_clusters
        #     # doc_vec = renormalize_vector(doc_vec, thresh)
        # else:
        # doc_vec = doc_sim
        doc_id = article_ids[lang][i]
        doc_vectors[lang].append(doc_vec)
        doc_ids[lang].append(doc_id)

# for each FI query article, rank similarity to all candidate SV articles
yle_related_ranking = {}
query_vectors = doc_vectors['fi']
candidate_vectors = doc_vectors['sv']
query_ids = doc_ids['fi']
candidate_ids = doc_ids['sv']

for i, query_vec in enumerate(query_vectors):
    if args.dist_metric in ['jsd', 'kld']:
        jsd = np.array([compute_jsd(query_vec, candidate_vec) for candidate_vec in candidate_vectors])
        ranked_indexes = np.argsort(jsd)
    else:
        sim = np.array([cosine_similarity([query_vec], [candidate_vec])[0][0] for candidate_vec in candidate_vectors])
        ranked_indexes = np.argsort(-sim)
    query_id = query_ids[i]
    ranked_ids = [candidate_ids[r] for r in ranked_indexes]
    yle_related_ranking[query_id] = ranked_ids


dump_file = result_file[:-4] + "_" + args.doc_rep + "_" + args.dist_metric + "_rankings.json"
with open(dump_file, 'w') as f:
    json.dump(yle_related_ranking, f)
    f.close()
    print("Saved clustering rankings as", dump_file, "!")








