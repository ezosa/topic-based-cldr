import numpy as np
import argparse
import pickle
from data import get_yle_docs_vocab, get_yle_docs_vocab_mem_friendly, load_word_embeddings, get_vocab_scores, get_tf_scores_mem_friendly
from clustering import kmeans_clustering, GMM_clustering

argparser = argparse.ArgumentParser()
argparser.add_argument('--emb_type', default='fasttext', type=str, help="embedding type: fasttext or bert")
argparser.add_argument('--n_clusters', default=100, type=int, help="number of clusters")
argparser.add_argument('--cluster_method', default='kmeans', type=str, help="clustering method to use")
argparser.add_argument('--cov_type', default='full', type=str, help="covariance type for gmm clustering: spherical, full, diag")
argparser.add_argument('--weight_type', default='tf', type=str, help="weighting scheme: tf or none")
argparser.add_argument('--lang_to_cluster', default='fi', type=str, help="language to cluster")
args = argparser.parse_args()

print("-"*5, "Clustering vocabulary from Yle data", "-"*5)
print("emb_type:", args.emb_type)
print("clusters:", args.n_clusters)
print("method:", args.cluster_method)
print("cov_type:", args.cov_type)
print("weight:", args.weight_type)
print("lang:", args.lang_to_cluster.upper())
print("-"*50)

ft_emb = {'fi': '/proj/zosa/data/fasttext/wiki.multi.fi.vec'}
bert_emb = {'fi': '/proj/zosa/data/yle/yle_articles_2017_bert_extract_fi_large.pkl',
            'sv': '/proj/zosa/data/yle/yle_articles_2017_bert_extract_sv_large.pkl'}

embeddings = {}
if args.emb_type == 'fasttext':
    # load fasttext word embeddings
    embeddings = load_word_embeddings(ft_emb[args.lang_to_cluster])
    print("Embeddings vocab:", len(embeddings.vocab))
else:
    # load BERT embeddings
    print("Loading extracted BERT embeddings from", bert_emb[args.lang_to_cluster])
    embeddings = pickle.load(open(bert_emb[args.lang_to_cluster], 'rb'))
    # languages = ['fi', 'sv']
    # for lang in languages:
    #     print("Loading extracted BERT embeddings from", bert_emb[lang])
    #     embeddings = pickle.load(open(bert_emb[lang], 'rb'))
    #     words = list(embeddings.keys())
    #     print("BERT vocab:", len(words))
    #     for i,word in enumerate(words):
    #         embeddings[word] = np.array(embeddings[word][1:])
    #         if i == 1:
    #             print(word, ":", np.array(embeddings[word][1:]))
    #     dump_file = bert_emb[lang][:-4]+"_large.pkl"
    #     with open(dump_file, 'wb') as f:
    #         pickle.dump(embeddings, f)
    # embeddings = pickle.load(open(bert_emb[args.lang_to_cluster][:-4]+"_large.pkl", 'rb'))

# load training data articles and vocab
train_filepath = "/proj/zosa/data/yle/yle_articles_2017.json"
print("Loading train docs and vocab from", train_filepath)
docs, vocab = get_yle_docs_vocab_mem_friendly(train_filepath, max_docs=50000)
valid_docs = docs[args.lang_to_cluster]
valid_vocab = vocab[args.lang_to_cluster]
valid_vocab = [w for w in valid_vocab if w in embeddings]

# cluster vocabulary embeddings from 1 language only
print("\nComputing word weights using", args.weight_type)
if args.weight_type == 'tf':
    weight_type = args.weight_type
    #vocab_scores = get_vocab_scores(valid_docs, valid_vocab, score_type=weight_type)
    tf_scores = get_tf_scores_mem_friendly(valid_docs)
    valid_vocab = [word for word in valid_vocab if word in embeddings and word in tf_scores]
    weights = np.array([tf_scores[w] for w in valid_vocab])
    emb_to_cluster = np.array([embeddings[w] for w in valid_vocab])
else:
    emb_to_cluster = np.array([embeddings[w] for w in valid_vocab])
print("Valid vocab:", len(valid_vocab))

print("\nClustering vocab words")
if args.cluster_method == 'kmeans':
    if args.weight_type != 'none':
        labels, centers, model = kmeans_clustering(vocab_embeddings=emb_to_cluster, vocab=valid_vocab, topics=args.n_clusters, weights=weights)
    else:
        labels, centers, model = kmeans_clustering(vocab_embeddings=emb_to_cluster, vocab=valid_vocab, topics=args.n_clusters)

elif args.cluster_method == 'gmm':
    labels, means, cov = GMM_clustering(vocab_embeddings=emb_to_cluster, vocab=valid_vocab, cov_type=args.cov_type,
                                        n_topics=args.n_clusters)

else:
    labels, centers, model = kmeans_clustering(vocab_embeddings=emb_to_cluster, vocab=valid_vocab,
                                               topics=args.n_clusters)

# Save clustering results to file
save_path = "/proj/zosa/results/cldr/"

if args.cluster_method != 'gmm':
    dump_file = "yle_" + args.emb_type + "_" + args.cluster_method + "_" + args.weight_type + "_" + \
                str(args.n_clusters) + "clusters" + "_" + args.lang_to_cluster + ".pkl"
else:
    dump_file = "yle_" + args.emb_type + "_" + args.cluster_method + "_" + args.cov_type + "_" + \
                str(args.n_clusters) + "clusters" + "_" + args.lang_to_cluster + ".pkl"

if args.cluster_method != 'gmm':
    results = {'labels': labels, 'centers': centers, 'vocab': valid_vocab}
else:
    results = {'labels': labels, 'means': means, 'covar': cov, 'vocab': valid_vocab}

with open(save_path + dump_file, 'wb') as f:
    pickle.dump(results, f)
    print("\nSaved clustering results to", save_path + dump_file, "!")
    f.close()