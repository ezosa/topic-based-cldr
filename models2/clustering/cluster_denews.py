import numpy as np
import argparse
import pickle
from data import get_denews_docs_vocab, load_word_embeddings, get_vocab_scores
from clustering import kmeans_clustering, affprop_clustering, GMM_clustering

argparser = argparse.ArgumentParser()
argparser.add_argument('--emb_type', default='fasttext', type=str, help="embedding type: fasttext or bert")
argparser.add_argument('--n_clusters', default=100, type=int, help="number of clusters")
argparser.add_argument('--cluster_method', default='kmeans', type=str, help="methods: kmeans, affprop, gmm")
argparser.add_argument('--cov_type', default='full', type=str, help="covariance type for gmm clustering: spherical, full, diag")
argparser.add_argument('--weight_type', default='tf', type=str, help="weighting scheme: tf or none")
argparser.add_argument('--lang_to_cluster', default='en', type=str, help="language to cluster")
args = argparser.parse_args()

print("-"*5, "Clustering vocabulary from DE-News", "-"*5)
print("emb_type:", args.emb_type)
print("clusters:", args.n_clusters)
print("method:", args.cluster_method)
print("cov_type:", args.cov_type)
print("weight_type:", args.weight_type)
print("lang:", args.lang_to_cluster.upper())
print("-"*40)

ft_emb = {'en': '/proj/zosa/data/fasttext/wiki.multi.en.vec'}
bert_emb = {'en': '/proj/zosa/data/denews/1997/parsed_articles_bert_extract_en.pkl',
            'de': '/proj/zosa/data/denews/1997/parsed_articles_bert_extract_de.pkl'}

embeddings = {}
if args.emb_type == 'fasttext':
    # load fasttext word embeddings
    embeddings = load_word_embeddings(ft_emb[args.lang_to_cluster])
    print("Embeddings vocab:", len(embeddings.vocab))
else:
    # load BERT embeddings
    print("Loading extracted BERT embeddings from", bert_emb[args.lang_to_cluster])
    embeddings = pickle.load(open(bert_emb[args.lang_to_cluster], 'rb'))
    print("Embeddings vocab:", len(embeddings))
    # for word in embeddings:
    #     x = embeddings[word]
    #     x = np.asarray(x, dtype='float64')
    #     embeddings[word] = x


# load training data articles and vocab
train_filepath = "/proj/zosa/data/denews/1997/parsed_articles.json"
print("\nLoading training docs and vocab from", train_filepath)
docs, vocab = get_denews_docs_vocab(train_filepath)
valid_docs = docs[args.lang_to_cluster]
valid_vocab = vocab[args.lang_to_cluster]
valid_vocab = [w for w in valid_vocab if w in embeddings]

# cluster vocabulary embeddings from 1 language only
print("\nComputing word weights using", args.weight_type)
weight_type = args.weight_type
vocab_scores = get_vocab_scores(valid_docs, valid_vocab, score_type=weight_type)
valid_vocab = list(vocab_scores.keys())
emb_to_cluster = np.array([embeddings[w] for w in valid_vocab])
weights = np.array([vocab_scores[w] for w in valid_vocab])
print("Valid vocab:", len(valid_vocab))

print("\nClustering vocab words")
if args.cluster_method == 'kmeans':
    if args.weight_type != 'none':
        labels, centers, model = kmeans_clustering(vocab_embeddings=emb_to_cluster, vocab=valid_vocab, n_topics=args.n_clusters, weights=weights)
    else:
        labels, centers, model = kmeans_clustering(vocab_embeddings=emb_to_cluster, vocab=valid_vocab, n_topics=args.n_clusters)

elif args.cluster_method == 'affprop':
    if args.weight_type != 'none':
        labels, centers = affprop_clustering(vocab_embeddings=emb_to_cluster, vocab=valid_vocab, weights=weights)
    else:
        labels, centers = affprop_clustering(vocab_embeddings=emb_to_cluster, vocab=valid_vocab)

elif args.cluster_method == 'gmm':
    labels, means, cov = GMM_clustering(vocab_embeddings=emb_to_cluster, vocab=valid_vocab, cov_type=args.cov_type,
                                        n_topics=args.n_clusters)

else:
    labels, centers, model = kmeans_clustering(vocab_embeddings=emb_to_cluster, vocab=valid_vocab,
                                               topics=args.n_clusters)

print("Saving clustering results")
save_path = "/proj/zosa/results/cldr/"
if args.cluster_method != 'gmm':
    dump_file = "denews_" + args.emb_type + "_" + args.cluster_method + "_" + args.weight_type + "_" + str(args.n_clusters) + "clusters" + "_" \
            + args.lang_to_cluster + ".pkl"
else:
    dump_file = "denews_" + args.emb_type + "_" + args.cluster_method + "_" + args.cov_type + "_" + args.weight_type + \
                "_" + str(args.n_clusters) + "clusters" + "_" + args.lang_to_cluster + ".pkl"

if args.cluster_method != 'gmm':
    results = {'labels': labels, 'centers': centers, 'vocab': vocab_scores}
else:
    results = {'labels': labels, 'means': means, 'covar': cov, 'vocab': vocab_scores}

with open(save_path + dump_file, 'wb') as f:
    pickle.dump(results, f)
    print("\nSaved clustering results to", save_path + dump_file, "!")
    f.close()
# with open(save_path + model_file, 'wb') as f:
#     pickle.dump(model, f)
#     print("\nSaved trained model to", save_path + model_file, "!")
#     f.close()





