import numpy as np
import argparse
import pickle
from gensim.models import KeyedVectors
from data import get_denews_docs_vocab, load_word_embeddings, get_vocab_scores
from clustering import kmeans_clustering

argparser = argparse.ArgumentParser()
argparser.add_argument('--emb_type', default='fasttext', type=str, help="embedding type: fasttext or bert")
argparser.add_argument('--n_clusters', default=100, type=int, help="number of clusters")
argparser.add_argument('--cluster_method', default='kmeans', type=str, help="clustering method to use")
argparser.add_argument('--weight_type', default='tf', type=str, help="weighting scheme: tf or tfidf or none")
#argparser.add_argument('--lang_to_cluster', default='en', type=str, help="language to cluster")
args = argparser.parse_args()

print("-"*5, "Clustering bilingual vocabulary from DE-News", "-"*5)
print("emb_type:", args.emb_type)
print("clusters:", args.n_clusters)
print("method:", args.cluster_method)
print("weight:", args.weight_type)
#print("lang:", args.lang_to_cluster.upper())
print("-"*50)

languages = ['en', 'de']
ft_emb = {'en': '/proj/zosa/data/fasttext/wiki.multi.en.vec',
          'de': '/proj/zosa/data/fasttext/wiki.multi.de.vec'}
bert_emb = {'en': '/proj/zosa/data/denews/1997/parsed_articles_bert_extract_en.pkl',
            'de': '/proj/zosa/data/denews/1997/parsed_articles_bert_extract_de.pkl'}

embeddings = {}
if args.emb_type == 'fasttext':
    # load fasttext word embeddings
    for lang in languages:
        emb = load_word_embeddings(ft_emb[lang])
        embeddings[lang] = emb
        print(lang.upper(), "vocab:", len(emb.vocab))
else:
    # load BERT embeddings
    for lang in languages:
        print("Loading extracted BERT embeddings from", bert_emb[lang])
        emb = pickle.load(open(bert_emb[lang], 'rb'))
        embeddings[lang] = emb
        print(lang.upper(), "vocab:", len(embeddings[lang]))


# load training data articles and vocab
train_filepath = "/proj/zosa/data/denews/1997/parsed_articles.json"
print("Loading train docs and vocab from", train_filepath)
docs, vocab = get_denews_docs_vocab(train_filepath)
valid_vocab = {}
for lang in languages:
    valid_vocab[lang] = [w for w in vocab[lang] if w in embeddings[lang]]

# cluster vocabulary embeddings from 1 language only
print("\nComputing word weights using", args.weight_type)
weight_type = args.weight_type
vocab_scores = {}
for lang in languages:
    scores = get_vocab_scores(docs[lang], valid_vocab[lang], score_type=weight_type)
    vocab_scores[lang] = scores
    words = list(scores.keys())
    valid_vocab[lang] = words
    print("Final", lang.upper(), "vocab:", len(valid_vocab[lang]))

combined_emb = []
combined_weights = []
combined_vocab = []
for lang in languages:
    valid_embs = np.array([embeddings[lang][w] for w in valid_vocab[lang]])
    combined_emb.append(valid_embs)
    w = np.array([vocab_scores[lang][w] for w in valid_vocab[lang]])
    combined_weights.append(w)
    combined_vocab.extend(valid_vocab[lang])

combined_emb = np.vstack(combined_emb)
combined_weights = np.concatenate(combined_weights)
print("Embs to cluster:", combined_emb.shape)
print("Weights:", combined_weights.shape)
print("Combined vocab:", len(combined_vocab))

print("\nClustering vocab words")
if args.cluster_method == 'kmeans':
    if args.weight_type != 'none':
        labels, centers, model = kmeans_clustering(vocab_embeddings=combined_emb, vocab=combined_vocab, topics=args.n_clusters,
                                            weights=combined_weights)
    else:
        labels, centers, model = kmeans_clustering(vocab_embeddings=combined_emb, vocab=combined_vocab, topics=args.n_clusters)
    save_path = "/proj/zosa/results/cldr/"
    dump_file = "denews_" + args.emb_type + "_" + args.cluster_method + "_" + args.weight_type + "_" + str(args.n_clusters) + "clusters" + "_bilingual.pkl"
    dump_file_model = dump_file[:-4] + "_model.pkl"
    results = {'labels': labels, 'centers': centers, 'vocab': vocab_scores}
    with open(save_path + dump_file, 'wb') as f:
        pickle.dump(results, f)
        f.close()
        print("\nSaved clustering results to", save_path + dump_file, "!")
    with open(save_path + dump_file_model, 'wb') as f:
        pickle.dump(model, f)
        f.close()
        print("\nSaved model to", save_path + dump_file_model, "!")





