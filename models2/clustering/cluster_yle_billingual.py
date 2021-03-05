import numpy as np
import argparse
import pickle
from data import get_yle_docs_vocab, get_yle_docs_vocab_mem_friendly, load_word_embeddings, get_vocab_scores
from clustering import kmeans_clustering

argparser = argparse.ArgumentParser()
argparser.add_argument('--emb_type', default='fasttext', type=str, help="embedding type: fasttext or bert")
argparser.add_argument('--n_clusters', default=100, type=int, help="number of clusters")
argparser.add_argument('--cluster_method', default='kmeans', type=str, help="clustering method to use")
argparser.add_argument('--weight_type', default='none', type=str, help="weighting scheme: tf or none")
args = argparser.parse_args()

print("-"*5, "Clustering bilingual vocabulary from Yle data", "-"*5)
print("emb_type:", args.emb_type)
print("clusters:", args.n_clusters)
print("method:", args.cluster_method)
print("weight:", args.weight_type)
print("-"*50)

languages = ['fi', 'sv']

ft_emb = {'fi': '/proj/zosa/data/fasttext/wiki.multi.fi.vec',
          'sv': '/proj/zosa/data/fasttext/wiki.multi.sv.vec'}
bert_emb = {'fi': '/proj/zosa/data/yle/yle_articles_2017_bert_extract_fi_large.pkl',
            'sv': '/proj/zosa/data/yle/yle_articles_2017_bert_extract_sv_large.pkl'}

embeddings = {}
if args.emb_type == 'fasttext':
    # load fasttext word embeddings
    for lang in languages:
        embeddings[lang] = load_word_embeddings(ft_emb[lang])
        print(lang.upper(), " vocab:", len(embeddings[lang].vocab))
else:
    # load BERT embeddings
    for lang in languages:
        print("Loading extracted BERT embeddings from", bert_emb[lang])
        embeddings[lang] = pickle.load(open(bert_emb[lang], 'rb'))
        print(lang.upper(), "vocab:", len(embeddings[lang]))
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


# load training data articles and vocab
train_filepath = "/proj/zosa/data/yle/yle_articles_2017.json"
print("Loading train docs and vocab from", train_filepath)
docs, vocab = get_yle_docs_vocab_mem_friendly(train_filepath, max_docs=50000)
valid_vocab = {}
for lang in languages:
    vocab_lang = [w for w in vocab[lang] if w in embeddings[lang]]
    valid_vocab[lang] = vocab_lang

# cluster embeddings from two vocabularies
valid_embeddings = {}
for lang in languages:
    if args.weight_type != 'none':
        print("\nComputing word weights using", args.weight_type)
        weight_type = args.weight_type
        vocab_scores = get_vocab_scores(docs[lang], valid_vocab[lang], score_type=weight_type)
        valid_vocab = list(vocab_scores.keys())
        weights = np.array([vocab_scores[w] for w in valid_vocab])
        emb_to_cluster = np.array([embeddings[w] for w in valid_vocab])
        valid_embeddings[lang] = emb_to_cluster
    else:
        valid_embeddings[lang] = np.array([embeddings[lang][word] for word in valid_vocab[lang]])
    print(lang.upper(), " valid vocab:", len(valid_vocab[lang]))

print("\nClustering vocab words")
embeddings_arr = np.concatenate((valid_embeddings['fi'], valid_embeddings['sv']), axis=0)
print("Concatenated embeddings:", embeddings_arr.shape)
vocab_arr = valid_vocab['fi'] + valid_vocab['sv']
print("Concatenated vocab:", len(vocab_arr))
if args.cluster_method == 'kmeans':
    if args.weight_type != 'none':
        labels, centers, model = kmeans_clustering(vocab_embeddings=embeddings_arr, vocab=vocab_arr, topics=args.n_clusters, weights=weights)
    else:
        labels, centers, model = kmeans_clustering(vocab_embeddings=embeddings_arr, vocab=vocab_arr, topics=args.n_clusters)
    save_path = "/proj/zosa/results/cldr/"
    dump_file = "yle_" + args.emb_type + "_" + args.cluster_method + "_" + args.weight_type + "_" + str(args.n_clusters) + "clusters" + "_bilingual.pkl"
    results = {'labels': labels, 'centers': centers, 'vocab': valid_vocab}
    with open(save_path + dump_file, 'wb') as f:
        pickle.dump(results, f)
        f.close()
        print("\nSaved clustering results to", save_path + dump_file, "!")





