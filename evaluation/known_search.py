import pickle
import numpy as np
import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--doc_rankings', default='./', type=str, help="doc rankings file")
args = argparser.parse_args()

results_path = "/proj/zosa/results/cldr/"
rankings_file = args.doc_rankings #"denews_fasttext_kmeans_tf_50clusters_en_doc_emb_rankings.pkl"
rankings = pickle.load(open(results_path + rankings_file, 'rb'))
rankings = [list(r) for r in rankings]
n_articles = len(rankings)

# evaluation metrics
accuracy = np.array([1 if ranking[0]==i else 0 for i, ranking in enumerate(rankings)])
mrr = np.array([1/((ranking.index(i))+1) for i, ranking in enumerate(rankings)])
# accuracy = [0] * n_articles
# mrr = [0] * n_articles
# for cur_index, ranking in enumerate(rankings):
#     # check if the correct article is in rank 1
#     if cur_index == ranking[0]:
#         accuracy[cur_index] = 1
#     # get reciprocal rank of the correct article
#     rec_rank = 1/(list(ranking).index(cur_index)+1)
#     mrr[cur_index] = rec_rank

print("Doc rankings from:", args.doc_rankings)
accuracy = np.round(np.mean(accuracy), 3)
print("Accuracy:", accuracy)

mrr = np.round(np.mean(mrr), 3)
print("MRR:", mrr)