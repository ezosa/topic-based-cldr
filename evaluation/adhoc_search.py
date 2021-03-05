import pickle
import json
import numpy as np
import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--doc_rankings', default='./', type=str, help="doc rankings file")
args = argparser.parse_args()


def precision_at_k(predicted, actual, k):
    pred_k = predicted[:k]
    tp = set(pred_k).intersection(set(actual))
    prec = len(tp) / len(pred_k)
    return prec

# open related articles dataset
related_path = "/proj/zosa/data/yle/yle_aligned_articles_2018_3_small.json"
related_data = json.load(open(related_path, 'r'))
related_ids = {}
related_counts = []
for related in related_data:
    fi_id = related['fi'][0]['id']
    sv_ids = [sv_art['id'] for sv_art in related['sv']]
    related_ids[fi_id] = sv_ids
    related_counts.append(len(sv_ids))
related_data = None
print("Related dataset:", len(related_counts))
print("Average related:", np.mean(related_counts))

# open predicted rankings file
results_path = "/proj/zosa/results/cldr/"
rankings_file = args.doc_rankings
pred_rankings = json.load(open(results_path + rankings_file, 'r'))
pred_ids = list(pred_rankings.keys())
print("Pred rankings:", len(pred_rankings))

# evaluation metrics
prec_1 = []
prec_5 = []
prec_10 = []
mrr = []
for fi_id in pred_ids:
    if fi_id in related_ids:
        true_related = related_ids[fi_id]
        predicted = pred_rankings[fi_id]
        p1 = precision_at_k(predicted, true_related, k=1)
        p5 = precision_at_k(predicted, true_related, k=5)
        p10 = precision_at_k(predicted, true_related, k=10)
        prec_1.append(p1)
        prec_5.append(p5)
        prec_10.append(p10)
        pred = np.array(predicted)
        reciprocal_ranks = []
        for sv_id in true_related:
            rank = np.where(pred==sv_id)[0]
            if len(rank) > 0:
                rr = 1/(rank[0]+1)
                reciprocal_ranks.append(rr)
        reciprocal_ranks = np.mean(reciprocal_ranks)
        mrr.append(reciprocal_ranks)
        #print(fi_id, ":", reciprocal_ranks)

mrr = np.mean(mrr)
prec_1 = np.mean(prec_1)
prec_5 = np.mean(prec_5)
prec_10 = np.mean(prec_10)
print("MRR:", mrr)
print("Precision@1:", prec_1)
print("Precision@5:", prec_5)
print("Precision@10:", prec_10)