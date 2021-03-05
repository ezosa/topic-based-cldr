import json
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--doc_emb', default='./', type=str, help="path to doc embeddings")
args = argparser.parse_args()

# Doc embeddings of query FI articles and candidate SV articles
yle_path = "/proj/zosa/data/yle/"
yle_doc_emb = args.doc_emb
doc_embeddings = pickle.load(open(yle_path + yle_doc_emb, 'rb'))
fi_embeddings = doc_embeddings['fi']
sv_embeddings = doc_embeddings['sv']
sv_ids = list(sv_embeddings.keys())
sv_embeddings = [sv_embeddings[sv_id] for sv_id in sv_ids]
print("FI embeddings:", len(fi_embeddings))
print("SV embeddings:", len(sv_embeddings))

# for each FI query article, rank SV candidate articles
yle_related_ranking = {}
for fi_id in fi_embeddings:
    fi_emb = fi_embeddings[fi_id]
    # compute cosine sim between FI article emb and all SV art embeddings
    similarity = cosine_similarity([fi_emb], sv_embeddings)[0]
    # order SV article ids according to similarity
    ranked_index = np.argsort(-similarity)
    ranked_ids = [sv_ids[r] for r in ranked_index]
    yle_related_ranking[fi_id] = ranked_ids

# save article rankings for evaluation
dump_file = 'results/cldr/' + yle_doc_emb[:-4] + "_rankings.pkl"
with open(dump_file, 'w') as f:
    json.dump(yle_related_ranking, f)
    f.close()
    print("Saved article rankings as", dump_file, "!")

