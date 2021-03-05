import json
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Yle related articles
yle_filepath = "/proj/zosa/data/yle/yle_aligned_articles_2018_3_small.json"
yle_related = json.load(open(yle_filepath, 'r'))
fi_related_ids = [related['fi'][0]['id'] for related in yle_related]
yle_related = None

# SBert encoded sentences of candidate articles
fi_emb_path = "/proj/zosa/data/yle/yle_aligned_articles_2018_3_small_fi_distilbert.pkl"
fi_embeddings = pickle.load(open(fi_emb_path, 'rb'))
fi_ids = list(fi_embeddings.keys())
#fi_embeddings = [fi_embeddings[i][0] for i in fi_art_ids if i in fi_related_ids]
print("FI embeddings:", len(fi_embeddings))

sv_emb_path = "/proj/zosa/data/yle/yle_aligned_articles_2018_3_small_sv_distilbert.pkl"
sv_embeddings = pickle.load(open(sv_emb_path, 'rb'))
sv_ids = list(sv_embeddings.keys())
sv_embeddings = [sv_embeddings[i] for i in sv_ids]
print("SV embeddings:", len(sv_embeddings))

# for each FI query article, rank SV candidate articles
yle_related_ranking = {}
for i,fi_id in enumerate(fi_related_ids):
    #fi_art_id = related['fi'][0]['id']
    print("FI art:", fi_id)
    if fi_id in fi_embeddings:
        fi_art_emb = fi_embeddings[fi_id]
        # compute cosine sim between FI article emb and all SV art embeddings
        similarity = cosine_similarity([fi_art_emb], sv_embeddings)[0]
        # order SV article id's according to similarity
        ranked_index = np.argsort(-similarity)
        ranked_ids = [sv_ids[r] for r in ranked_index]
        yle_related_ranking[fi_id] = ranked_ids

# save article rankings for evaluation
dump_file = 'results/cldr/yle_3_small_distilbert_rankings.json'
with open(dump_file, 'w') as f:
    json.dump(yle_related_ranking, f)
    f.close()
    print("Saved article rankings as", dump_file, "!")

