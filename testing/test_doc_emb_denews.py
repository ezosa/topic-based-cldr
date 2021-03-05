import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--doc_emb', default='./', type=str, help="path to doc embeddings")
args = argparser.parse_args()

print("-"*5, "Test cross-lingual linking for DE-News articles ", "-"*5)
print("doc_emb:", args.doc_emb)
print("-"*50)
# load DE-News doc embeddings
doc_embeddings = pickle.load(open(args.doc_emb, 'rb'))
print("Loaded doc embeddings from", args.doc_emb)

languages = ['en', 'de']
en_embeddings = doc_embeddings['en']
de_embeddings = doc_embeddings['de']

print("EN emb:", len(en_embeddings))
print("DE emb:", len(de_embeddings))

# for each EN query article, rank candidate DE articles
de_embeddings = [de_embeddings[i] for i in range(len(de_embeddings))]
ranked_articles = []
for i in range(len(en_embeddings)):
    similarity = cosine_similarity([en_embeddings[i]], de_embeddings)[0]
    # rank candidate articles according to similarity
    ranked_indexes = np.argsort(-similarity)
    ranked_articles.append(ranked_indexes)

dump_file = "results/cldr/denews_doc_emb_" + args.doc_emb.split("_")[-2] + "_rankings.pkl"
with open(dump_file, 'wb') as f:
    pickle.dump(ranked_articles, f)
    f.close()
    print("Saved DE-News doc emb rankings as", dump_file, "!")


