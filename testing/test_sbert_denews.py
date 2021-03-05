import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# load DE-News BERT doc embeddings
denews_emb_file = "/proj/zosa/data/denews/1998/parsed_articles_distilbert.pkl"
doc_embeddings = pickle.load(open(denews_emb_file, 'rb'))
print("Loaded SBERT embeddings from", denews_emb_file)

languages = ['en', 'de']
en_embeddings = doc_embeddings['en']
de_embeddings = doc_embeddings['de']

print("EN emb:", len(en_embeddings))
print("DE emb:", len(de_embeddings))

# rank DE articles
de_embeddings = np.array([de_embeddings[i][0] for i in range(len(de_embeddings))])
ranked_articles = []
for i in range(len(en_embeddings)):
    similarity = cosine_similarity(en_embeddings[i], de_embeddings)[0]
    # rank SV articles according to similarity
    ranked_indexes = np.argsort(-similarity)
    ranked_articles.append(ranked_indexes)

dump_file = "results/cldr/denews_rankings_distilbert.pkl"
with open(dump_file, 'wb') as f:
    pickle.dump(ranked_articles, f)
    f.close()
    print("Saved DE-News SBERT rankings as", dump_file, "!")


