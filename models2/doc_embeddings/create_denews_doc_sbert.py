import json
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from data import get_denews_docs_paired

model = SentenceTransformer('xlm-r-distilroberta-base-paraphrase-v1')

denews_filepath = "/proj/zosa/data/denews/1998/parsed_articles.json"
paired_docs = get_denews_docs_paired(denews_filepath)

languages = ['en', 'de']
doc_embeddings = {'en':[], 'de':[]}
# encode articles SBERT embeddings
print("Encoding sentences with SentenceTransformer")
for lang in languages:
    print("Lang:", lang.upper())
    docs = paired_docs[lang]
    for doc in docs:
        encoded_doc = model.encode(doc)
        doc_embeddings[lang].append(encoded_doc)

# sanity check
for lang in languages:
    print(lang.upper(), "encoded docs:", len(doc_embeddings[lang]))
# save embeddings
dump_file = denews_filepath[:-5] + "_distilbert.pkl"
with open(dump_file, 'wb') as f:
    pickle.dump(doc_embeddings, f)
    f.close()
    print("Saved BERT doc embeddings as", dump_file, "!")


