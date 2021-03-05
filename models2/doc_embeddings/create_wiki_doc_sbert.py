import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# model = SentenceTransformer('xlm-r-distilroberta-base-paraphrase-v1')
model = SentenceTransformer('distiluse-base-multilingual-cased-v2')


languages = ['fi', 'sv']

# create Wiki test data
wiki_file = "/proj/zosa/data/wiki/wikialign_clean_fi-sv_test.pkl"
wiki_data = pickle.load(open(wiki_file, 'rb'))

# wiki SBERT embeddings
doc_embeddings = {'fi':[], 'sv':[]}
for art_pair in wiki_data:
    for lang in languages:
        article = art_pair[lang].lower()
        encoded_art = model.encode(article)
        doc_embeddings[lang].append(encoded_art)


# sanity check
print("FI doc embeddings:", len(doc_embeddings['fi']))
print("SV doc embeddings:", len(doc_embeddings['sv']))
for i in range(len(doc_embeddings['fi'])):
    doc_fi = doc_embeddings['fi'][i]
    doc_sv = doc_embeddings['sv'][i]
    cos_sim = cosine_similarity(doc_fi, doc_sv)[0]
    print("Doc pair", i, ":", cos_sim)


if len(doc_embeddings['fi']) == len(doc_embeddings['sv']):
    dump_file = wiki_file[:-4]+"_distilbert.pkl"
    with open(dump_file, 'wb') as f:
        pickle.dump(doc_embeddings, f)
        f.close()


