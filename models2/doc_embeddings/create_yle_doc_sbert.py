import json
import pickle
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('xlm-r-distilroberta-base-paraphrase-v1')

languages = ['fi', 'sv']
max_len = 500

# Yle aligned articles
filepaths = {'sv': "/proj/zosa/data/yle/yle_aligned_articles_2018_3_small.json",
             'fi': "/proj/zosa/data/yle/yle_aligned_articles_2018_3_small.json"}


# encode articles SBERT embeddings
for lang in languages:
    yle_data = json.load(open(filepaths[lang], 'r'))
    articles = {}
    print("Lang:", lang.upper())
    for related in yle_data:
        for related_art in related[lang]:
            art_id = related_art['id']
            if art_id not in articles:
                articles[art_id] = related_art
    print("Articles:", len(articles))
    doc_embeddings = {}
    for art_id in articles:
        #art_id = art['id']
        art_text = articles[art_id]['headline'] + '. ' + articles[art_id]['content']
        art_text = art_text.lower()
        # if len(art_text) > max_len:
        #     art_text = ' '.join(art_text[:max_len])
        # else:
        # art_text = ' '.join(art_text)
        encoded_art = model.encode(art_text)
        doc_embeddings[art_id] = encoded_art
    # sanity check
    print("Doc embeddings:", len(doc_embeddings))
    # save embeddings
    dump_file = filepaths[lang][:-5] + "_" + lang + "_distilbert.pkl"
    with open(dump_file, 'wb') as f:
        pickle.dump(doc_embeddings, f)
        f.close()
        print("Saved SBERT doc embeddings as", dump_file, "!")


# import json
# filepath = "/proj/zosa/data/yle/yle_aligned_articles_2018_3.json"
# data = json.load(open(filepath, 'r'))
#
# yle_small = []
# sv_ids = []
# for related in data:
#     if 20 <= len(related['sv']) <= 30:
#         yle_small.append(related)
#         for art in related['sv']:
#             sv_ids.append(art['id'])
#
# dump_file = "/proj/zosa/data/yle/yle_aligned_articles_2018_3_small.json"
# with open(dump_file, 'w') as f:
#     json.dump(yle_small, f)


