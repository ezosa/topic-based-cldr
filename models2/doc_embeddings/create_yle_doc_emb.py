import pickle
import json
import numpy as np
import argparse
from collections import Counter
from data import load_word_embeddings, create_document_embedding

argparser = argparse.ArgumentParser()
argparser.add_argument('--emb_type', default='fasttext', type=str, help="embedding type: fasttext or bert")
argparser.add_argument('--agg_type', default='mean', type=str, help="aggregation type: mean or sum")
args = argparser.parse_args()

print("-"*5, "Creating doc embeddings for Yle articles ", "-"*5)
print("emb_type:", args.emb_type)
print("aggregation:", args.agg_type)
print("-"*50)


languages = ['fi', 'sv']
ft_emb = {'fi': '/proj/zosa/data/fasttext/wiki.multi.fi.vec',
          'sv': '/proj/zosa/data/fasttext/wiki.multi.sv.vec'}
bert_emb = {'fi': '',
            'sv': ''}

# load embeddings
embeddings = {}
for lang in languages:
    if args.emb_type == 'fasttext':
        emb_file = ft_emb[lang]
        emb = load_word_embeddings(emb_file)
        embeddings[lang] = emb
    else:
        emb_file = ""
        emb = pickle.load(open(emb_file, 'rb'))
        # for word in emb:
        #     emb[word] = emb[word][1:]
        embeddings[lang] = emb


# build doc embeddings by averaging word embeddings
doc_embeddings = {'fi': {}, 'sv': {}}
yle_path = "/proj/zosa/data/yle/"

# get FI articles with in aligned dataset
yle_filepath = yle_path + "yle_aligned_articles_2018.json"
yle_related = json.load(open(yle_filepath, 'r'))
for related in yle_related:
    fi_art = related['fi'][0]
    fi_id = fi_art['id']
    fi_text = fi_art['headline'] + ' ' + fi_art['content']
    fi_tokens = fi_text.lower().split()
    fi_emb = create_document_embedding(fi_tokens, embeddings['fi'], lang='fi')
    if fi_emb is not None:
        doc_embeddings['fi'][fi_id] = fi_emb

for related in yle_related:
    for sv_art in related['sv']:
        #sv_art = related['sv'][0]
        sv_id = sv_art['id']
        if sv_id not in doc_embeddings['sv']:
            sv_text = sv_art['headline'] + ' ' + sv_art['content']
            sv_tokens = sv_text.lower().split()
            sv_emb = create_document_embedding(sv_tokens, embeddings['sv'], lang='sv')
            if sv_emb is not None:
                doc_embeddings['sv'][sv_id] = sv_emb

# get all SV 2018 articles
# yle_filepath = yle_path + "yle_articles_2018.json"
# yle_articles = json.load(open(yle_filepath, 'r'))
# for sv_art in yle_articles['sv']:
#     sv_id = sv_art['id']
#     sv_text = sv_art['headline'] + ' ' + sv_art['content']
#     sv_tokens = sv_text.lower().split()
#     sv_emb = create_document_embedding(sv_tokens, embeddings['sv'], lang='sv')
#     if sv_emb is not None:
#         doc_embeddings['sv'][sv_id] = sv_emb


print("FI doc emb:", len(doc_embeddings['fi']))
print("SV doc emb:", len(doc_embeddings['sv']))

dump_file = yle_path + "yle_doc_emb_2018_" + args.emb_type + "_" + args.agg_type + "_small.pkl"
with open(dump_file, 'wb') as f:
    pickle.dump(doc_embeddings, f)
    f.close()
    print("Done! Saved doc embeddings at", dump_file, "!")



