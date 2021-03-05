import json
import pickle
from data import get_denews_docs_paired, load_word_embeddings, create_document_embedding
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--emb_type', default='fasttext', type=str, help="embedding type: fasttext, fasttext-big, cr5, or bert")
argparser.add_argument('--agg_type', default='mean', type=str, help="aggregation type: mean or sum")
args = argparser.parse_args()

print("-"*5, "Create doc embeddings for DE-News articles ", "-"*5)
print("emb_type:", args.emb_type)
print("agg_type:", args.agg_type)
print("-"*50)

languages = ['en', 'de']

ft_emb = {'en': '/proj/zosa/data/fasttext/wiki.multi.en.vec',
          'de': '/proj/zosa/data/fasttext/wiki.multi.de.vec'}

ft_big_emb = {'en': '/proj/zosa/data/fasttext/wiki.en.align.vec',
              'de': '/proj/zosa/data/fasttext/wiki.de.align.vec'}

cr5_emb = {'en': '/proj/zosa/data/Cr5/en.txt',
           'de': '/proj/zosa/data/Cr5/de.txt'}

bert_emb = {'en': '',
            'de': ''}

# load word embeddings
embeddings = {}
for lang in languages:
    if args.emb_type == 'fasttext':
        emb_file = ft_emb[lang]
        emb = load_word_embeddings(emb_file)
        embeddings[lang] = emb
    elif args.emb_type == 'fasttext-big':
        emb_file = ft_big_emb[lang]
        emb = load_word_embeddings(emb_file)
        embeddings[lang] = emb
    elif args.emb_type == 'cr5':
        emb_file = cr5_emb[lang]
        emb = load_word_embeddings(emb_file)
        embeddings[lang] = emb
    else:
        emb_file = ""
        emb = pickle.load(open(emb_file, 'rb'))
        embeddings[lang] = emb

denews_filepath = "/proj/zosa/data/denews/1998/parsed_articles.json"
paired_docs = get_denews_docs_paired(denews_filepath)


doc_embeddings = {'en':[], 'de':[]}
# create doc embeddings from word embeddings
print("Creating doc embeddings")
for lang in languages:
    print("Lang:", lang.upper())
    docs = paired_docs[lang]
    for doc in docs:
        tokens = doc.lower().split()
        doc_emb = create_document_embedding(tokens, embeddings[lang], lang)
        doc_embeddings[lang].append(doc_emb)

# sanity check
for lang in languages:
    print(lang.upper(), "doc embeddings:", len(doc_embeddings[lang]))

# save embeddings
dump_file = denews_filepath[:-5] + "_" + args.emb_type + "_" + args.agg_type + ".pkl"
with open(dump_file, 'wb') as f:
    pickle.dump(doc_embeddings, f)
    f.close()
    print("Saved doc embeddings as", dump_file, "!")


