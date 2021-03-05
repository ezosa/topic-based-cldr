# Datasets for Contextual TM (CTM) must be in a single text file with one document per line

import json
import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--yle_data', default='yle_articles_2017.json', type=str, help="parsed Yle articles")
args = argparser.parse_args()

yle_path = "/proj/zosa/data/yle/"

print("Loading articles from", yle_path + args.yle_data)
data = json.load(open(yle_path + args.yle_data, 'r'))


languages = ['fi', 'sv']
for lang in languages:
    print("Lang:", lang.upper())
    docs = data[lang]
    doc_texts = []
    for doc in docs:
        doc_text = doc['headline'] + ". " + doc['content']
        doc_texts.append(doc_text)
    print("Docs:", len(doc_texts))
    outfile = yle_path + args.yle_data[:-5] + "_" + lang + ".txt"
    with open(outfile, 'w') as f:
        f.write("\n".join(doc_texts))
        f.close()
    print("Done! Wrote", len(doc_texts), "texts to", outfile, "!" )

