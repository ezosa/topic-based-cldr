# Datasets for Contextual TM (CTM) must be in a single text file with one document per line

import json
import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--parsed_data', default='parsed_articles.json', type=str, help="parsed DE-News articles")
args = argparser.parse_args()

print("Loading articles from", args.parsed_data)
parsed_denews_file = args.parsed_data
data = json.load(open(parsed_denews_file, 'r'))


languages = ['en', 'de']
for lang in languages:
    docs = data[lang]
    doc_texts = []
    for doc_id in docs:
        doc_text = docs[doc_id]['headline'] + ". " + docs[doc_id]['content']
        doc_texts.append(doc_text)
    outfile = parsed_denews_file[:-5] + "_" + lang + ".txt"
    with open(outfile, 'w') as f:
        f.write("\n".join(doc_texts))
        f.close()
    print("Done! Wrote", len(doc_texts), "texts to", outfile, "!" )

