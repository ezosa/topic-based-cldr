import os
import numpy as np
import json

languages = ['en', 'de']

def get_denews_corpus(path):
    print("Getting DE-News articles from: ", path)
    documents = {lang: {} for lang in languages}
    for lang in languages:
        print("Lang:", lang.upper())
        docs_path = path + lang
        filenames = sorted(os.listdir(docs_path))
        for f in filenames:
            text = open(docs_path + "/" + f, 'r').read().split()
            index_start = list(np.where(np.array(text) == "<DOC")[0])
            for i in range(len(index_start) - 1):
                start_art = index_start[i] + 2
                end_art = index_start[i + 1]
                art_id = text[start_art-1][:-1]
                article = text[start_art:end_art]
                h_start = article.index("<H1>")
                h_end = article.index("</H1>")
                headline = article[h_start+1:h_end]
                headline = " ".join(headline)
                art_text = article[h_end+1:]
                art_text = " ".join(art_text)
                # print("\nID:", art_id)
                # print("Headline:", headline)
                # print("Article:", art_text)
                documents[lang][art_id] = {'headline': headline, 'content': art_text}
        print("Articles:", len(documents[lang]))
    dump_file = path + "parsed_articles.json"
    with open(dump_file, 'w') as f:
        json.dump(documents, f)
        print("Saved parsed articles as", dump_file, "!")


denews_path = "/proj/zosa/data/denews/1998/"
get_denews_corpus(denews_path)