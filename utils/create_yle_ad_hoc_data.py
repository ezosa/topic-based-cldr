import json
import os
import numpy as np

languages = ['fi', 'sv']
yle_path = "/proj/zosa/data/yle/"


def parse_articles(year):
    articles = {'fi': [], 'sv':[]}
    for lang in languages:
        print("-"*5, "Lang:", lang.upper(), "-"*5)
        filepath = yle_path + lang + "/" + str(year)
        months = sorted(os.listdir(filepath))
        for month in months:
            filepath = yle_path + lang + "/" + str(year) + "/" + month
            files = sorted(os.listdir(filepath))
            for json_file in files:
                filepath = yle_path + lang + "/" + str(year) + "/" + month + "/" + json_file
                print("Filepath:", filepath)
                data = json.load(open(filepath, "r"))
                data = data['data']
                # extract article content + metadata
                for art in data:
                    # not all articles have subjects
                    if 'subjects' in art:
                        art_id = art['id']
                        art_headline = art['headline']['full']
                        art_content = art['content']
                        art_text = ""
                        for content in art_content:
                            if 'text' in content:
                                art_text += content['text']
                        subjects = art['subjects']
                        art_subjects = [s['id'] for s in subjects if 'id' in s]
                        article = {'id': art_id,
                                   'headline': art_headline,
                                   'content': art_text,
                                   'subjects': art_subjects}
                        articles[lang].append(article)
    print("FI articles:", len(articles['fi']))
    print("SV articles:", len(articles['sv']))
    dump_file = "data/yle/yle_articles_"+str(year)+".json"
    with open(dump_file, 'w') as f:
        json.dump(articles, f)
        print("Saved articles as", dump_file, "!")


def align_parsed_articles(year, min_subjects=3):
    # align 1 FI articles with >= 1 SV articles based on common subjects
    articles_file = yle_path + "yle_articles_"+str(year)+".json"
    articles = json.load(open(articles_file, "r"))
    print("Loaded parsed articles from", articles_file)
    print("Aligning articles with at least", min_subjects, "shared subjects")
    fi_articles = articles['fi']
    sv_articles = articles['sv']
    print("FI articles:", len(fi_articles))
    print("SV articles:", len(sv_articles))
    aligned_articles = []
    for fi_art in fi_articles:
        aligned_art = {'fi': [fi_art], 'sv': []}
        fi_subjects = fi_art['subjects']
        for sv_art in sv_articles:
            sv_subjects = sv_art['subjects']
            common_sub = set(fi_subjects).intersection(set(sv_subjects))
            if len(common_sub) > min_subjects:
                #found SV article aligned with current FI article
                aligned_art['sv'].append(sv_art)
        if len(aligned_art['sv']) > 0:
            aligned_articles.append(aligned_art)
    print("FI articles with alignment:", len(aligned_articles))
    art_counts = [len(art['sv']) for art in aligned_articles]
    print("Mean aligned art:", np.mean(art_counts))
    dump_file = yle_path + "yle_aligned_articles_" + str(year) + "_" + str(min_subjects) + ".json"
    with open(dump_file, 'w') as f:
        json.dump(aligned_articles, f)


align_parsed_articles(year=2018)