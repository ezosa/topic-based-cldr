from contextualized_topic_models.models.ctm import ZeroShotTM
from contextualized_topic_models.utils.data_preparation import QuickText
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing
# from contextualized_topic_models.datasets.dataset import CTMDataset
# from gensim.corpora.dictionary import Dictionary
# import nltk
import pickle

import argparse
argparser = argparse.ArgumentParser()
argparser.add_argument('--train_data', default='/proj/zosa/data/denews/1997/parsed_articles_en.txt', type=str, help="text file with one document per line")
argparser.add_argument('--num_topics', default=100, type=int, help="topics to train")
argparser.add_argument('--epochs', default=50, type=int, help="epochs to train")
argparser.add_argument('--lang', default='en', type=str, help="lang code of training data")
args = argparser.parse_args()


print("----- Training ContextualizedTM on DE-News data -----")
print("train_data:", args.train_data)
print("num_topics:", args.num_topics)
print("epochs:", args.epochs)
print("lang:", args.lang)
print("-"*50)

# open training docs
print("Open training data")
documents = open(args.train_data, encoding='utf-8').readlines()
documents = [doc.strip().lower() for doc in documents]
print("training docs:", len(documents))

# pre-process docs
print("Pre-process docs")
if args.lang == 'en':
    stopwords_lang = 'english'
else:
    stopwords_lang = 'german'

sp = WhiteSpacePreprocessing(documents, stopwords_language=stopwords_lang)
preprocessed_documents, unpreprocessed_corpus, vocab = sp.preprocess()

# get BERT embeddings of docs
print("Get BERT encoding")
qt = QuickText("distiluse-base-multilingual-cased",
                text_for_bow=preprocessed_documents,
                text_for_bert=unpreprocessed_corpus)

training_dataset = qt.load_dataset()

# train CTM
print("Start training CTM")
ctm = ZeroShotTM(input_size=len(qt.vocab), bert_input_size=512, n_components=args.num_topics, num_epochs=args.epochs)
ctm.fit(training_dataset)

# check topics found
print("Done training CTM!")
topics = ctm.get_topic_lists(20)
for i, topic in enumerate(topics):
    print("Topic", i, ":", ', '.join(topic))

# save trained model

model_file = "results/cldr/ctm_" + args.train_data.split("/")[-1][:-4] + "_" + str(args.num_topics) + "topics_" + str(args.epochs) + "epochs.pkl"
with open(model_file, 'wb') as f:
    pickle.dump(ctm, f)
    f.close()
    print("Saved trained model as", model_file, "!")