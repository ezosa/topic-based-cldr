import pickle
import random

# create Wiki test data
wiki_file = "/proj/zosa/data/wiki/wikialign_clean_fi-sv.pkl"
wiki_data = pickle.load(open(wiki_file, 'rb'))
random.shuffle(wiki_data)

# test set
test_size = 10000
wiki_test = wiki_data[:test_size]
dump_file = wiki_file[:-4]+"_test.pkl"
with open(dump_file, 'wb') as f:
    pickle.dump(wiki_test, f)
    f.close()
    print("Saved test set as:", dump_file)

# train set
train_size = 50000
wiki_train = wiki_data[test_size:train_size+test_size]
dump_file = wiki_file[:-4]+"_train.pkl"
with open(dump_file, 'wb') as f:
    pickle.dump(wiki_train, f)
    f.close()
    print("Saved train set as:", dump_file)

