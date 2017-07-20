################################ 
#                              #
#    @_PUCHA_LEARNING_@        #
#                              #
################################ 
import pandas as pd
import numpy as np 
import nltk
import string
from collections import Counter
from nltk.corpus import stopwords
from nltk import word_tokenize
import pickle

nltk.download()
# Set Path to dataset folder 
path = "dataset/"

### Training and test set 
training_text = pd.read_csv(path + "training_text",sep = '\|\|', header = None, skiprows = 1, names = ['ID', 'Text'], engine = 'python', encoding = 'utf-8')
training_variants = pd.read_csv(path +"training_variants")
training_df = pd.merge(training_variants, training_text, on='ID', how='left')
training_df = training_df.set_index(['ID'])
del training_text, training_variants

test_text = pd.read_csv(path + "test_text",sep = '\|\|', header = None, skiprows = 1, names = ['ID', 'Text'], engine = 'python', encoding = 'utf-8')
test_variants = pd.read_csv(path +"test_variants")
test_df = pd.merge(test_variants, test_text, on='ID', how='left')
test_df = test_df.set_index(['ID'])
del test_text, test_variants

# Prior analysis 
gene_count_df = training_df.groupby(["Class"])["Gene"].aggregate("count").reset_index().sort_values(["Gene"], ascending=False)
gene_count_df

#Create dict with most common words
dict_words = [{}, {}, {}, {}, {}, {}, {}, {}, {}]
stop = set(stopwords.words('english'))
punctuations = list(string.punctuation)
for index, row in training_df.iterrows():
    # Text 
    text_example = row['Text']
    class_example = row['Class']

    # Tokenize and remove stopwords 
    text_example_tokenize = nltk.word_tokenize(text_example)
    test = [i.lower() for i in text_example_tokenize if i.lower() not in stop and i not in punctuations and type(int(i)) is not int]
    c = Counter(test)
    c_15 = c.most_common(15)
    
    #Add to dict    
    for word_occ in c_15: 
        if word_occ[0] not in dict_words[class_example-1]:
            dict_words[class_example-1][word_occ[0]] = word_occ[1]
        else:
            dict_words[class_example-1][word_occ[0]] += word_occ[1]
    print (index)

pickle.dump(dict_words, open("dict_word.pickle", "wb"))

####