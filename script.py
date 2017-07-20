################################ 
#                              #
#    @_PUCHA_LEARNING_@        #
#                              #
################################ 
from time import gmtime, strftime
from collections import Counter
from nltk.corpus import stopwords
from nltk import word_tokenize

import pandas as pd
import numpy as np 
import nltk
import string
import socket
import pickle

# Set Path to dataset folder 
path_to_data = "dataset/"
path_to_output = "output/"

### Training and test set 
training_text = pd.read_csv(path_to_data + "training_text",sep = '\|\|', header = None, skiprows = 1, names = ['ID', 'Text'], engine = 'python', encoding = 'utf-8')
training_variants = pd.read_csv(path_to_data +"training_variants")
training_df = pd.merge(training_variants, training_text, on='ID', how='left')
training_df = training_df.set_index(['ID'])
del training_text, training_variants

test_text = pd.read_csv(path_to_data + "test_text",sep = '\|\|', header = None, skiprows = 1, names = ['ID', 'Text'], engine = 'python', encoding = 'utf-8')
test_variants = pd.read_csv(path_to_data+"test_variants")
test_df = pd.merge(test_variants, test_text, on='ID', how='left')
test_df = test_df.set_index(['ID'])
test_df['Classified_Class'] = np.zeros((test_df.shape[0], 1)) #Added 'Classified_Class' as the column for result
del test_text, test_variants


# Train Stage 

##Create dict with most common words
dict_words = [{}, {}, {}, {}, {}, {}, {}, {}, {}]
stop = set(stopwords.words('english'))
punctuations = list(string.punctuation)
for index, row in training_df.iterrows():
    ### Text 
    text_example = row['Text']
    class_example = row['Class']

    ### Tokenize and remove stopwords 
    text_example_tokenize = nltk.word_tokenize(text_example)
    test = [i.lower() for i in text_example_tokenize if i.lower() not in stop and i not in punctuations and isinstance(i, int) is False]
    c = Counter(test)
    
    ### Add to dict    
    for word_occ in c.most_common(15): 
        if word_occ[0] not in dict_words[class_example-1]:
            dict_words[class_example-1][word_occ[0]] = word_occ[1]
        else:
            dict_words[class_example-1][word_occ[0]] += word_occ[1]
    print ("Working wih observation: ", index)

## To persist the dict_words object
pickle.dump(dict_words, open("dict_word.pickle", "wb"))


# Test Stage


#Generate submission file
tags = ["class1","class2","class3","class4","class5","class6","class7","class8","class9"]
submission_data = pd.DataFrame(np.zeros((test_df.shape[0], len(tags))), columns = tags)

# @TODO : Add a '1' in the selected classs for each row
# 5,0,1,0,0,0,0,0,0,0 <@ID, @[class1 - class9]
submission_data.to_csv(path_to_output + "submissionFile_" + strftime("%Y-%m-%d_%H-%M-%S", gmtime()) + "_" + socket.gethostname(),index_label="ID")
                       




















