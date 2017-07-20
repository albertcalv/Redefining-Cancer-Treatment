################################ 
#                              #
#    @_PUCHA_LEARNING_@        #
#                              #
################################ 
import pandas as pd
import numpy as np 
import nltk

from collections import Counter

# Set Path to dataset folder 
#viva pucha libre
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

## Text example  
text_example = training_df.iloc[2]['Text']
text_example