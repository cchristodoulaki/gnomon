
import string
import re
import spacy
from nltk import download

print('\nloading en_core_web_sm ...')
try:
    sp = spacy.load('en_core_web_sm')
except OSError:
  download(model="en_core_web_sm")
  sp = spacy.load("en_core_web_sm")
print('en_core_web_sm loaded!')



def dict_table_headers(data_df, header_df):
    headers = {}
    for csv_idx in data_df.columns:
        attribute_name = ""
        if header_df is not None and csv_idx in header_df.columns:
            attribute_name = header_df[csv_idx].iloc[0]
        headers[csv_idx] =  attribute_name
    return headers


def normalize_whitespace(s='The       quick, brown fox jumps over the 2 lazy dogs.'):
    if  s and isinstance(s, str):
        return ' '.join(s.split())
    elif  s and isinstance(s, list):
        return '' 
      
# Punctuation Removal
def remove_punctuation( s= 'The 1 quick, brown fox jumps over the 2 lazy dogs.'):
    # # initializing punctuations string
    # punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    if s and isinstance(s, str):
        for c in string.punctuation:
            s = s.replace(c, " ")
    return s
  
def lemmatize(s = 'The quickest brown fox was jumping over the lazy dog'):
    return ' '.join([token.lemma_ for token in sp(s)])  

# CamelCase split
def split_camelCase(s = 'TheQuick, brownFOX jumps over TheLazyDogs.'): 
    if s and isinstance(s, str):  
        splitted = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', s)).split()
        return ' '.join(splitted)
    elif  s and isinstance(s, list):
        return ''         
    else:
        return s