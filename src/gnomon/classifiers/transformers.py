from gnomon import utils
from gnomon.language_models.GloVe import glove_util

from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import class_weight
import time

import numpy as np
import pandas as pd
import json
import logging
# Configure logging
logging.basicConfig(level=logging.INFO)
# Get the logger
logger = logging.getLogger()
logger.disabled = True

from nltk import word_tokenize
from urllib.parse import urlparse
from statistics import mean
        
class SampleWeightTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, class_weight=None):
        self.class_weight = class_weight
        
    """
    We compute sample_weight dynamically 
    during training using the compute_sample_weight 
    function from the class_weight module. 
    """
    def fit(self, X, y=None):
        if self.class_weight is not None:
            self.sample_weight = class_weight.compute_sample_weight(
                class_weight = self.class_weight, 
                y=y
            )
            print(len(y))
            print(len(self.sample_weight))
        else:
            self.sample_weight = None
        return self
    
    """
    We modify the transform method to handle 
    cases where self.sample_weight is None, 
    which may happen if class_weight is 
    not provided to the SampleWeightTransformer.
    """
    def transform(self, X):
        if self.sample_weight is not None:
            return X * self.sample_weight[:, None]
        else:
            return X

class FeatureSelector(TransformerMixin, BaseEstimator):
    def __init__(self, features = []):
        self.features = features
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X[self.features]
    
    
class CategoricalTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, normalize_steps = [], feature=""):
        self.normalize_steps = normalize_steps
        self.feature = feature
        
    def __normalize(self, s):
        if not isinstance(s, str):
            s = str(s)
            
        if self.normalize_steps is None:
            return s
        
        for step in self.normalize_steps:
                
            if step == 'lowercase':
                s = s.lower()             
            elif step == 'strip':
                s = s.strip()
            elif step == 'split_camelCase':            
                s = utils.split_camelCase(s)
            elif step == 'remove_punctuation':
                s = utils.remove_punctuation(s)
            elif step == 'lemmatize':
                s = utils.lemmatize(s)
            elif step == 'encode':
                token_array = s.split()
                for i, t in enumerate(token_array):
                    if t.strip().isdigit():
                        token_array[i]= " __NUMBER__ "
                s = ' '.join(token_array)
        return s

    def fit(self, X, y=None):
        logger.info(f"CategoricalTransformer: Fitting data...")
        return self

    def transform(self, X, y = None ):   
        logger.info(f"CategoricalTransformer: Transforming data...")   
        logger.info(f'input X=\n{X}')  
        transformed_data = X[self.feature].apply(lambda x: self.__normalize(str(x)))
        logger.info(f'\n transformed_data=\n{transformed_data}\n---')
        return transformed_data

def get_transformer_class(meta_feature):
    # create a string by replacing underscores in meta_feature with spaces and coverting to CamelCase
    class_name = f"{meta_feature}_FeatureTransformer"
    # print(f'{class_name=}')
    return globals().get(class_name)

def get_transformer_function(func_name):
    return globals().get(func_name)

def avg(x_list):
    return round(mean(x_list), 0)

def is_valid_url(url):
    parsed_url = urlparse(url)
    return bool(parsed_url.scheme and parsed_url.netloc)

def is_empty(s):
    if s is None:
        return True
    
    if str(s).strip() == '':
        return True
    
    if s == np.nan:
        return True
    
def extract_structure_column_index(X, feature_name):
    Xnew = X.copy()
    Xnew.loc[:,feature_name] =  X['csv_column']
    return Xnew[[feature_name]]

def extract_values(X, feature_name):
    Xnew = X.copy()
    
    Xnew.loc[:,'table_column'] = Xnew.loc[:,'table_column'].apply(json.loads)
    Xnew[feature_name] = Xnew.apply(lambda x: str((' ').join([str(v).strip() for v in x['table_column']['values']])).strip(), axis=1)    
    
    # Xnew.loc[:,'values'] = Xnew.loc[:,'values'].apply(json.loads)
    # Xnew[feature_name] = Xnew.apply(lambda x: str((' ').join([str(v).strip() for v in x['values']])).strip(), axis=1)    
    
    return Xnew[[feature_name]]

def extract_values_max_token_count(X, feature_name):
    Xnew = X.copy()
    Xnew.loc[:,feature_name] =  X['table_column'].apply(lambda x: max(len(word_tokenize(str(value))) for value in json.loads(x)['values']))
    # Xnew.loc[:,feature_name] =  X['values'].apply(lambda x: max(len(word_tokenize(str(value))) for value in json.loads(x)))

    return Xnew[[feature_name]]  

def extract_values_avg_token_count(X, feature_name):
    Xnew = X.copy()
    Xnew.loc[:,feature_name] =  X['table_column'].apply(lambda x: avg(len(word_tokenize(str(value))) for value in json.loads(x)['values']))
    # Xnew.loc[:,feature_name] =  X['values'].apply(lambda x: avg(len(word_tokenize(str(value))) for value in json.loads(x)))
    return Xnew[[feature_name]] 

def extract_values_min_token_count(X, feature_name):
    Xnew = X.copy()
    Xnew.loc[:,feature_name] =  X['table_column'].apply(lambda x: min(len(word_tokenize(str(value))) for value in json.loads(x)['values']))
    # Xnew.loc[:,feature_name] =  X['values'].apply(lambda x: min(len(word_tokenize(str(value))) for value in json.loads(x)))
    return Xnew[[feature_name]] 

def extract_values_max_char_count(X, feature_name):
    Xnew = X.copy()
    Xnew.loc[:,feature_name] = X['table_column'].apply(lambda x: max(len(str(value)) for value in json.loads(x)['values']))
    # Xnew.loc[:,feature_name] = X['values'].apply(lambda x: max(len(str(value)) for value in json.loads(x)))
    return Xnew[[feature_name]]

def extract_values_avg_char_count(X, feature_name):
    Xnew = X.copy()
    Xnew.loc[:,feature_name] = X['table_column'].apply(lambda x: avg(len(str(value)) for value in json.loads(x)['values']))
    # Xnew.loc[:,feature_name] = X['values'].apply(lambda x: avg(len(str(value)) for value in json.loads(x)))
    return Xnew[[feature_name]]

def extract_values_min_char_count(X, feature_name):
    Xnew = X.copy()
    Xnew.loc[:,feature_name] = X['table_column'].apply(lambda x: min(len(str(value)) for value in json.loads(x)['values']))
    # Xnew.loc[:,feature_name] = X['values'].apply(lambda x: min(len(str(value)) for value in json.loads(x)))
    return Xnew[[feature_name]]


def extract_values_all_url(X, feature_name):        
    Xnew = X.copy()
    Xnew.loc[:, feature_name] =  X['table_column'].apply(lambda x: all(is_valid_url(str(value).strip()) for value in json.loads(x)['values']))
    # Xnew.loc[:, feature_name] =  X['values'].apply(lambda x: all(is_valid_url(str(value).strip()) for value in json.loads(x)))
    return Xnew[[ feature_name]]  
  
def extract_values_any_url(X, feature_name):
    Xnew = X.copy()
    Xnew.loc[:,feature_name] =  X['table_column'].apply(lambda x: any(is_valid_url(str(value).strip()) for value in json.loads(x)['values']))
    # Xnew.loc[:,feature_name] =  X['values'].apply(lambda x: any(is_valid_url(str(value).strip()) for value in json.loads(x)))
    return Xnew[[feature_name]]

def extract_values_all_has_underscore(X, feature_name):
    Xnew = X.copy()
    Xnew.loc[:,feature_name] =  X['table_column'].apply(lambda x: all('_' in str(value).strip() for value in json.loads(x)['values']))
    # Xnew.loc[:,feature_name] =  X['values'].apply(lambda x: all('_' in str(value).strip() for value in json.loads(x)))
    return Xnew[[feature_name]]    

def extract_values_any_has_underscore(X, feature_name):
    Xnew = X.copy()
    Xnew.loc[:,feature_name] =  X['table_column'].apply(lambda x: any('_' in str(value).strip() for value in json.loads(x)['values']))
    # Xnew.loc[:,feature_name] =  X['values'].apply(lambda x: any('_' in str(value).strip() for value in json.loads(x)))
    return Xnew[[feature_name]]

def extract_values_all_unique(X,feature_name):
    Xnew = X.copy()
    Xnew.loc[:,feature_name] =  X['table_column'].apply(lambda x: len(json.loads(x)['values'])==len(set(json.loads(x)['values'])))
    # Xnew.loc[:,feature_name] =  X['values'].apply(lambda x: len(json.loads(x))==len(set(json.loads(x))))
    return Xnew[[feature_name]]

def extract_values_any_empty(X, feature_name):
    Xnew = X.copy()
    Xnew.loc[:,feature_name] =  X['table_column'].apply(lambda x: any(is_empty(value) for value in json.loads(x)['values']))
    # Xnew.loc[:,feature_name] =  X['values'].apply(lambda x: any(is_empty(value) for value in json.loads(x)))
    return Xnew[[feature_name]]

def extract_values_all_lower(X, feature_name):
    Xnew = X.copy()
    Xnew.loc[:,feature_name] =  X['table_column'].apply(lambda x: all(str(value).islower() for value in json.loads(x)['values']))
    # Xnew.loc[:,feature_name] =  X['values'].apply(lambda x: all(str(value).islower() for value in json.loads(x)))
    return Xnew[[feature_name]]    
    
def extract_values_all_upper(X, feature_name):
    Xnew = X.copy()
    Xnew.loc[:,feature_name] =  X['table_column'].apply(lambda x: all(str(value).isupper() for value in json.loads(x)['values']))
    # Xnew.loc[:,feature_name] =  X['values'].apply(lambda x: all(str(value).isupper() for value in json.loads(x)))
    return Xnew[[feature_name]]
            
    
def extract_values_all_title(X, feature_name):
    Xnew = X.copy()
    Xnew.loc[:,feature_name] =  X['table_column'].apply(lambda x: all(str(value).istitle() for value in json.loads(x)['values']))
    # Xnew.loc[:,feature_name] =  X['values'].apply(lambda x: all(str(value).istitle() for value in json.loads(x)))
    return Xnew[[feature_name]] 

def extract_header(X, feature_name):
    Xnew = X.copy()
    Xnew.loc[:,'table_column'] = Xnew.loc[:,'table_column'].apply(json.loads)
    Xnew[feature_name] = Xnew.apply(lambda x: x['table_column']['header'], axis=1)
    return Xnew[[feature_name]]

def extract_header_is_title(X, feature_name):
    X_new = X.copy()
    X_new[feature_name] = X['table_column'].apply(lambda x: str(json.loads(x)['header']).istitle())
    return X_new[[feature_name]]

def extract_header_is_lower(X, feature_name):
    X_new = X.copy()
    X_new[feature_name] = X['table_column'].apply(lambda x: str(json.loads(x)['header']).islower())
    return X_new[[feature_name]]

def extract_header_is_upper(X, feature_name):
    X_new = X.copy()
    X_new[feature_name] = X['table_column'].apply(lambda x: str(json.loads(x)['header']).isupper())
    return X_new[[feature_name]]  

def extract_header_char_count(X, feature_name):
    X_new = X.copy()
    X_new[feature_name] = X['table_column'].apply(lambda x: len(str(json.loads(x)['header'])))
    return X_new[[feature_name]]      
                      
def extract_header_token_count(X, feature_name):
    Xnew = X.copy()
    Xnew.loc[:,feature_name] =  X['table_column'].apply(lambda x: len( word_tokenize( str(json.loads(x)['header']) ) ))
    return Xnew[[feature_name]]    


           
class Values_Semantic_Glove_FeatureTransformer(TransformerMixin):
    def __init__(self, feature_name, embeddings):
        self.feature_name = feature_name 
        self.feature_names = []  # Store feature names during transformation
        self.glove_model = embeddings
        logger.info(f"Values_Semantic_Glove_FeatureTransformer init")
        logger.info(f'{len(self.glove_model)=}')
        
        self.pipeline = Pipeline([            
                
                (f'{feature_name}_extractor', FunctionTransformer(
                    func=extract_values, 
                    kw_args={'feature_name': feature_name},
                    validate=False) 
                ),                       
                ('categorical_transformer', CategoricalTransformer(
                        normalize_steps=[
                                'strip',
                                'split_camelCase', 
                                'remove_punctuation'
                                # 'lowercase',
                                # 'lemmatize',
                                # 'encode'
                            ],
                        feature = self.feature_name
                        ) ),                  
                ('glove', GloveEmbeddingTransformer(self.glove_model) )   
        ])
            
    def fit(self, X, y=None):
        logger.info(f"Values_Semantic_Glove_FeatureTransformer: Fitting data...")
        logger.info(f'\n---\ninput X=\n{X}\n---\n')
        fitted =  self.pipeline.fit(X, y)
        
        # DEBUG ONLY
        data_transformed = X
        for name, step in self.pipeline.named_steps.items():
            data_transformed = step.transform(data_transformed)
        # Get the output of the glove vectorizer
        glove_output = data_transformed
        logger.info(f'glove_output example=\n{glove_output[0:2]}')
        # DEBUG ONLY ENDS
        
        return fitted 

    def transform(self, X):
        logger.info(f"Values_Semantic_Glove_FeatureTransformer: Transforming data...")
        # Transform the data and store the feature names
        transformed_data = self.pipeline.transform(X)
        logger.info(f'Values_Semantic_Glove_FeatureTransformer.transform: transformed_data = \n{transformed_data}')
        glove_step = self.pipeline.named_steps['glove']
        self.feature_names = glove_step.get_feature_names_out()
        logger.info(f'Values_Semantic_Glove_FeatureTransformer.transform: self.feature_names={self.feature_names} \n\n')
        return transformed_data
    
    def get_feature_names(self):
        return [ name + "___ValuesGlove" for name in self.feature_names]
            
class Values_Semantic_Tfidf_FeatureTransformer(TransformerMixin):
    def __init__(self, feature_name):
        self.feature_name = feature_name 
        self.feature_names = []  # Store feature names during transformation
               
        self.pipeline = Pipeline([            
                
                (f'{feature_name}_extractor', FunctionTransformer(
                    func=extract_values, 
                    kw_args={'feature_name': feature_name},
                    validate=False) ),                       
                ('categorical_transformer', CategoricalTransformer(
                        normalize_steps=[
                                'strip',
                                'split_camelCase', 
                                'remove_punctuation',
                                'lowercase',
                                'lemmatize',
                                'encode'
                            ],
                        feature = self.feature_name
                        ) ),  
                
                ('tfidf', TfidfVectorizer(
                            encoding = 'utf-8', 
                            decode_error='ignore', 
                            smooth_idf=True, 
                            min_df=1  ) ) 
                
        ])
        
    def fit(self, X, y=None):
        return self.pipeline.fit(X, y)

    def transform(self, X):
        transformed_data = self.pipeline.transform(X)
        tfidf_step = self.pipeline.named_steps['tfidf']
        self.feature_names = tfidf_step.get_feature_names_out()
        return transformed_data
    
    def get_feature_names(self):
        return [ name + "___ValuesTfidf" for name in self.feature_names] 
       
class Header_Semantic_Tfidf_FeatureTransformer(TransformerMixin):
    def __init__(self, feature_name):
        self.feature_name = feature_name 
        self.feature_names = []  # Store feature names during transformation
               
        self.pipeline = Pipeline([            
                
                (f'{feature_name}_extractor', FunctionTransformer(func=extract_header, 
                                                    kw_args={'feature_name': feature_name},
                                                    validate=False) ),                       
                ('categorical_transformer', CategoricalTransformer(
                        normalize_steps=[
                                'strip',
                                'split_camelCase', 
                                'remove_punctuation',
                                'lowercase',
                                'lemmatize',
                                'encode'
                            ],
                        feature = self.feature_name
                        ) ),  
                
                ('tfidf', TfidfVectorizer(
                            encoding = 'utf-8', 
                            decode_error='ignore', 
                            smooth_idf=True, 
                            min_df=1  ) ) 
                
        ])
        
    def fit(self, X, y=None):
        logger.info(f"Header_Semantic_Tfidf_FeatureTransformer: Fitting data...")
        logger.info(f'\n---\nX=\n{X}\n---\n')
        fitted =  self.pipeline.fit(X, y)
        # DEBUG ONLY
        data_transformed = X
        for name, step in self.pipeline.named_steps.items():
            data_transformed = step.transform(data_transformed)
        # Get the output of the tfidf vectorizer
        tfidf_output = data_transformed
        logger.info(f'tfidf_output=\n{tfidf_output}')
        # DEBUG ONLY ENDS
        
        return fitted 

    def transform(self, X):
        logger.info(f"Header_Semantic_Tfidf_FeatureTransformer: Transforming data...")
        # Transform the data and store the feature names
        logger.info(f'{X=}')
        transformed_data = self.pipeline.transform(X)
        logger.info(f'HeaderTextExtractor out = \n{transformed_data}')
        tfidf_step = self.pipeline.named_steps['tfidf']
        self.feature_names = tfidf_step.get_feature_names_out()
        logger.info(f'self.feature_names={self.feature_names} \n\n')
        return transformed_data
    
    def get_feature_names(self):
        return [ name + "___HeaderTfidf" for name in self.feature_names]


class GloveEmbeddingTransformer(BaseEstimator, TransformerMixin):
        
    def __init__(self, glove_model, post_process=True):
        self.embedding_model = glove_model  
        self.post_process=post_process#ensure that the embeddings produced by this transformer are non-negative 

    def fit(self, X, y=None):
        logger.info("GloveEmbeddingTransformer: GloVe vectorizer initialized")
        logger.info(f"GloveEmbeddingTransformer: Fitting data...")
        return self

    def transform(self, X):
        logger.info(f"GloveEmbeddingTransformer: Transforming data...")
        logger.info(f'input X=\n{X}\n')
        X_embeddings = []
        for i, datapoint in enumerate(X):
            logger.info(f"Transforming datapoint {i}...")
            
            start_time = time.time()
            phrase_embedding = glove_util.get_phrase_embedding(datapoint, self.embedding_model)
            elapsed_time = time.time() - start_time
            logger.info(f"Datapoint {i} transformed in {elapsed_time:.2f} seconds")
            
            X_embeddings.append(phrase_embedding.embedding)
            logger.info(f"... datapoint {i} transformed.")
        
        if isinstance(X_embeddings, list):
            logger.info(f'Lengths of embeddings: {[embedding.shape for embedding in X_embeddings]}')

        return np.array(X_embeddings)
    
    def get_feature_names_out(self):
        embedding_dim = len(next(iter(self.embedding_model.values())))
        return [f'glove_{i}' for i in range(1, embedding_dim + 1)]

    def close(self):
        logger.info("GloveEmbeddingTransformer: Closing")
        if hasattr(self.glove_model, 'close'):
            self.glove_model.close()
           
class Header_Semantic_Glove_FeatureTransformer(TransformerMixin):
    def __init__(self, feature_name, embeddings):
        self.feature_name = feature_name 
        self.feature_names = []  # Store feature names during transformation
        self.glove_model = embeddings
        logger.info(f"Header_Semantic_Glove_FeatureTransformer init")
        logger.info(f'{len(self.glove_model)=}')
        
        self.pipeline = Pipeline([            
                
                (f'{feature_name}_extractor', FunctionTransformer(
                    func=extract_header, 
                    kw_args={'feature_name': feature_name},
                    validate=False) 
                ),                       
                ('categorical_transformer', CategoricalTransformer(
                        normalize_steps=[
                                'strip',
                                'split_camelCase', 
                                'remove_punctuation'
                                # 'lowercase',
                                # 'lemmatize',
                                # 'encode'
                            ],
                        feature = self.feature_name
                        ) ),  
                
                # # ('glove', GloVeVectorizer(load_glove_model()) ) 
                # ('glove', WordEmbeddingTransformer(self.glove_model) ) 
                
                ('glove', GloveEmbeddingTransformer(self.glove_model) )   
        ])
            
    def fit(self, X, y=None):
        logger.info(f"Header_Semantic_Glove_FeatureTransformer: Fitting data...")
        logger.info(f'\n---\ninput X=\n{X}\n---\n')
        fitted =  self.pipeline.fit(X, y)
        
        # DEBUG ONLY
        data_transformed = X
        for name, step in self.pipeline.named_steps.items():
            data_transformed = step.transform(data_transformed)
        # Get the output of the glove vectorizer
        glove_output = data_transformed
        logger.info(f'glove_output example=\n{glove_output[0:2]}')
        # DEBUG ONLY ENDS
        
        return fitted 

    def transform(self, X):
        logger.info(f"Header_Semantic_Glove_FeatureTransformer: Transforming data...")
        # Transform the data and store the feature names
        transformed_data = self.pipeline.transform(X)
        logger.info(f'Header_Semantic_Glove_FeatureTransformer.transform: transformed_data = \n{transformed_data}')
        glove_step = self.pipeline.named_steps['glove']
        self.feature_names = glove_step.get_feature_names_out()
        logger.info(f'Header_Semantic_Glove_FeatureTransformer.transform: self.feature_names={self.feature_names} \n\n')
        return transformed_data
    
    def get_feature_names(self):
        return [ name + "___HeaderGlove" for name in self.feature_names]

# class WordEmbeddingTransformer(BaseEstimator, TransformerMixin):
        
#     def __init__(self, glove_model, post_process=True):
#         self.embedding_model = glove_model  
#         self.post_process=post_process#ensure that the embeddings produced by this transformer are non-negative 

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         # Convert text data to word embeddings
#         embedded_data = []
#         for doc in X:
#             words = doc.split()  # Tokenize the text
#             # Calculate the average word embeddings for the document
#             doc_embeddings = [self.embedding_model[word] for word in words if word in self.embedding_model]
#             if doc_embeddings:
#                 doc_embeddings = np.mean(doc_embeddings, axis=0)
#                 if self.post_process:
#                     doc_embeddings[doc_embeddings < 0] = 0  # Clip negative values to zero
            
#             else:
#                 doc_embeddings = np.zeros(self.embedding_model.vector_size)
#             embedded_data.append(doc_embeddings)
            
#         # print(np.array(embedded_data))
#         return np.array(embedded_data)
    
# class GloVeVectorizer(BaseEstimator, TransformerMixin):
#     def __init__(self, glove_model):
#         self.glove_model = glove_model     
    
#     def fit(self, X, y=None):
#         logger.info(f"GloVeVectorizer: Fitting data...")
#         return self
    
#     def transform(self, X):        
#         document_embeddings = []
#         for doc in X:
#             tokens = doc.lower().split()  # Simple splitting for illustration
#             doc_embedding = np.zeros(self.glove_model.vector_size)
#             num_tokens = 0
#             for token in tokens:
#                 if token in self.glove_model:
#                     doc_embedding += self.glove_model[token]
#                     num_tokens += 1
#             if num_tokens > 0:
#                 doc_embedding /= num_tokens
#             document_embeddings.append(doc_embedding)
#         return np.array(document_embeddings)
    
class Values_Structure_AllLower_FeatureTransformer(TransformerMixin):
    def __init__(self, feature_name):
        self.feature_name = feature_name 
        self.feature_names = [feature_name]  # Store feature names during transformation
        self.pipeline = Pipeline(
            [(f'{feature_name}_extractor', FunctionTransformer(
                                            func=extract_values_all_lower, 
                                            kw_args={'feature_name': feature_name},
                                            validate=False))])       
        
    def fit(self, X, y=None):
        return self.pipeline.fit(X, y)
    
    def transform(self, X):  
        transformed_data = self.pipeline.transform(X)   
        return transformed_data
    
    def get_feature_names(self):
        return self.feature_names 
    
class Values_Structure_AllUpper_FeatureTransformer(TransformerMixin):
    def __init__(self, feature_name):
        self.feature_name = feature_name 
        self.feature_names = [feature_name]  # Store feature names during transformation
        self.pipeline = Pipeline(
            [(f'{feature_name}_extractor', FunctionTransformer(
                                            func=extract_values_all_upper, 
                                            kw_args={'feature_name': feature_name},
                                            validate=False))])       
        
    def fit(self, X, y=None):
        return self.pipeline.fit(X, y)
    
    def transform(self, X):  
        transformed_data = self.pipeline.transform(X)   
        return transformed_data
    
    def get_feature_names(self):
        return self.feature_names 

            
class Values_Structure_AllTitle_FeatureTransformer(TransformerMixin):
    def __init__(self, feature_name):
        self.feature_name = feature_name 
        self.feature_names = [feature_name]  # Store feature names during transformation
        self.pipeline = Pipeline(
            [(f'{feature_name}_extractor', FunctionTransformer(
                                            func=extract_values_all_title, 
                                            kw_args={'feature_name': feature_name},
                                            validate=False))])       
        
    def fit(self, X, y=None):
        return self.pipeline.fit(X, y)
    
    def transform(self, X):  
        transformed_data = self.pipeline.transform(X)   
        return transformed_data
    
    def get_feature_names(self):
        return self.feature_names 


    
class Values_Structure_AllUrl_FeatureTransformer(TransformerMixin):
    def __init__(self, feature_name):
        self.feature_name = feature_name 
        self.feature_names = [feature_name]  # Store feature names during transformation
        self.pipeline = Pipeline(
            [(f'{feature_name}_extractor', FunctionTransformer(
                                            func=extract_values_all_url, 
                                            kw_args={'feature_name': feature_name},
                                            validate=False))])       
        
    def fit(self, X, y=None):
        return self.pipeline.fit(X, y)
    
    def transform(self, X):  
        transformed_data = self.pipeline.transform(X)   
        return transformed_data
    
    def get_feature_names(self):
        return self.feature_names 
    
class Values_Structure_AnyUrl_FeatureTransformer(TransformerMixin):
    def __init__(self, feature_name):
        self.feature_name = feature_name 
        self.feature_names = [feature_name]  # Store feature names during transformation
        self.pipeline = Pipeline(
            [(f'{feature_name}_extractor', FunctionTransformer(
                                            func=extract_values_any_url, 
                                            kw_args={'feature_name': feature_name},
                                            validate=False))])       
        
    def fit(self, X, y=None):
        return self.pipeline.fit(X, y)
    
    def transform(self, X):  
        transformed_data = self.pipeline.transform(X)   
        return transformed_data
    
    def get_feature_names(self):
        return self.feature_names 

        

    
class Values_Structure_AllHasUnderscore_FeatureTransformer(TransformerMixin):
    def __init__(self, feature_name):
        self.feature_name = feature_name 
        self.feature_names = [feature_name]  # Store feature names during transformation
        self.pipeline = Pipeline(
            [(f'{feature_name}_extractor', FunctionTransformer(
                                            func=extract_values_all_has_underscore, 
                                            kw_args={'feature_name': feature_name},
                                            validate=False))])       
        
    def fit(self, X, y=None):
        return self.pipeline.fit(X, y)
    
    def transform(self, X):  
        transformed_data = self.pipeline.transform(X)   
        return transformed_data
    
    def get_feature_names(self):
        return self.feature_names 

   

    
class Values_Structure_AnyHasUnderscore_FeatureTransformer(TransformerMixin):
    def __init__(self, feature_name):
        self.feature_name = feature_name 
        self.feature_names = [feature_name]  # Store feature names during transformation
        self.pipeline = Pipeline(
            [(f'{feature_name}_extractor', FunctionTransformer(
                                            func=extract_values_any_has_underscore, 
                                            kw_args={'feature_name': feature_name},
                                            validate=False))])       
        
    def fit(self, X, y=None):
        return self.pipeline.fit(X, y)

    
    def transform(self, X):  
        # Transform the data and store the feature names
        transformed_data = self.pipeline.transform(X)   
        # print(f'HeaderTokenCountFeature out = \n{transformed_data[0:5]}')     
        return transformed_data
    
    def get_feature_names(self):
        return self.feature_names 


    
class Values_Structure_AllUnique_FeatureTransformer(TransformerMixin):
    def __init__(self, feature_name):
        self.feature_name = feature_name 
        self.feature_names = [feature_name]  # Store feature names during transformation
        self.pipeline = Pipeline(
            [(f'{feature_name}_extractor', FunctionTransformer(
                                            func=extract_values_all_unique, 
                                            kw_args={'feature_name': feature_name},
                                            validate=False))])       
        
    def fit(self, X, y=None):
        return self.pipeline.fit(X, y)

    
    def transform(self, X):  
        # Transform the data and store the feature names
        transformed_data = self.pipeline.transform(X)   
        # print(f'HeaderTokenCountFeature out = \n{transformed_data[0:5]}')     
        return transformed_data
    
    def get_feature_names(self):
        return self.feature_names 

    
class Values_Structure_AnyEmpty_FeatureTransformer(TransformerMixin):
    def __init__(self, feature_name):
        self.feature_name = feature_name 
        self.feature_names = [feature_name]  # Store feature names during transformation
        self.pipeline = Pipeline(
            [(f'{feature_name}_extractor', FunctionTransformer(
                                            func=extract_values_any_empty, 
                                            kw_args={'feature_name': feature_name},
                                            validate=False))])       
        
    def fit(self, X, y=None):
        return self.pipeline.fit(X, y)
    
    def transform(self, X):  
        # Transform the data and store the feature names
        transformed_data = self.pipeline.transform(X)   
        # print(f'HeaderTokenCountFeature out = \n{transformed_data[0:5]}')     
        return transformed_data
    
    def get_feature_names(self):
        return self.feature_names 
    
class Header_Structure_IsUpper_FeatureTransformer(TransformerMixin):
    def __init__(self, feature_name):
        self.feature_name = feature_name 
        self.feature_names = [feature_name]  # Store feature names during transformation
        self.pipeline = Pipeline(
            [(f'{feature_name}_extractor', FunctionTransformer(
                                            func=extract_header_is_upper, 
                                            kw_args={'feature_name': feature_name},
                                            validate=False))])       
        
    def fit(self, X, y=None):
        return self.pipeline.fit(X, y)
    
    def transform(self, X):  
        transformed_data = self.pipeline.transform(X)   
        return transformed_data
    
    def get_feature_names(self):
        return self.feature_names 
    
class Header_Structure_IsLower_FeatureTransformer(TransformerMixin):
    def __init__(self, feature_name):
        self.feature_name = feature_name 
        self.feature_names = [feature_name]  # Store feature names during transformation
        self.pipeline = Pipeline(
            [(f'{feature_name}_extractor', FunctionTransformer(
                                            func=extract_header_is_lower, 
                                            kw_args={'feature_name': feature_name},
                                            validate=False))])       
        
    def fit(self, X, y=None):
        return self.pipeline.fit(X, y)
    
    def transform(self, X):  
        transformed_data = self.pipeline.transform(X)   
        return transformed_data
    
    def get_feature_names(self):
        return self.feature_names 
    
class Header_Structure_IsTitle_FeatureTransformer(TransformerMixin):
    def __init__(self, feature_name):
        self.feature_name = feature_name 
        self.feature_names = [feature_name]  # Store feature names during transformation
        self.pipeline = Pipeline(
            [(f'{feature_name}_extractor', FunctionTransformer(
                                            func=extract_header_is_title, 
                                            kw_args={'feature_name': feature_name},
                                            validate=False))])       
        
    def fit(self, X, y=None):
        return self.pipeline.fit(X, y)
    
    def transform(self, X):  
        transformed_data = self.pipeline.transform(X)   
        return transformed_data
    
    def get_feature_names(self):
        return self.feature_names     
class Header_Structure_ColumnIndex_FeatureTransformer(TransformerMixin):
    def __init__(self, feature_name):
        self.feature_name = feature_name 
        self.feature_names = [feature_name]  # Store feature names during transformation
        self.pipeline = Pipeline(
            [(f'{feature_name}_extractor', FunctionTransformer(
                                            func=extract_structure_column_index, 
                                            kw_args={'feature_name': feature_name},
                                            validate=False))])       
        
    def fit(self, X, y=None):
        return self.pipeline.fit(X, y)

    
    def transform(self, X):  
        # Transform the data and store the feature names
        transformed_data = self.pipeline.transform(X)   
        # print(f'HeaderTokenCountFeature out = \n{transformed_data[0:5]}')     
        return transformed_data
    
    def get_feature_names(self):
        return self.feature_names 
    

    
 
class Values_Structure_AvgCharCount_FeatureTransformer(TransformerMixin):
    def __init__(self, feature_name):
        self.feature_name = feature_name 
        self.feature_names = [feature_name]  # Store feature names during transformation
        self.pipeline = Pipeline(
            [(f'{feature_name}_extractor', FunctionTransformer(
                                            func=extract_values_avg_char_count, 
                                            kw_args={'feature_name': feature_name},
                                            validate=False))])       
        
    def fit(self, X, y=None):
        return self.pipeline.fit(X, y)

    
    def transform(self, X):  
        # Transform the data and store the feature names
        transformed_data = self.pipeline.transform(X)   
        # print(f'HeaderTokenCountFeature out = \n{transformed_data[0:5]}')     
        return transformed_data
    
    def get_feature_names(self):
        return self.feature_names 

        
class Values_Structure_MinCharCount_FeatureTransformer(TransformerMixin):
    def __init__(self, feature_name):
        self.feature_name = feature_name 
        self.feature_names = [feature_name]  # Store feature names during transformation
        self.pipeline = Pipeline(
            [(f'{feature_name}_extractor', FunctionTransformer(func=extract_values_min_char_count, 
                                                    kw_args={'feature_name': feature_name},
                                                    validate=False))])       
        
    def fit(self, X, y=None):
        return self.pipeline.fit(X, y)

    
    def transform(self, X):  
        # Transform the data and store the feature names
        transformed_data = self.pipeline.transform(X)   
        # print(f'HeaderTokenCountFeature out = \n{transformed_data[0:5]}')     
        return transformed_data
    
    def get_feature_names(self):
        return self.feature_names 
    
class Values_Structure_MaxCharCount_FeatureTransformer(TransformerMixin):
    def __init__(self, feature_name):
        self.feature_name = feature_name 
        self.feature_names = [feature_name]  # Store feature names during transformation
        self.pipeline = Pipeline(
            [(f'{feature_name}_extractor', FunctionTransformer(func=extract_values_max_char_count, 
                                                    kw_args={'feature_name': feature_name},
                                                    validate=False))])       
        
    def fit(self, X, y=None):
        return self.pipeline.fit(X, y)

    
    def transform(self, X):  
        # Transform the data and store the feature names
        transformed_data = self.pipeline.transform(X)   
        return transformed_data
    
    def get_feature_names(self):
        return self.feature_names
       
# Custom transformer to extract the minimum token count of 'values'
class Values_Structure_MinTokenCount_FeatureTransformer(TransformerMixin):
    def __init__(self, feature_name):
        self.feature_name = feature_name 
        self.feature_names = [feature_name]  # Store feature names during transformation
        self.pipeline = Pipeline(
            [(f'{feature_name}_extractor', FunctionTransformer(func=extract_values_min_token_count, 
                                                    kw_args={'feature_name': feature_name},
                                                    validate=False))])       
        
    def fit(self, X, y=None):
        return self.pipeline.fit(X, y)

    
    def transform(self, X):  
        # Transform the data and store the feature names
        transformed_data = self.pipeline.transform(X)   
        return transformed_data
    
    def get_feature_names(self):
        return self.feature_names
    
class Values_Structure_MaxTokenCount_FeatureTransformer(TransformerMixin):
    def __init__(self, feature_name):
        self.feature_name = feature_name 
        self.feature_names = [feature_name]  # Store feature names during transformation
        self.pipeline = Pipeline(
            [(f'{feature_name}_extractor', FunctionTransformer(func=extract_values_max_token_count, 
                                                    kw_args={'feature_name': feature_name},
                                                    validate=False))])       
        
    def fit(self, X, y=None):
        return self.pipeline.fit(X, y)

    def transform(self, X):  
        transformed_data = self.pipeline.transform(X)   
        return transformed_data
    
    def get_feature_names(self):
        return self.feature_names 

    
# Custom transformer to extract the avg token count of 'values'
class Values_Structure_AvgTokenCount_FeatureTransformer(TransformerMixin):
    def __init__(self, feature_name):
        self.feature_name = feature_name 
        self.feature_names = [feature_name]  # Store feature names during transformation
        self.pipeline = Pipeline(
            [(f'{feature_name}_extractor', FunctionTransformer(func=extract_values_avg_token_count, 
                                                    kw_args={'feature_name': feature_name},
                                                    validate=False))])       
        
    def fit(self, X, y=None):
        return self.pipeline.fit(X, y)

    def transform(self, X):  
        transformed_data = self.pipeline.transform(X)   
        return transformed_data
    
    def get_feature_names(self):
        return self.feature_names 

                
# Custom transformer to extract 'header_char_count'
class Header_Structure_CharCount_FeatureTransformer(TransformerMixin):
    def __init__(self, feature_name):
        self.feature_name = feature_name 
        self.feature_names = [feature_name]  # Store feature names during transformation
        self.pipeline = Pipeline(
            [(f'{feature_name}_extractor', FunctionTransformer(func=extract_header_char_count, 
                                                    kw_args={'feature_name': feature_name},
                                                    validate=False))])       
        
    def fit(self, X, y=None):
        return self.pipeline.fit(X, y)

    def transform(self, X):  
        transformed_data = self.pipeline.transform(X)   
        return transformed_data
    
    def get_feature_names(self):
        return self.feature_names 
 
# Custom transformer to extract 'header_token_count'
class Header_Structure_TokenCount_FeatureTransformer(TransformerMixin):
    def __init__(self, feature_name):
        self.feature_name = feature_name 
        self.feature_names = [feature_name]  # Store feature names during transformation
        self.pipeline = Pipeline(
            [(f'{feature_name}_extractor', FunctionTransformer(func=extract_header_token_count, 
                                                    kw_args={'feature_name': feature_name},
                                                    validate=False))])       
        
    def fit(self, X, y=None):
        return self.pipeline.fit(X, y)

    def transform(self, X):  
        transformed_data = self.pipeline.transform(X)   
        return transformed_data
    
    def get_feature_names(self):
        return self.feature_names       

# Custom transformer to drop a specific column from the DataFrame
class DropColumnTransformer:
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.column)  
    
# Custom transformer to drop everything but keep_columns from the DataFrame
class KeepColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, keep_columns):
        self.keep_columns = keep_columns

    def fit(self, X, y=None):
        logger.info(f"KeepColumnsTransformer: Fitting with columns: {self.keep_columns}")
        return self

    def transform(self, X):
        logger.info(f"KeepColumnsTransformer: Transforming data")
        
        X_new = X.copy()
        X_new = X_new.loc[:, self.keep_columns]
        return X_new
    
# Custom transformer to select specific columns from the original data
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]
# Custom class for handling preprocessing
class MetaPreprocessingStep:
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        print(X)
        if isinstance(X, pd.DataFrame):
            return X  # Already a DataFrame, no need for conversion
        elif isinstance(X, (list, tuple, np.ndarray)):
            return pd.DataFrame(X)  # Convert to a DataFrame
        else:
            raise ValueError("Input data type not supported. Expected DataFrame, list, tuple, or ndarray.")    
class FeaturePruner(BaseEstimator, TransformerMixin):
    
    def __init__(self, importance_function, threshold=0.5):
        self.importance_function = importance_function
        self.threshold = threshold
        self.selected_indices = None

    def fit(self, X, y=None):
        if self.threshold>0:
            # calculate the feature importance during training.
            feature_importance = self.importance_function(X, y)
            print(f'# features: {len(feature_importance)} ')
            self.selected_indices = [i  for i, imp in enumerate(feature_importance) if imp>= self.threshold]
            print(f'# features remaining : {len(self.selected_indices)} ')
        return self

    def transform(self, X):
        # print(f'FeaturePruner.transform input:\n\n{X}')
        # apply the feature importance threshold and prune the features during prediction.
        if self.selected_indices is None:
            return X
            # raise ValueError("fit method must be called first")
        return X[:, self.selected_indices]    