import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.neighbors import KNeighborsClassifier

from .transformers import  KeepColumnsTransformer, get_transformer_class
from gnomon.language_models.GloVe import glove_util

import numpy as np

class kNNLearner(Pipeline):
    def __init__(self,  
            target_features = [
                            ('header','semantic'),
                            ('values','semantic')
                        ], 
            meta_features= {
                    "header":{
                        "semantic":{
                            "tfidf":"[float]:term frequency–inverse document frequency vectors"
                            # ,
                            # "glove":"[float]:GloVe word embedding vectors"
                        },
                        "structure":{
                            "column_index":"int: column index",
                            "char_count":"int: character count", 
                            "token_count": "int: token count",
                            "is_upper": "boolean: is upper case",
                            "is_lower": "boolean: is lower case"
                        }
                    },
                    "values":{
                        "semantic":{
                            "tfidf":"[float]: term frequency–inverse document frequency vectors"
                            # , 
                            # "glove":"[float]: GloVe word embedding vectors"
                        },
                        "structure":{
                            "min_char_count":"int: minimum character count",
                            "max_char_count":"int: maximum character count", 
                            "avg_char_count":"int: average character count",
                            "min_token_count":"int: minimum token count",
                            "max_token_count":"int: maximum token count",
                            "avg_token_count":"int: average token count", 
                            "all_lower":"boolean: all values are lower case",
                            "all_upper":"boolean: all values are upper case", 
                            "all_title":"boolean: all values are title case",
                            "all_url":"boolean: all values are URLs", 
                            "any_url":"boolean: any value is a URL",
                            "all_has_underscore":"boolean: all values have underscores", 
                            "any_has_underscore":"boolean: any value has an underscore",
                            "all_unique":"boolean: all values are unique",
                            "any_empty":"boolean: any value is empty"
                            # ,"any_missing":"boolean: any value is missing"
                            # ,"all_missing":"boolean: all values are missing"
                            # ,"consistent_datatype":"boolean: all values have the same datatype"
                            # ,"consistent_length":"boolean: all values have the same length"
                        }
                    }
                },
            n_neighbors=3, 
            emb_type= None, 
            semantic='tfidf'):
        
        self.target_features = target_features
        self.meta_features = meta_features
        self.n_neighbors = n_neighbors
        self.emb_type = emb_type
        self.semantic = semantic
        if self.emb_type is not None:
            self.embeddings =  glove_util.load_glove_model('/home/christina/git/Periplous/src/opendata_periplous/language_models/GloVe/glove.6B.100d.txt')
            print(f'{len(self.embeddings)=}')
        if self.semantic=='glove':
            for input in self.meta_features.keys():
                if 'semantic' in self.meta_features[input].keys():
                    self.meta_features[input]['semantic'] = {"glove":"[float]: GloVe word embedding vectors"}
                    self.embeddings =  glove_util.load_glove_model('/home/christina/git/Periplous/src/opendata_periplous/language_models/GloVe/glove.6B.100d.txt')
                
        print('\n\n--- CREATING FEATURES ---')
        transformer_pipelines = {}     
        if self.meta_features:
            for source, type in self.target_features:
                print(f'{source=}, {type=}')
                if source in self.meta_features.keys():
                    if self.emb_type is None:
                        for feature_name in self.meta_features[source][type].keys():                                                           
                            transformer_name = f"{source.title()}_{type.title()}_{feature_name.replace('_',' ').title().replace(' ','')}"
                            print(f'{transformer_name=}')
                            transformer_class = get_transformer_class(transformer_name)
                            if transformer_class:
                                if type not in transformer_pipelines.keys():                                
                                    transformer_pipelines[type] = []
                                if 'glove' in transformer_name.lower():
                                    transformer_pipelines[type].append((transformer_name, transformer_class(transformer_name, self.embeddings)))
                                else:
                                    transformer_pipelines[type].append((transformer_name, transformer_class(transformer_name)))
                                
                    else:# handle embeddings
                        transformer_name = f"{source.title()}_{type.title()}_{self.emb_type.replace('_',' ').title().replace(' ','')}"
                        print(f'{transformer_name=}')
                        transformer_class = get_transformer_class(transformer_name)
                        if transformer_class:
                            print(f'{transformer_class=}')
                            if type not in transformer_pipelines.keys():                                
                                transformer_pipelines[type] = []
                            transformer_pipelines[type].append((transformer_name, transformer_class(transformer_name, self.embeddings)))
       
        print('\ntransformer_pipelines=')
        print(transformer_pipelines)
        
        print('\n\n--- COMBINING FEATURES ---')
        combined_feature_list = []                    
        for type in transformer_pipelines.keys():
            sub_pipe_steps = [
                (f'union_{type}',FeatureUnion(transformer_pipelines[type]))
            ]
            if type == 'structure':
                sub_pipe_steps.append((f'scaler_{type}', StandardScaler(with_mean=False)))
            sub_pipe = Pipeline(sub_pipe_steps)
            combined_feature_list.append((f'unioned_{type}',sub_pipe))
                
        combined_features = FeatureUnion(combined_feature_list) 
        
        print(f'combined_features=')
        print(combined_features) 
               
        clf = KNeighborsClassifier(
                    metric='cosine',
                    weights='distance', #weigh points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.
                    n_neighbors=self.n_neighbors, 
                    algorithm='auto', # default
                    n_jobs=-1 # use all processors
                    )
        print('FINAL PIPELINE ---\n')
        # Create the final pipeline
        super().__init__([
            ('drop_columns', KeepColumnsTransformer(['csv_column','table_column'])),
            ('feature_extraction', combined_features),
            ('clf', clf)
        ])
        
