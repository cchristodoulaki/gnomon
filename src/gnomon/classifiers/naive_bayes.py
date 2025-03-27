import scipy
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion

import logging
logging.basicConfig(level=logging.INFO)

from sklearn.preprocessing import  StandardScaler
from .transformers import KeepColumnsTransformer, SampleWeightTransformer, CategoricalTransformer, FeatureSelector, extract_header, extract_values, get_transformer_class
from sklearn.utils import compute_sample_weight

class NaiveBayesLearner(Pipeline):
    
    def __init__(self,  
                 target_features = [
                    ('header','semantic'),
                    ('values','semantic'),
                    ('header','structure'),
                    ('values','structure')
                    ], 
                 meta_features= {
                    "header":{
                        "semantic":{
                            "tfidf": "[float]: term frequency–inverse document frequency vectors"
                            # ,
                            # "glove": "[float]: GloVe word embedding vectors"
                        },
                        "structure":{
                            "column_index": "int: column index",
                            "char_count": "int: character count", 
                            "token_count": "int: token count",
                            "is_upper": "boolean: is upper case",
                            "is_lower": "boolean: is lower case"
                        }
                    },
                    "values":{
                        "semantic":{
                            "tfidf": "[float]: term frequency–inverse document frequency vectors"
                            # , 
                            # "glove":"[float]: GloVe word embedding vectors"
                        },
                        "structure":{
                            "min_char_count": "int: minimum character count",
                            "max_char_count": "int: maximum character count", 
                            "avg_char_count": "int: average character count",
                            "min_token_count": "int: minimum token count",
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
                }
            ):
        print(f'Naive Bayes Learner target_features:\n{target_features}\n\n')
        self.model = 'naive_bayes'        
        self.target_features = target_features
        self.meta_features = meta_features
        transformer_pipelines = {}     
        if self.meta_features:
            for source, type in self.target_features:
                if source in self.meta_features.keys():
                    for feature_name in self.meta_features[source][type].keys():                                                           
                        transformer_name = f"{source.title()}_{type.title()}_{feature_name.replace('_',' ').title().replace(' ','')}"
                        print(f'{transformer_name=}')
                        transformer_class = get_transformer_class(transformer_name)
                        print(f'transformer_class={transformer_class}')
                        if transformer_class:
                            if type not in transformer_pipelines.keys():                                
                                transformer_pipelines[type] = []
                            transformer_pipelines[type].append((transformer_name, transformer_class(transformer_name)))
        
        combined_feature_list = []                    
        for type in transformer_pipelines.keys():
            sub_pipe_steps = [
                (f'union_type',FeatureUnion(transformer_pipelines[type]))
            ]
            if type == 'structure':
                sub_pipe_steps.append((f'scaler_{type}', StandardScaler(with_mean=False)))
            sub_pipe = Pipeline(sub_pipe_steps)
            combined_feature_list.append((f'unioned_{type}',sub_pipe))
                
        combined_features = FeatureUnion(combined_feature_list) 
        
        # create the classifier
        clf = MultinomialNB()
        
        super().__init__([
            ('drop_columns', KeepColumnsTransformer(['csv_column','table_column'])),
            ('feature_extraction', combined_features),
            ('clf', clf)
        ])
        
    def fit(self, X, y):
        sample_weights = compute_sample_weight('balanced', y = y)
        super().fit(X, y, clf__sample_weight=sample_weights)
        # self.feature_importance = self.calculate_feature_importance(X, y)
        return self    
        
    def calculate_feature_importance(self, X, y):
        transformed_features = self.named_steps['feature_extraction'].transform(X)

        transformer_feature_names = []
        if 'feature_extraction' in self.named_steps:
            for _, feature_type_pipe in self.named_steps['feature_extraction'].transformer_list:        
                for _, transformer_pipeline in feature_type_pipe.named_steps['union_type'].transformer_list:
                    print(f'{transformer_pipeline.get_feature_names()=}')
                    transformer_feature_names.extend(transformer_pipeline.get_feature_names())                    
                    
        dense_transformed_data = transformed_features.toarray()

        transformed_data = pd.DataFrame(data = dense_transformed_data, 
                                        columns = transformer_feature_names)

        # Access the trained Random Forest classifier
        classifier = self.named_steps['clf']

        # Fit the Random Forest classifier
        classifier.fit(transformed_data, y)

        # Access the feature importances
        feature_importances = classifier.feature_importances_        

        importance_df = pd.DataFrame(
            {
                'feature':transformer_feature_names, 
                'importance':feature_importances}
            ).sort_values(by='importance', ascending=False)
        return importance_df
    
    def calculate_feature_importance_per_class(self, X, y):
        
        transformed_features = self.named_steps['feature_extraction'].transform(X)
        transformer_feature_names = []
        
        if 'feature_extraction' in self.named_steps:
            for _, transformer_pipeline in self.named_steps['feature_extraction'].transformer_list:
                print(f'{transformer_pipeline=}')
                transformer_feature_names.extend(transformer_pipeline.get_feature_names())
                
        dense_transformed_data = transformed_features.toarray()
        transformed_data = pd.DataFrame(data=dense_transformed_data, 
                                        columns=transformer_feature_names, 
                                        index=X.index)

        feature_extraction = self.named_steps['feature_extraction']
        
        # Access the trained Random Forest classifier
        classifier = self.named_steps['clf']
        
        # Fit the Random Forest classifier
        classifier.fit(feature_extraction.transform(transformed_data.values), y)
        
        # Get the classes present in the target variable
        unique_classes = np.unique(y)

        # Create a dictionary to store feature importances per class
        importance_per_class = {}

        # Calculate feature importance for each class
        for class_label in unique_classes:
            # Create binary labels indicating whether a sample belongs to the current class
            binary_labels = (y == class_label).astype(int)
            
            # Fit the Random Forest classifier on the entire transformed dataset
            classifier.fit(feature_extraction.transform(transformed_data.values), binary_labels)            
            
            # Access the feature importances for the current class
            class_feature_importances = classifier.feature_importances_
            importance_df = pd.DataFrame(
                {
                    'feature':[name for i, name in enumerate(transformer_feature_names) if i in feature_extraction.selected_indices], 
                    'importance':class_feature_importances
                    }
                ).sort_values(by='importance', ascending=False)
            importance_df.set_index('feature', inplace=True)
            importance_per_class[class_label] = importance_df

        return importance_per_class    