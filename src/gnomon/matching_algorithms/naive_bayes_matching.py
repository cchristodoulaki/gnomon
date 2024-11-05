
import pandas as pd
import numpy as np
from sklearn.utils import compute_sample_weight
from typing import List

#relative imports
from .base_matcher import BaseMatcher
from gnomon.model import TableColumn
from gnomon.classifiers.naive_bayes import NaiveBayesLearner
from gnomon.utils import dict_table_headers

class NaiveBayesMatcher(BaseMatcher):
    def __init__(self, config):
        self.maximize = True # Naive Bayes uses predict_proba in the alignment matrix        
        self.matcher_measure = 'probability'
        self.model = NaiveBayesLearner(
                    target_features=config["target_features"]
                ) 
            
    def train(self, X_train, y_train):
        self.model.fit( X_train, y_train)
    
    def explain(self):
        pass
        # col_transformer_obj = self.model.named_steps['column_transformer_pipeline']
        # col_transformer_obj
        # Tfidf_transformer = col_transformer_obj.named_transformers_['tfidf_vectorizer']
        # print(f'\nVocabulary (unique strings) in our data:\n{Tfidf_transformer.get_feature_names_out()}\n')
        
    def get_matches(self, source_name, source_df, target_schema):
        source_headers =  {i:h for i,h in enumerate(source_df.columns)}  
        data_df =  source_df.copy()
        data_df.columns = range(len(data_df.columns))
        alignment_df = self.generate_similarity_matrix(source_headers, data_df, target_schema)
        return alignment_df
        
    def generate_similarity_matrix(self, source_headers, data_df, target_schema):

        data = list()
        for key, value in source_headers.items():
            table_column = TableColumn(
                tablecolumn_key = None, 
                csv_column = key, 
                datatable_key = None,
                datafile_key = None,
                header = value,
                values = data_df[key].to_list()
                )
            row = [key, table_column.toJson()]
            data.append(row)

        df_X = pd.DataFrame(data=data, columns=["csv_column","table_column"] )
        ###
        predictions = self.model.predict_proba(df_X)
        matrix = pd.DataFrame(predictions, columns=self.model.classes_, index=df_X["csv_column"].values )
        for field in target_schema:
            if field not in self.model.classes_:
                matrix.loc[:, field]=np.NaN
                
        # print(f'matrix.columns={matrix.columns}')
        
        # the order of columns (target fields) in the matrix is important
        ordered_col = [value for value in target_schema if value in matrix.columns]
        ordered_col.append('UNMAPPED')
        matrix = matrix[ordered_col]
        
        # print('naive_bayes > generate_similarity_matrix:')
        # print(matrix)
        
        # keep only target_schema columns
        # matrix = matrix[[value for value in target_schema if value in matrix.columns]]
        return matrix
        
              