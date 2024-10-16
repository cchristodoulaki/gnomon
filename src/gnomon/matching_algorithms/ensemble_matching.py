
from typing import List
import json
# from matching_algorithms.periplous_matching.base_learners.kNN import kNNLearner
# from matching_algorithms.periplous_matching.base_learners import NaiveBayesLearner
# from matching_algorithms.periplous_matching.base_learners.RandomForestLearner import RandomForestLearner
from gnomon.classifiers.stacking import EnsembleLearner

import pandas as pd
import numpy as np
from gnomon.model import TableColumn
from .base_matcher import BaseMatcher
from gnomon.utils import dict_table_headers

from sklearn.utils import compute_sample_weight

import joblib

""" 
This class implements the Gnomon matching algorithm.
Match weights are calculated with a StackingEnsemble model.
"""

class GnomonMatcher(BaseMatcher):
       
    def __init__(self, config):
        self.name='Gnomon'
        self.matcher_measure = 'probability'
        self.config = config   
        print(f'GnomonMatcher config:\n{self.config}')                                            
        self.maximize = True 
        self.model = EnsembleLearner(self.config)
          
    # # save trained model to file
    # def save(self, path):
    #     joblib.dump(self.model, path)
    # # load saved model from file
    # def load(self, path):    
    #     self.model = joblib.load(path)
    #     print(f'Loaded model from {path}')
    
    def train():
        print('Load pretrained model')
        pass
        
    def train(self, X_train, y_train, class_weight = 'balanced'):
        # sample_weights = compute_sample_weight('balanced', y = y_train)
        # for learner in self.base_learners:
        #     if learner=='header_nb':
        #         self.model.set_params(header_nb__nb_clf__sample_weight=sample_weights)
        #     elif learner=='values-nb':
        #         self.model.set_params(values_nb__nb_clf__sample_weight=sample_weights)
        self.model.fit( X_train, y_train)
        
    def get_matches(self, source_name, data_df, header_df, target_schema):
        source_headers = dict_table_headers(data_df, header_df)  
        alignment_df = self.generate_similarity_matrix(source_headers, data_df, target_schema)
        return alignment_df
        
    def generate_similarity_matrix(self, source_headers, data_df, target_schema):
        # print(f'generate_similarity_matrix:')        
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
        predictions = self.model.predict_proba(df_X)
        matrix = pd.DataFrame(predictions, columns=self.model.classes_, index=df_X["csv_column"].values )
        for field in target_schema:
            if field not in self.model.classes_:
                matrix.loc[:, field]=np.NaN                
        
        # the order of columns (target fields) in the matrix is important
        ordered_col = [value for value in target_schema if value in matrix.columns]
        ordered_col.append('UNMAPPED')
        matrix = matrix[ordered_col]
        
        # keep only target_schema columns
        # matrix = matrix[[value for value in target_schema if value in matrix.columns]]
        return matrix    

