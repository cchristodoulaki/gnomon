import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier 
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

from .naive_bayes import NaiveBayesLearner
from .random_forest import RandomForestLearner
from .kNN import kNNLearner
from .transformers import  ColumnSelector, MetaPreprocessingStep, KeepColumnsTransformer, extract_header, extract_values, get_transformer_class
       
class EnsembleLearner(StackingClassifier):
    def __init__(self, config):
        self.config=config
        print(f'EnsembleLearner config:\n{self.config}')
        

        self.final_estimator = LogisticRegression(
                penalty='l2',# default choice;
                dual=False,
                tol=1e-4,
                C=1.0,
                fit_intercept=True, # Specifies if a bias should be added to the decision function.
                class_weight='balanced', # the model automatically assigns the class weights inversely proportional to their respective frequencies.
                solver='liblinear', # good for smaller datasets
                random_state=42,
                multi_class='ovr', # one versus rest
                max_iter=self.config["final_estimator_params"]["max_iter"],
                verbose=0,
                n_jobs=-1
            )
        super().__init__(#StackingClassifier
            estimators=self.create_base_learners(),
            final_estimator=self.final_estimator,
            # When passthrough=False, only the predictions of estimators 
            # will be used as training data for final_estimator. 
            # When True, the final_estimator is trained on the predictions 
            # as well as the original training data.
            passthrough=self.config["passthrough"],
            verbose=0,
            cv = self.config["cv"],
            stack_method='predict_proba',
            n_jobs=-1
        )
        
                            
    def create_base_learners(self):
        print(f'\n\nconfig=\n{self.config}')
        base_learners = list()
        for learner, learner_arg in self.config["base_learners"].items():
            if learner == 'NaiveBayes':
                base_learners.append(
                    (learner, NaiveBayesLearner(**learner_arg) 
                ))
            elif learner == 'RandomForest':
                base_learners.append(
                    (learner, RandomForestLearner(**learner_arg) 
                ))
            elif learner == 'kNearestNeighbors':
                print(f"\n\n\n{learner_arg['kwargs']}")
                base_learners.append(
                    (learner, kNNLearner(learner_arg['target_features'], **learner_arg['kwargs']) 
                ))                        
            else:
                raise ValueError(f'Unknown learner: {learner}')
        return base_learners    
    
