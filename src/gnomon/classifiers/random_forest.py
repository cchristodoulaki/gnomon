from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from .transformers import FeaturePruner, get_transformer_class, get_transformer_function, KeepColumnsTransformer,DropColumnTransformer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import StandardScaler
import scipy

class RandomForestLearner(Pipeline):
    def __init__(self,  
            target_features = [
                            ('header','semantic'),
                            ('values','semantic'),
                            ('header','structure')
                        ],
            n_trees = 100, 
            meta_features = {
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
                }
            ):    
    # This class constructor defines a pipeline object with a Random Forest Classifier
        self.n_trees = n_trees
        # print(f'{self.n_trees=}')
        self.target_features = target_features
        # print(f'{self.target_features=}')
        self.meta_features = meta_features
        # print(f'{self.meta_features=}')
        self.feature_importance = None
        
        clf = RandomForestClassifier(
                    random_state=42,
                    n_estimators=self.n_trees, # default (The number of trees in the forest.)
                    criterion="gini", # default (The function to measure the quality of a split. 
                    max_depth=None, # default (The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.)

                    min_samples_split=2, # default (The minimum number of samples required to split an internal node)
                    min_samples_leaf = 1, # default
                    min_weight_fraction_leaf=0.0, # default (The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node. Samples have equal weight when sample_weight is not provided.)
                    max_features="sqrt", # default (The number of features to consider when looking for the best split)
                    max_leaf_nodes=None, # default (Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.)
                    min_impurity_decrease=0.0, # default (A node will be split if this split induces a decrease of the impurity greater than or equal to this value.)

                    
                    bootstrap =True, # default (Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.)
                    oob_score=False, # default (Whether to use out-of-bag samples to estimate the generalization accuracy. By default, accuracy_score is used.)
                    n_jobs=None, # default (The number of jobs to run in parallel. fit, predict, decision_path and apply are all parallelized over the trees. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details.)


                    verbose=0, # default (Controls the verbosity when fitting and predicting.)

                    warm_start=False, #default  (When set to True, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just fit a whole new forest. See Glossary and Fitting additional weak-learners for details.)


                    class_weight="balanced", # default=None (Weights associated with classes in the form {class_label: weight}. If not given, all classes are supposed to have weight one. For multi-output problems, a list of dicts can be provided in the same order as the columns of y.

                    # Note that for multioutput (including multilabel) weights should be defined for each class of every column in its own dict. For example, for four-class multilabel classification weights should be [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of [{1:1}, {2:5}, {3:1}, {4:1}].

                    # The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))

                    # The “balanced_subsample” mode is the same as “balanced” except that weights are computed based on the bootstrap sample for every tree grown.

                    # For multi-output, the weights of each column of y will be multiplied.

                    # Note that these weights will be multiplied with sample_weight (passed through the fit method) if sample_weight is specified.

                    ccp_alpha=0.0, # default=0.0 (Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen. By default, no pruning is performed. See Minimal Cost-Complexity Pruning for details.

                    max_samples=None # default=None (If bootstrap is True, the number of samples to draw from X to train each base estimator.
                    # If None (default), then draw X.shape[0] samples.
                    # If int, then draw max_samples samples.
                    # If float, then draw max(round(n_samples * max_samples), 1) samples. Thus, max_samples should be in the interval (0.0, 1.0].
                    )       
        
        transformer_pipelines = {}     
        if self.meta_features:
            for source, type in self.target_features:
                if source in self.meta_features.keys():
                    for feature_name in self.meta_features[source][type].keys():                                                           
                        transformer_name = f"{source.title()}_{type.title()}_{feature_name.replace('_',' ').title().replace(' ','')}"
                        transformer_class = get_transformer_class(transformer_name)
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
                sub_pipe_steps.append((f'{type}_scaler', StandardScaler(with_mean=False)))
            sub_pipe = Pipeline(sub_pipe_steps)
            combined_feature_list.append((f'unioned_{type}',sub_pipe))
                
        combined_features = FeatureUnion(combined_feature_list)          
        
        
        # Create the final pipeline
        super().__init__([
            ('drop_columns', KeepColumnsTransformer(['csv_column','table_column'])),
            ('feature_extraction', combined_features),
            ('clf', clf)
        ])

    def fit(self, X, y):
        super().fit(X, y)
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
    

def main():
    df = pd.read_csv('GT_data.csv')
    # # Splitting data into features and labels
    X = df.copy()
    y = df['target']

    # Set a fixed random seed value for reproducibility
    random_seed = 42
    np.random.seed(random_seed)
    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)
    
    # Creating the RF_Learner instance
    rf_learner = RandomForestLearner(target_features=["structure","header","values"])

    # Training the model
    rf_learner.fit(X_train, y_train)

    # Making predictions on the test set
    y_pred = rf_learner.predict(X_test)

    # Evaluating the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print(rf_learner.feature_importance(X_train, y_train))

if __name__ == "__main__":
    main()
         