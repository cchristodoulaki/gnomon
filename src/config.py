config = {
    'gnomon': {
        "matcher":"StackingMatcher",
        "algorithm": "stacking",
        "base_learners":{
            "NaiveBayes":{
                "target_features":[("header","semantic"),("values","semantic")]
                },
            "kNearestNeighbors":{
                "kwargs": {
                       "n_neighbors":3
                   } ,
                "target_features":[("header","semantic"),("values","semantic")]
                },
            "RandomForest":{
                "n_trees":100,
                "target_features":[("header","semantic"),("header","structure")]
                }
        },
        "meta_learner":{
            "classifier":  "LogisticRegression"
        },
        "cv":5,
        "passthrough":False,
        "final_estimator_params":{
            "max_iter":100
        }  
    }
}