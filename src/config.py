config = {
    'naive_bayes':{
        "matcher":"NaiveBayesMatcher",
        "target_features":[("header","semantic"),("values","semantic")]
        },
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

metadata_model=[
    {"Name": "dataset",
    "Description": "The name of the file the described attribute exists in."},
    {"Name": "name",
    "Description":"[mandatory] The attribute name, as seen in the data table."},
    {"Name": "title",
    "Description": "A human-readable version of the attribute name."},
    {"Name": "definition",
    "Description": "A text description of the attribute, may include context."},
    {"Name": "datatype",
    "Description":"Specifies which type of value the attribute can have."},
    {"Name": "scale",
    "Description":"For numeric attributes, the multiplier (e.g., millions)."},
    {"Name": "format",
    "Description":"A definition of the structure of data (e.g., ‘YYYY-MM-DD’)."},
    {"Name": "unit",
    "Description": "The unit of the associated values (e.g., meters, KWh)."},
    {"Name": "key",
    "Description": "An indication that the attribute is a primary key."},
    {"Name": "nullable",
    "Description":"An indication that an attribute value is or isn’t mandatory."},
    {"Name": "schema",
    "Description": "Name/URL of an existing schema the attribute belongs to."},
    {"Name": "examples",
    "Description": "Sampled values from the attribute domain."},
    {"Name": "notes",
    "Description": "Text with miscellaneous extra information"},
    {"Name": "UNMAPPED",
    "Description": "Undefined metadata field"}
]