import pandas as pd
from gnomon.schema_mapping import GnomonMapper
from gnomon.matching_algorithms.naive_bayes_matching import NaiveBayesMatcher
from gnomon.matching_algorithms.ensemble_matching import GnomonMatcher
from config import config, metadata_model
import os

# Get the path of the main.py file
current_dir = os.path.dirname(__file__)

# Construct the relative path to the datasets folder
source_path = os.path.join(current_dir, '..', 'gnomon_datasets', 'test/test_1.csv')
# Extract the file name
file_name = os.path.basename(source_path)
# Instantiate dataframes for the mapping
# Read the CSV with custom handling for missing headers
source_df = pd.read_csv(source_path, header=0)
# Assign a name to the DataFrame
source_df.name = "file_name"

# Check for "Unnamed" headers, which indicate missing values in the first line
if all(col.startswith("Unnamed") for col in source_df.columns):
    source_df.columns = ['' for _ in source_df.columns]  # Replace with empty strings as headers
    
target_df = pd.DataFrame(columns=[field["Name"] for field in metadata_model])
target_df.name = "metadata_model"

print("Source DataFrame:\n", source_df)
print("Target DataFrame:\n", target_df)


# Instantiate custom matcher
matcher = NaiveBayesMatcher(config["naive_bayes"])
# matcher = GnomonMatcher(config["gnomon"])
# matcher.train() # load pretrained
# Train the matcher with custom data
# Construct the relative path to the datasets folder
datasets_path = os.path.join(os.path.dirname(__file__), '..', 'gnomon_datasets')
dev_mapping_path = os.path.join(datasets_path, 'dev_column_mappings.csv')
dev_mapping_df = pd.read_csv(dev_mapping_path)
#all columns except the last one
X_df = dev_mapping_df.iloc[:, :-1]
#only last column
y_df = dev_mapping_df.iloc[:, -1]
# input(X_df.head())
# input(y_df.head())

matcher.train(X_df, y_df) # train with custom data





# Use GnomonMapper with the custom classifier
mapper = GnomonMapper()
mapping = mapper.find_mapping(source_df, target_df, matcher)

print("Mapping:", mapping)