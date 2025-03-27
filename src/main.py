import pandas as pd
from gnomon.schema_mapping import GnomonMapper
from gnomon.matching_algorithms.naive_bayes_matching import NaiveBayesMatcher
from gnomon.matching_algorithms.ensemble_matching import GnomonMatcher
from config import config, metadata_model
import os
from pprint import pprint
# Get the path of the main.py file
current_dir = os.path.dirname(__file__)

# Instantiate custom matcher
print("Select a matcher:")
print("1. Naive Bayes Matcher")
print("2. kNN Matcher")
print("3. Random Forest Matcher")
print("4. Gnomon Matcher")

choice = input("Enter the number of the matcher you want to use: ")

if choice == '1':
    matcher = NaiveBayesMatcher(config["naive_bayes"])
elif choice == '2':
    matcher = NaiveBayesMatcher(config["naive_bayes"])
elif choice == '3':
    matcher = NaiveBayesMatcher(config["naive_bayes"])
elif choice == '4':
    matcher = GnomonMatcher(config["gnomon"])
else:
    raise ValueError("Invalid choice")

train_data_path = input("Enter the path to the training data CSV file or press Enter to skip: ")

if train_data_path:
    # Construct the relative path to the Ground Truth Dev datasets folder
    datasets_path = os.path.join(os.path.dirname(__file__), '..', 'gnomon_datasets')
    train_data_path = os.path.join(datasets_path, 'dev_column_mappings.csv') 
    if not os.path.isfile(train_data_path):
        raise ValueError("The provided path is not a valid file.")
    train_data_df = pd.read_csv(train_data_path)
    # Train the matcher with custom data
    dev_mapping_df = pd.read_csv(train_data_path)

    X_df = dev_mapping_df.iloc[:, :-1]
    y_df = dev_mapping_df.iloc[:, -1]

    matcher.train(X_df, y_df) # train with ground truth data in DEV
    
else:
    matcher.load_pretrained() # TODO: Implement this method


# Use GnomonMapper with the custom classifier  
mapper = GnomonMapper(matcher=matcher)

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

mappings = mapper.map_schemas(source_df, target_df)
pprint(mappings[0])