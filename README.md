# UNDER CONSTRUCTION
I'm still undergoing code cleanup, stay tuned!

## Gnomon datasets
Explore our annotated datasets at [https://github.com/cchristodoulaki/gnomon_datasets/](https://github.com/cchristodoulaki/gnomon_datasets/).

## Gnomon
Attribute Metadata Unification in Open Data documentation

### Example Usage

```python
import pandas as pd
from src.gnomon import GnomonMapper
from src.gnomon.matching_algorithms.ensemble_matching import GnomonMatcher

# Instantiate custom matcher
matcher = GnomonMatcher(config)
matcher.train()# load pretrained
# matcher.train(X_df, y_df)


# Instantiate dataframes for the mapping
source_df = pd.DataFrame(...)  # Replace with actual data
target_df = pd.DataFrame(...)

# Use GnomonMapper with the custom classifier
mapper = GnomonMapper()
mapping = mapper.find_mapping(source_df, target_df, matcher)

print("Mapping:", mapping)
```

