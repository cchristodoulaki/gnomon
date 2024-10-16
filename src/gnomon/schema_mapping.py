import pandas as pd



class GnomonMapper:
    def __init__(self, source_df: pd.DataFrame, target_df: pd.DataFrame, matching_technique: str = 'classification', classifier='Stacking'):
        self.source_df = source_df
        self.target_df = target_df
        self.matching_technique = matching_technique
        self.classifier = classifier  # External classifier, passed as parameter

    def find_mapping(self) -> dict:
        mapping = {}
        for source_col in self.source_df.columns:
            best_match = None
            best_score = float('-inf')
            for target_col in self.target_df.columns:
                score = self._calculate_score(source_col, target_col)
                if score > best_score:
                    best_score = score
                    best_match = target_col
            mapping[source_col] = best_match
        return mapping