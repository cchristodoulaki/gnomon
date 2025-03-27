import pandas as pd
import numpy as np
from gnomon.matching_algorithms.base_matcher import BaseMatcher,AutoColumnMatching
import time
from scipy.optimize import linear_sum_assignment
from gnomon.alignment.WbBm import WBbM


# # Concrete implementation for a classification-based matcher
# class ClassificationMatcher(BaseMatcher):
#     def __init__(self, model):
#         self.model = model

#     def get_matches(self, schema1, schema2):
        
#         # Implement matching logic using a classification model
#         pass

# # Factory to create matcher instances
# class MatcherFactory:
#     @staticmethod
#     def get_matcher(type, **kwargs):
#         if type == 'classification':
#             matcher = kwargs.get('matcher')
#             return ClassificationMatcher(matcher)
#         # elif type == 'valentine':
#         #     method = kwargs.get('method')
#         #     return ValentineMatcher()
#         else:
#             raise ValueError(f"Unknown matcher type: {type}")

# Main class to perform schema mapping
class GnomonMapper:
    # def __init__(self, matcher_type, **kwargs):
    #     self.matcher = MatcherFactory.get_matcher(matcher_type, **kwargs)
    
    def __init__(self, matcher, **kwargs):
        self.matcher = matcher
        
    def map_schemas(self, source_df, target_df, alignment="WbBM"):
        
        # Check for "Unnamed" headers, which indicate missing values in the first line
        if all(col.startswith("Unnamed") for col in source_df.columns):
            source_df.columns = ['' for _ in source_df.columns]  # Replace with empty strings as headers
        source_schema =  {i:h for i,h in enumerate(source_df.columns)}
        start = time.time()
        match_matrix =  self.matcher.get_matches(source_df, target_df)
        end = time.time()
        matching_time = end - start
        print(f'match_matrix={match_matrix}')
        print(f'matching_time={matching_time}')
        # TODO: Update for this repo
        start = time.time()
        mapping = resolve_alignment(
            match_matrix, 
            source_schema, 
            self.matcher.matcher_measure,
            alignment_policy = alignment,
            maximize = self.matcher.maximize
        )            
        end = time.time()
        alignment_time = end - start
        # print(alignment_matrix)
        return mapping, match_matrix, source_schema, matching_time, alignment_time
        
# TODO: update for this repo    
def resolve_alignment(original_df, source_headers, matcher_measure, alignment_policy, maximize=True):
    """
    Resolve alignment based on the alignment policy.
    alignment policies:
     - naive:  assigns each source column first alignment with best match
     - sum_alignment: treats alignment as a 
                    maximum weighted bipartite matching problem (1-1 constraint), 
                    uses the Hungarian algorithm and scipy implementation      
     - greedy: giving precedence to left most attributes 
               greedily selects alignment from available candidates 
               with maximum weight (1-1 constraint)
     - WbBM: weight bipartite B-matching
            maximizes weight of alignments but allows for vertices with B capacity (N-M alignment)
            adapted from https://github.com/tbabm93/Bipartite_b_matching
    """
    matches={}
    alignment_matrix = original_df.copy(deep=True)
    original_row_labels = original_df.index
    original_col_labels = original_df.columns
    
    print(f"base_matcher > resolve_alignment > alignment_matrix:\n\n{alignment_matrix}") #row labels are csv_column_idxs, col labels are fields
        
    alignment_matrix.replace([np.inf, -np.inf],np.nan, inplace = True)    
    alignment_matrix.dropna(axis=0, how="all", inplace = True)
    alignment_matrix.dropna(axis=1, how="all", inplace = True)  
    
    all_fields = alignment_matrix.columns
    print(f'target fields = {all_fields}') 
    print(f'alignment_matrix.shape={alignment_matrix.shape}')

    if not alignment_matrix.empty:
        if alignment_policy=="WbBM":
            selected_edges = list()
            mtrx_cells=list()
            cells=list()
            
            threshold = 0
            num_left, num_right = alignment_matrix.shape #rows (attributes), columns (fields)
            # W = alignment_matrix.values.tolist()
            W = alignment_matrix.to_numpy()
            # print(f'\nW={W}')
            left=range(0,num_left)
            right=range(0,num_right)
            # print(f'\nleft={left}, right={right}\n')
            # print(f'num_left ({alignment_matrix.index})={num_left}, num_right ({alignment_matrix.columns})={num_right}\n')

            ldp = 0 # minimum fields matched to a source attribute SHOULD BE 1?
            udp = 1 # maximum fields matched to a source attribute
            
            lda = [ 1 if metadata_field.lower()=='name' else 0  for metadata_field in alignment_matrix.columns ]
            uda = [ len(alignment_matrix.columns)-1 if metadata_field.lower() == 'unmapped' else 1 for metadata_field in alignment_matrix.columns ]
            
                    
            # print(f'----\nlda={lda}, \nuda = {uda}, \nldp={ldp}, \nudp={udp}\n----')
            # print(f'type(ldp)= {type(ldp).__name__}')
            # print(f'type(udp)= {type(udp).__name__}')
            
            b_matching = WBbM(num_left, num_right, [j for j in list(np.concatenate(W))] , lda, uda, ldp, udp, LogToConsole=0)       
            results, total_weight = b_matching.Bb_matching(optimization_mode = "max")

            # print(f'results={results}')
            # print(f'total_weight={total_weight}')
            
            for row_index in range(len( results )):
                for column_index in range(len( results[row_index] )):
                    if results[row_index][column_index] == 1:
                        selected_edges.append( (list(right)[column_index], list(left)[row_index]) ) # the order based on the gold-standard
                        mtrx_cells.append((list(left)[row_index], list(right)[column_index] ))
                        
            print("\nSelected edges are:", selected_edges, "\nTotal weight:", total_weight)
            
            for row, col in mtrx_cells:                
                original_col_idx = np.where(original_col_labels == alignment_matrix.columns[col])[0][0]                
                original_row_idx = np.where(original_row_labels == alignment_matrix.index[row])[0][0] 
                cells.append((original_row_idx, original_col_idx))
                csv_column = alignment_matrix.index[row]                
                match = AutoColumnMatching(
                    field = alignment_matrix.columns[col],
                    csv_column = csv_column,
                    extracted_name = source_headers[csv_column],
                    measure = matcher_measure,
                    metric_value = alignment_matrix.iloc[row,col],
                    matrix_row = original_row_idx,
                    matrix_col = original_col_idx          
                    )                                    
                    
                matches[str(csv_column)] = match        
            
        elif alignment_policy=="naive":
            mtrx_row = -1
            cells=[]
            for csv_column, row in alignment_matrix.iterrows():
                mtrx_row+=1
                if maximize:
                    field = row.idxmax(skipna=True)
                else:
                    field = row.idxmin(skipna=True)     
                        
                mtrx_col = np.where(all_fields == field)[0][0]
                original_col_idx = np.where(original_col_labels == field)[0][0]
                original_row_idx = np.where(original_row_labels == alignment_matrix.index[mtrx_row])[0][0]
                
                match = AutoColumnMatching(
                            field = field,
                            lang = 'eng',
                            csv_column = csv_column,
                            extracted_name = source_headers[csv_column],
                            measure = matcher_measure,
                            metric_value = alignment_matrix.loc[csv_column, field],
                            matrix_row = original_row_idx,  
                            matrix_col = original_col_idx
                            )
                cells.append((original_row_idx, original_col_idx ))
                matches[str(csv_column)] = match   

        elif alignment_policy == "sum_assignment":
            cells = []
            if maximize:
                alignment_matrix.replace(np.nan, -np.inf, inplace = True)
            else:
                alignment_matrix.replace(np.nan, np.inf, inplace = True)
            try:    
                row_idx, col_idx = linear_sum_assignment(alignment_matrix.to_numpy(), maximize=maximize)
                mtrx_cells = [ (row_idx[i], v) for i, v in enumerate(col_idx) ]             
                
                for row, col in mtrx_cells:                
                    original_col_idx = np.where(original_col_labels == alignment_matrix.columns[col])[0][0]                
                    original_row_idx = np.where(original_row_labels == alignment_matrix.index[row])[0][0] 
                    cells.append((original_row_idx, original_col_idx))
                    csv_column = alignment_matrix.index[row]                
                    match = AutoColumnMatching(
                        field = alignment_matrix.columns[col],
                        csv_column = csv_column,
                        extracted_name = source_headers[csv_column],
                        measure = matcher_measure,
                        metric_value = alignment_matrix.iloc[row,col],
                        matrix_row = original_row_idx,
                        matrix_col = original_col_idx          
                        )                                    
                    
                    matches[str(csv_column)] = match
        
            except:
                print(f'{alignment_matrix}\n is an infeasible cost matrix')  
                
        elif alignment_policy == "greedy":
            mapped = []
            mtrx_row = -1
            
            cells=[]
            for csv_column, row in alignment_matrix.iterrows():
                mtrx_row+=1
                candidates = row.copy()
                candidates.drop(mapped, inplace=True)

                if len(candidates)==0:
                    break
                
                if maximize:
                    field = candidates.idxmax(skipna=True)
                else:
                    field = candidates.idxmin(skipna=True)
                
                
                if field is not np.NaN:
                    mapped.append(field)
                    mtrx_col = np.where(all_fields == field)[0][0]

                    original_col_idx = np.where(original_col_labels == field)[0][0]
                    original_row_idx = np.where(original_row_labels == alignment_matrix.index[mtrx_row])[0][0]                
                    cells.append((original_row_idx, original_col_idx))
                    # cells.append((mtrx_row, mtrx_col))
                    match = AutoColumnMatching(
                            field = field,
                            csv_column = csv_column,
                            extracted_name = source_headers[csv_column],
                            measure = matcher_measure,
                            metric_value = alignment_matrix.loc[csv_column,field],
                            matrix_col = original_col_idx,
                            matrix_row = original_row_idx
                            )
                    matches[str(csv_column)] = match
                
        # print(f'{alignment_policy} results = {cells}')
            
    return matches


    # def __init__(self, source_df: pd.DataFrame, target_df: pd.DataFrame, matching_technique: str = 'classification', classifier='Stacking'):
    #     self.source_df = source_df
    #     self.target_df = target_df
    #     self.matching_technique = matching_technique
    #     self.classifier = classifier  # External classifier, passed as parameter

    # def find_mapping(self) -> dict:
    #     mapping = {}
    #     for source_col in self.source_df.columns:
    #         best_match = None
    #         best_score = float('-inf')
    #         for target_col in self.target_df.columns:
    #             score = self._calculate_score(source_col, target_col)
    #             if score > best_score:
    #                 best_score = score
    #                 best_match = target_col
    #         mapping[source_col] = best_match
    #     return mapping