from abc import ABC, abstractmethod
from typing import Dict, TypedDict, List
from pandas import DataFrame, Series
import pprint as pp
# from opendata_periplous.guides import extraction
from gnomon import file_utilities
# from opendata_periplous.db import csvtable, csvfile
from gnomon.utils import normalize_whitespace
# import logging
# logging.basicConfig()
import numpy as np
from scipy.optimize import linear_sum_assignment
from gnomon.alignment.WbBm import WBbM
import joblib

import time

class ColumnAligner():
    
    def __init__(self, schema_matcher, alignment_policy):
        self.schema_matcher = schema_matcher
        self.alignment_policy = alignment_policy
        # print(f'base_matcher.ColumnAligner.__init__>{self.alignment_policy=}')
        
    def align_df(self,
                 source_df:DataFrame,
                 target_df: DataFrame,
                 source_name:str = "source"): 
        source_headers = {i:h for i,h in enumerate(source_df.columns)}
        if source_name == "source":
            source_name+=":"+'__'.join(range(0,source_df.shape[1]))
        
        if len(source_df.columns)>0:
            start = time.time()
            alignment_matrix = self.schema_matcher.get_matches(
                source_name, 
                source_df, 
                target_df
            )
            end = time.time()
            matching_time = end - start
            
            start = time.time()
            column_alignments = resolve_alignment(
                alignment_matrix, 
                source_headers, 
                self.schema_matcher.matcher_measure,
                alignment_policy = self.alignment_policy,
                maximize = self.schema_matcher.maximize
            )            
            end = time.time()
            alignment_time = end - start
            
        # print(alignment_matrix)
        return column_alignments, alignment_matrix, source_headers, matching_time, alignment_time
    
    def align_table(self, source_df:DataFrame, target_df: DataFrame):
        """
        Given a source dataframe and a target schema
        align the source schema to the target schema
        """        
        
        column_alignments, alignment_matrix, source_headers, matching_time, alignment_time = self.align_df(
            source_df,
            target_df,
            str(source_df.name)
        )
        return column_alignments, alignment_matrix, source_headers, matching_time, alignment_time
                  
        

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
     - WBM: weight bipartite B-matching
            maximizes weight of alignments but allows for vertices with B capacity (N-M alignment)
            adapted from https://github.com/tbabm93/Bipartite_b_matching
    """
    matches={}
    alignment_matrix = original_df.copy(deep=True)
    original_row_labels = original_df.index
    original_col_labels = original_df.columns
    
    # print(f"base_matcher > resolve_alignment > alignment_matrix:\n\n{alignment_matrix}") #row labels are csv_column_idxs, col labels are fields
        
    alignment_matrix.replace([np.inf, -np.inf],np.nan, inplace = True)    
    alignment_matrix.dropna(axis=0, how="all", inplace = True)
    alignment_matrix.dropna(axis=1, how="all", inplace = True)  
    
    all_fields = alignment_matrix.columns
    # print(f'target fields = {all_fields}') 
    # print(f'alignment_matrix.shape={alignment_matrix.shape}')

    if not alignment_matrix.empty:
        if alignment_policy=="WBM":
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
            
            # lda = [0,1,0,0,0,0,0,0,0,0,0] #0 minimum source attributes matched to a metadata field
            # uda = [1,1,1,1,1,1,1,1,1,1,10] #3 maximum source attributes matched to a metadata field
                    
            # print(f'----\nlda={lda}, \nuda = {uda}, \nldp={ldp}, \nudp={udp}\n----')
            # print(f'type(ldp)= {type(ldp).__name__}')
            # print(f'type(udp)= {type(udp).__name__}')
            
            b_matching = WBbM.WBbM(num_left, num_right, [j for j in list(np.concatenate(W))] , lda, uda, ldp, udp, LogToConsole=0)       
            results, total_weight = b_matching.Bb_matching(optimization_mode = "max")

            # print(f'results={results}')
            # print(f'total_weight={total_weight}')
            
            for row_index in range(len( results )):
                for column_index in range(len( results[row_index] )):
                    if results[row_index][column_index] == 1:
                        selected_edges.append( (list(right)[column_index], list(left)[row_index]) ) # the order based on the gold-standard
                        mtrx_cells.append((list(left)[row_index], list(right)[column_index] ))
                        
            # print("Selected edges are:", selected_edges, "Total weight:", total_weight)
            
            for row, col in mtrx_cells:                
                original_col_idx = np.where(original_col_labels == alignment_matrix.columns[col])[0][0]                
                original_row_idx = np.where(original_row_labels == alignment_matrix.index[row])[0][0] 
                cells.append((original_row_idx, original_col_idx))
                csv_column = alignment_matrix.index[row]                
                match = AutoColumnMatching(
                    field = alignment_matrix.columns[col],
                    lang = 'eng',
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
                        lang = 'eng',
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
                            lang = 'eng',
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



class ColumnMatching(TypedDict):
    field: int
    csv_column: int
    extracted_name: str   

    
class AutoColumnMatching(ColumnMatching):
    metric_value: float
    measure: str
    matrix_col: int #col INDEX (not label) for the final matrix (cols removed)
    matrix_row: int # row INDEX (not label) for the final matrix (rows removed)
    
class BaseMatcher(ABC):

    @abstractmethod
    def get_matches(self, source_df: DataFrame, target_df:DataFrame) -> Dict[int, AutoColumnMatching]:
        """
        Get the column matches from a schema matching algorithm
        :returns List of matches
        """
        raise NotImplementedError
    
    def match_schema(self, csv_table, alignment_policy, target_df:DataFrame):
        aligner = ColumnAligner(schema_matcher = self, 
                                alignment_policy = alignment_policy)
        
        schema_matching, alignment_matrix, source_headers, matching_time, alignment_time = aligner.align_table(csv_table,
                                        target_df) 
        
        return {
            "schema_matching" : schema_matching,
            "alignment_matrix" : alignment_matrix,
            "source_headers" : source_headers,
            "matching_time" : matching_time,
            "alignment_time" : alignment_time
        }
        
    def match_schemas_in_file(self, datafile_key, alignment_policy, target_df:DataFrame):
        """
        Given a datafile_key
        align columns from the schema of each table in that file
        to a given target schema in matcher_args
        """
        # print(f'\nmatch_tables_in_file(datafile_key={datafile_key})')
        tables = csvfile.read_tables_GroundTruth(datafile_key)
        table_matching = {}
        for csv_table in tables:
            table_matching[csv_table.datatable_key] = self.match_schema(csv_table, alignment_policy, target_df)
            break # Only use first table in file
            
        return table_matching
    
    # save trained model to file
    def save(self, path):
        joblib.dump(self.model, path)
    # load saved model from file
    def load(self, path):    
        self.model = joblib.load(path)
        print(f'Loaded model from {path}') 
           
    def train(self, features_df: DataFrame, labels: Series):
        """        
        """
        pass    
    
    def explain(self):
        """        
        """
        pass
    
def eval_match_limited_fields(mapping, matching, policy_fields):
    """
    - mapping: holds the ground truth
    - matching: the automated alignment
    """
    
    is_same=True
    
    # check only targets of interest
    for target in policy_fields:
        # if the target exists in ground truth mappings 
        if any(normalize_whitespace(map["field"]) == normalize_whitespace(target) for map in mapping.values()):
            # find the source column that is mapped to this target
            csv_column = next((csv_column for csv_column, map in mapping.items() if normalize_whitespace(map["field"]) == normalize_whitespace(target)), None)
            # was that column matched with the target? if not there is an error
            if csv_column not in matching.keys() or normalize_whitespace(matching[csv_column]["field"]) != normalize_whitespace(target):
                is_same=False
                break    
            
            # were other columns matched to this target?
            for col, match in matching.items():
                if col!=csv_column and normalize_whitespace(match["field"])== normalize_whitespace(target):
                    is_same=False
                    break  
        else:
            # if the target doesnt exist in a ground truth mapping but exist in an automated matching
            if any(normalize_whitespace(match["field"]) == normalize_whitespace(target) for match in matching.values()):
                is_same=False
                break   
    
    return is_same   


