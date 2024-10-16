import json

class CSV_table:
    def __init__():
        pass
    
class TableColumnEncoder(json.JSONEncoder):
    def default(self, o):
        return o.__dict__
            
class TableColumn:
    def __init__(self, 
                 tablecolumn_key = None, 
                 csv_column = None, 
                 datatable_key = None,
                 datafile_key = None,
                 header = None,
                 values = None
                 ):
        self.tablecolumn_key = tablecolumn_key
        self.csv_column = csv_column
        self.datatable_key = datatable_key
        self.datafile_key = datafile_key
        self.header = header
        self.values = values
        
    def toJson(self):
        return json.dumps(self.__dict__)
        # return json.dumps(self, default=lambda o: o.__dict__)

    def to_dict(self):
        return json.loads(TableColumnEncoder().encode(self))
    
    def get_column(cls, tablecolumn_key, file_df):
        table_column = cls()
        table_column.tablecolumn_key = tablecolumn_key

        return "TODO implement"