# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
from Common.example_bank import Example
from Common.safe_data import parse_example_fields
from Common.taxonomy import AnaTaxonomy
from Evaluation.checker import Checker, get_L2_anatype, FORECASTING_GT
import os

SYSTEM_PROMPT = """
PROBLEM:
I want you to act as a data analytics scientist. I will give you a table, an user query, and you need to generate Python code to answer the user query. You should follow the rules:
1. Return the Python code.
2. There is already a DataFrame variable for `table` in the environment, so you should not reassign `table=pd.read_html(...)` in the code.
3. Stores the final result in the variable `result`. I will directly utilize the 'result' variable for assessing the subsequent code outcomes.
4. The result must be one variable being one of pd.DataFrame, List[int], List[float], List[str], int, float, str, dict.
5. For Pivot/groupby, Aggregation, Filter, Sort, Add virtual field, Calculation operations, you should use pandas library.
6. For forecasting, you should use greykite or prophet library.
7. For chart generation:  
7.1 You should use matplotlib library.
7.2 You should assign a !!!result!!! dictionary: {"x_fields": field_name, "y_fields": [field_name1, field_name2], "chart_type": chart_type}  
chart_type choose from lineChart, barChart, scatterChart, pieChart.   
7.3 You should print the result at the end.  
8. For basic insight, there are 7 types: Rank, RankLast, Attribution, Trend, Monotonicity, Outlier, Unimodality. 
8.1 There are two descriptions of insight, and you can choose one of them or their values to answer the query:
    + is_insight: bool, whether the insight is true
    + property: dict, the property of the insight. For "Rank", key is "First" and value is str; for "RankLast", key is "Last" and value is str; for "Attribution", key is "Dominator" and value is str; for "Trend", key is "IsIncreasing" and valus is bool; for "Monotonicity", key is "Trend" and value is "Increasing" or "Decreasing"; for "Outlier", key is "TemporalOutlierLocation" and value is List[str]; for "Unimodality", key is "Location" and value is List[str].
8.2 The result of your answer to the query needs to refer the description. The result must be a direct answer to the query, not a direct dict of the above descriptions. The result can be True, "Increasing", "Decreasing" and so on.
"""

DATSET_PATH = {
    AnaTaxonomy.Chart: 'Tables/chart4annotation-202308',
    AnaTaxonomy.Forecasting: 'Tables/forecasting',
    AnaTaxonomy.Insights: "Tables/insight_dataset",
    AnaTaxonomy.L1: 'Tables/source',
    "chart": 'Tables/chart4annotation-202308',
    "pivot": 'Tables/pivot4annotation-202308',
}

class Text2AnalysisDataset:
    def __init__(
            self,
            source_dir="Tables/text2analysis_1220.csv",
            is_eval=False
    ):
        dataset = pd.read_csv(source_dir)
        self.examples = []
        for index, row in dataset.iterrows():
            example = Example.init_subset_query(row)
            if AnaTaxonomy.Chart in example.ana_taxonomy:
                prompt = [SYSTEM_PROMPT, '',
                        example.generate_table_query() + "\nplease print the result dictionary at the end."]
            else:
                prompt = [SYSTEM_PROMPT, '',
                        example.generate_table_query()]
            example.prompt = "\r\n".join(prompt)
            example.problem_id=index
            example.source=row["source"]

            if is_eval:
                fields = parse_example_fields(row)
                example.python = fields["python"]
                example.python_res = fields["python_res"]
                example.ori_query = row["ori_query"]
                example.given_parameter = fields["given_parameter"]
                if pd.isna(example.ori_query):
                    example.ori_query = example.query
                
                task = get_L2_anatype(example.ana_taxonomy)
                example.task = task
                table_name = example.table_name
                if table_name.startswith("chart"):
                    dataset_path = DATSET_PATH["chart"]
                    table_name = example.table_name[6:]
                elif table_name.startswith("pivot"):
                    dataset_path = DATSET_PATH["pivot"]
                    table_name = example.table_name[6:]
                else:
                    dataset_path = DATSET_PATH[task]
                if table_name.endswith("csv"):
                    table = pd.read_csv(os.path.join(dataset_path, table_name))
                else:
                    table = pd.read_excel(os.path.join(dataset_path, table_name))
                example.table = table

                example.code_result = fields["code_result"]

            self.examples.append(example)

    def __getitem__(self, key):
        return self.examples[key]

    def __len__(self):
        return len(self.examples)