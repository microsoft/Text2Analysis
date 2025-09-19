# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import csv
from collections import defaultdict

import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
import jsonlines
from GCR.run import FewShotLLM

from Common.example_bank import Example
from Common.taxonomy import AnaTaxonomy


SYSTEM_PROMPT = """
I want you to act as a data analytics scientist. I will give you a tables, an user query, and you need to generate Python code to answer the user query. You should follow the rules:
1. Return the Python code.
2. There is already a DataFrame variable for `tables` in the environment, so you should not reassign `tables=pd.read_html(...)` in the code.
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
There are two descriptions of insight:
Add there are a private API:
+ is_insight: bool, whether the insight is true
+ property: dict, the property of the insight. For "Rank", key is "First" and value is str; for "RankLast", key is "Last" and value is str; for "Attribution", key is "Dominator" and value is str; for "Trend", key is "IsIncreasing" and valus is bool; for "Monotonicity", key is "Trend" and value is "Increasing" or "Decreasing"; for "Outlier", key is "TemporalOutlierLocation" and value is List[str]; for "Unimodality", key is "Location" and value is List[str].
You should answer the query with the same format of one or more descriptions of insight.
"""


class MyDataset(Dataset):

    def __init__(self, examples, head=None):
        self.examples = examples
        self.head = head if head else Example.TABLE_HEAD

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        example = self.examples[i]
        example.TABLE_HEAD = self.head
        if AnaTaxonomy.Chart in example.ana_taxonomy:
            prompt = [SYSTEM_PROMPT, '',
                      example.generate_table_query() + "\nplease print the result dictionary at the end"]
        else:
            prompt = [SYSTEM_PROMPT, '',
                      example.generate_table_query()]
        return prompt


# dataset_paths = ["subset_0811_insights.csv","subset_0808_2.csv","subset_0809_chart.csv"]
dataset_paths = ["Tables/text2analysis_1220.csv"]
outputs=[]
for dataset_path in dataset_paths:
    dataset = pd.read_csv(dataset_path)

    table2examples = defaultdict(list)
    for index, row in dataset.iterrows():
        example = Example.init_subset_query(row)
        table2examples[example.table_name].append(example)
    print(
        f"There are {len(table2examples)} tables with {sum([len(i) for i in table2examples.values()])} examples in the dataset.")
    continue_flag = True
    cnt=0
    for table_name in tqdm(table2examples):
        print(f"Start to generate for {table_name}")
        if not continue_flag:
            if table_name == "110781.t0.csv":
                continue_flag = True
            else:
                continue
        dataset = MyDataset(table2examples[table_name])
        for idx in tqdm(range(len(dataset))):
            prompt = dataset[idx]
            outputs.append({"table_name": table_name, "query": dataset.examples[idx].query, "prompt": '\r\n'.join(prompt)})
        cnt+=len(dataset)
        print(f"{cnt/2583} ({cnt}/{2583})")


with jsonlines.open("gpt_family_1219.jsonl", mode='w') as writer:
    writer.write_all(outputs)
print(f"dump {len(outputs)} qa")
