# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import csv
from collections import defaultdict
import time
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
from Baseline.prompt import get_examples
# from GCR.run import FewShotLLM
#
# few_shot_llm = FewShotLLM()
from Common.example_bank import Example
from Common.taxonomy import AnaTaxonomy

from api.run import LLMClient
llm_client = LLMClient()

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
                      example.generate_table_query() + "\nplease print the result dictionary at the end."]
        else:
            prompt = [SYSTEM_PROMPT, '',
                      example.generate_table_query()]
        return prompt


csvfile = open("gpt35_1220.csv", mode='w', newline='', encoding="utf-8")
writer = csv.writer(csvfile)
writer.writerow(
    ["table_name", "query", "gpt4"])

FEW_SHOT=False

dataset_path = "Tables/text2analysis_1220.csv"
dataset = pd.read_csv(dataset_path)

table2examples = defaultdict(list)
for index, row in dataset.iterrows():
    example = Example.init_subset_query(row)
    table2examples[example.table_name].append(example)
print(
    f"There are {len(table2examples)} tables with {sum([len(i) for i in table2examples.values()])} examples in the dataset.")
continue_flag = True
# keys_in_reverse_order = list(table2examples.keys())[::-1]
keys_in_reverse_order = list(table2examples.keys())[100:]
for table_name in keys_in_reverse_order:
    print(f"Start to generate for {table_name}")
    if not continue_flag:
        if table_name == "110781.t0.csv":
            continue_flag = True
        else:
            continue
    dataset = MyDataset(table2examples[table_name])
    for idx in tqdm(range(len(dataset))):
        prompt = dataset[idx]
        prompt[1]="" if not FEW_SHOT else get_examples(need_role=False)
        outputs = llm_client.run_llm('\r\n'.join(prompt))
        # time.sleep(5)
        # outputs = few_shot_llm._completion(prompt[0],
        #                                    prompt[1],
        #                                    prompt[2])

        outputs = outputs.split("```python")[1].split("```")[0] if outputs!=None and "```" in outputs else str(outputs)
        writer.writerow([dataset.examples[idx].table_name, dataset.examples[idx].query, outputs])
csvfile.close()
