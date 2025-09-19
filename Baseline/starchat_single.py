# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import csv
from collections import defaultdict
from Common.taxonomy import AnaTaxonomy
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import pipeline

from Common.example_bank import Example

TABLE_CONFIG = {
    "rail_lines.csv": {"batch": 1, "head": 0},
}
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

    def __init__(self, examples, head):
        self.examples = examples
        self.head = head

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        example = self.examples[i]
        example.TABLE_HEAD = self.head
        if AnaTaxonomy.Chart in example.ana_taxonomy:
            prompt = f"<|system|>\n{SYSTEM_PROMPT}\n<|end|>\n<|user|>\n{example.generate_table_query()}\nplease print the result dictionary at the end\n<|end|>\n<|assistant|>"
        else:
            prompt = f"<|system|>\n{SYSTEM_PROMPT}\n<|end|>\n<|user|>\n{example.generate_table_query()}\n<|end|>\n<|assistant|>"
        return prompt


csvfile = open("starchat_1219_beta.csv", mode='w', newline='')
writer = csv.writer(csvfile)
writer.writerow(
    ["table_name", "query", "starchat"])

dataset_path = "Tables/text2analysis_1220.csv"
dataset = pd.read_csv(dataset_path)

# We use a special <|end|> token with ID 49155 to denote ends of a turn
pipe = pipeline("text-generation", model="HuggingFaceH4/starchat-beta", torch_dtype=torch.bfloat16,
                device_map="auto")
pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id
table2examples = defaultdict(list)
for index, row in dataset.iterrows():
    example = Example.init_subset_query(row)
    table2examples[example.table_name].append(example)
print(
    f"There are {len(table2examples)} tables with {sum([len(i) for i in table2examples.values()])} examples in the dataset.")

for table_name in table2examples:
    if table_name in TABLE_CONFIG:
        batch_size = TABLE_CONFIG[table_name]["batch"]
        head = TABLE_CONFIG[table_name]["head"]
    else:
        batch_size = 0
        head = Example.TABLE_HEAD
    while 1:
        print(f"Start to generate for {table_name}, batch_size: {batch_size}, head: {head}")
        try:
            dataset = MyDataset(table2examples[table_name], head)
            idx = 0
            for idx in tqdm(range(len(dataset)),total = len(dataset)):
                out=pipe(dataset[idx], max_new_tokens=1024, do_sample=True, temperature=0.2, top_k=50, top_p=0.95,
                     eos_token_id=49155, batch_size=batch_size)
                outputs = out[0]["generated_text"]
                outputs = outputs.split("```python")[1].split("```")[0] if "```" in outputs else outputs
                writer.writerow([dataset.examples[idx].table_name, dataset.examples[idx].query, outputs])
                idx += 1
            break
        except Exception as e:
            if batch_size > 1:
                batch_size = batch_size // 2
            elif head > 0:
                head -= 1
            else:
                print(e)
                break
        del dataset
    del dataset

csvfile.close()
