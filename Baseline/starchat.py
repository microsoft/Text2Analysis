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
3. Print the result at the end.
4. The result must be one variable being one of pd.DataFrame, List[int], List[float], List[str], int, float and str.
5. For Pivot/groupby, Aggregation, Filter, Sort, Add virtual field, Calculation operations, you should use pandas library.
6. For Forecasting operation, you should use greykite or prophet library.
7. For plotting chart:  
7.1 You should use matplotlib library.
7.2 You should assign a !!!result!!! dictionary: {"x_fields": field_name, "y_fields": [field_name1, field_name2], "chart_type": chart_type}  
chart_type choose from lineChart, barChart, scatterChart, pieChart.   
7.3 You should print the result at the end.  
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


csvfile = open("starchat_0809_chart_alpha.csv", mode='w', newline='')
writer = csv.writer(csvfile)
writer.writerow(
    ["table_name", "query", "starchat"])

dataset_path = "subset_0809_chart.csv"
dataset = pd.read_csv(dataset_path)

# We use a special <|end|> token with ID 49155 to denote ends of a turn
pipe = pipeline("text-generation", model="HuggingFaceH4/starchat-alpha", torch_dtype=torch.bfloat16,
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
        batch_size = 16
        head = Example.TABLE_HEAD
    while 1:
        print(f"Start to generate for {table_name}, batch_size: {batch_size}, head: {head}")
        try:
            dataset = MyDataset(table2examples[table_name], head)
            idx = 0
            for out in tqdm(pipe(dataset, max_new_tokens=1024, do_sample=True, temperature=0.2, top_k=50, top_p=0.95,
                                 eos_token_id=49155, batch_size=batch_size), total=len(dataset)):
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
