# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import csv
from collections import defaultdict

import pandas as pd
from tqdm import tqdm
from transformers import TapasTokenizer, TapasForQuestionAnswering


from Common.example_bank import Example

csvfile = open("tapas_0811.csv", mode='w', newline='')
writer = csv.writer(csvfile)
writer.writerow(
    ["table_name", "query", "tapas"])

dataset_path = "subset_0808_2.csv"
# dataset_path = "subset_0809_chart.csv"
dataset = pd.read_csv(dataset_path)

tokenizer = TapasTokenizer.from_pretrained('google/tapas-large-finetuned-wtq')
model = TapasForQuestionAnswering.from_pretrained('google/tapas-large-finetuned-wtq')

table2examples = defaultdict(list)
for index, row in tqdm(dataset.iterrows(), total=len(dataset)):
    example = Example.init_subset_query(row)
    query = example.query
    if "chart" in dataset_path:
        query += ' You should answer the result as follows: {chart_type}, {x_field_name}, {y_field_name1}, {y_field_name2}...'
    encoding = tokenizer(table=example.table.astype(str), queries=[query], return_tensors="pt", truncation=True)
    print(example.query, example.table)
    outputs = model(**encoding)[0]
    outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(outputs)
    writer.writerow([example.table_name, example.query, outputs])

csvfile.close()
