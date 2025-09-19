# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import csv
from collections import defaultdict

import pandas as pd
from tqdm import tqdm
from transformers import TapexTokenizer, BartForConditionalGeneration

from Common.example_bank import Example

# dataset_path = "subset_0808_2.csv"
# dataset_path = "subset_0809_chart.csv"
# dataset_path = "subset_0811_insights.csv"
dataset_path = "Tables/text2analysis_1220.csv"
dataset = pd.read_csv(dataset_path)


# for model_name in ["tapex-large", "tapex-large-sql-execution", "tapex-large-finetuned-wtq",
#                    "tapex-large-finetuned-wikisql", "tapex-large-finetuned-tabfact"]:
for model_name in ["tapex-large-finetuned-wtq"]:
    print("!!!!!!!Start to generate for ", model_name)
    csvfile = open(f"tapex_1219_{model_name}.csv", mode='w', newline='')
    writer = csv.writer(csvfile)
    writer.writerow(
        ["table_name", "query", "wtq"])

    tokenizer = TapexTokenizer.from_pretrained(f"microsoft/{model_name}")
    model = BartForConditionalGeneration.from_pretrained(f"microsoft/{model_name}")

    table2examples = defaultdict(list)
    for index, row in tqdm(dataset.iterrows(), total=len(dataset)):
        example = Example.init_subset_query(row)
        query = str(example.query)
        if "chart" in dataset_path:
            query += ' You should answer the result as follows: {chart_type}, {x_field_name}, {y_field_name1}, {y_field_name2}...'
        encoding = tokenizer(table=example.table.astype(str), query=query, return_tensors="pt", truncation=True)
        print(example.query, example.table)
        outputs = model.generate(**encoding)
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].split()
        if type(outputs)==str and "," in outputs:
            outputs = outputs.split(",")
        print(outputs)
        writer.writerow([example.table_name, example.query, outputs])

    csvfile.close()
