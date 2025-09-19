# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import csv
import random
from collections import defaultdict

import pandas as pd

from Common.example_bank import Example
from Common.taxonomy import QueryTaxonomy, AnaTaxonomy
from Evaluation.checker import get_L2_anatype

random.seed(0)
csvfile = open("human_eval.csv", mode='w', newline='', encoding="utf-8")
writer = csv.writer(csvfile)
writer.writerow(
    ["table_name", "query", "html", "code", "code_result", "given_parameter","ambiguities","ana_taxonomy"])

dataset_paths = ["subset_0811_insights.csv", "subset_0808_2.csv", "subset_0809_chart.csv"]

examples = defaultdict(lambda: defaultdict(list))  # task -> unclear -> [example]

outputs = []
for dataset_path in dataset_paths:
    dataset = pd.read_csv(dataset_path)
    for index, row in dataset.iterrows():
        example = Example.init_subset_query(row)
        task = get_L2_anatype(example.ana_taxonomy)
        example.python = "###### Code ######".join([str(i) for i in eval(row["python"])])
        example.python_res = "###### Result ######".join([str(i) for i in eval(row["python_res"])])
        example.ori_query = row["ori_query"]
        example.given_parameter = (eval(row["given_parameter"]) if row["given_parameter"].startswith("[") else [
            row["given_parameter"]]) if pd.notna(row["given_parameter"]) else [None]
        examples[task]["Clear" if QueryTaxonomy.Everything_is_clear in example.query_taxonomy else "Unclear"].append(
            example)
human_eval_examples = []
human_eval_examples.extend(random.sample(examples[AnaTaxonomy.Insights]["Clear"], 10))
human_eval_examples.extend(random.sample(examples[AnaTaxonomy.Insights]["Unclear"], 15))
human_eval_examples.extend(random.sample(examples[AnaTaxonomy.Chart]["Clear"], 10))
human_eval_examples.extend(random.sample(examples[AnaTaxonomy.Chart]["Unclear"], 15))
human_eval_examples.extend(random.sample(examples[AnaTaxonomy.Forecasting]["Clear"], 15))
human_eval_examples.extend(random.sample(examples[AnaTaxonomy.L1]["Clear"], 15))
human_eval_examples.extend(random.sample(examples[AnaTaxonomy.L1]["Unclear"], 10))
human_eval_examples.extend(random.sample(examples[AnaTaxonomy.Forecasting]["Unclear"], 10))



for example in human_eval_examples:
    writer.writerow(
        [example.table_name, example.query, example.html, example.python,
         example.python_res, example.given_parameter,example.query_taxonomy,example.ana_taxonomy])