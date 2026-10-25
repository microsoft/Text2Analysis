# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import csv
import os
from collections import defaultdict

import pandas as pd

from Common.example_bank import Example
from Common.safe_data import parse_example_fields
from Common.taxonomy import AnaTaxonomy
from Evaluation.check_DIAL_result import python_res2res
from Evaluation.checker import cut_forecast_table, eval_python, get_L2_anatype
from Evaluation.evaluate import DATSET_PATH

ori_file = f"code2query 1.csv"
FORECASTING_GT = pd.read_csv(ori_file)

subset_path = f"subset_0808_2.csv"
subset_table = pd.read_csv(subset_path)
table2examples = defaultdict(dict)  # table_name -> query -> example
for index, row in subset_table.iterrows():
    example = Example.init_subset_query(row)
    fields = parse_example_fields(row)
    example.python = fields["python"]
    example.python_res = fields["python_res"]
    example.ori_query = row["ori_query"]
    table2examples[example.table_name][example.query] = example

input_file = f"code2query 1.csv"
# input_file = f"gpt4_0809.csv"
# input_file = f"starchat_0808_2.csv"
# input_file = f"starchat_0808_2_alpha.csv"
# input_table = pd.read_csv(input_file, encoding='cp1252')
input_table = pd.read_csv(input_file)

output_file = input_file[:-len(".csv")] + "_result.csv"
csvfile = open(output_file, mode='w', newline='', encoding="utf-8")
writer = csv.writer(csvfile)
writer.writerow(input_table.columns.tolist() + ["code_result"])

for index, row in input_table.iterrows():
    print("!!!!!!!!!!!!!!!!!!!!!", row["query"])
    try:
        example = table2examples[row["table_name"]][row["query"]]
    except:
        writer.writerow(row.tolist() + [""])
        continue
    task = get_L2_anatype(example.ana_taxonomy)
    if task != AnaTaxonomy.Forecasting or row["table_name"] != "rail_lines.csv":
        writer.writerow(row.tolist() + [""])
        continue
    try:
        delete_line = FORECASTING_GT.loc[
            FORECASTING_GT["query"] == (example.ori_query if pd.notna(example.ori_query) else example.query), [
                "delete_lines"]].values[0][
            0] if input_file != ori_file else row["delete_lines"]
    except:
        writer.writerow(row.tolist() + [""])
        continue
    table = pd.read_csv(os.path.join(DATSET_PATH[AnaTaxonomy.Forecasting], row["table_name"]))
    table = cut_forecast_table(table, delete_line, row["table_name"])

    if "gpt4" in input_file:
        code = row["gpt4"]
    elif "star" in input_file:
        code = row["starchat"]
    else:
        code = row["code"]
    code_gt_result = eval_python(code.replace("fbprophet", "prophet"), table)
    writer.writerow(row.tolist() + [python_res2res(code_gt_result)])

csvfile.close()
