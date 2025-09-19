# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import csv

import jsonlines
import pandas as pd
from tqdm import tqdm

insight_dataset = pd.read_csv("subset_0811_insights.csv")
chart_dataset = pd.read_csv("subset_0809_chart.csv")
forecast_dataset = pd.read_csv("subset_0808_2.csv")

# output_path = "chatgpt_0816"
# output_gpt_path = "chatgpt_0815_samples.0.jsonl"

# output_path = "gpt3_0816"
# output_gpt_path = "gpt3_0815_samples.0.jsonl"

output_path = "gpt35_0816"
output_gpt_path = "gpt35_0815_samples.0.jsonl"

input_gpt_path = "gpt_family_0815.jsonl"



csvfile_insight = open(f"{output_path}_insight.csv", mode='w', newline='', encoding="utf-8")
writer_insight = csv.writer(csvfile_insight)
writer_insight.writerow(
    ["table_name", "query", "gpt4"])

csvfile_chart = open(f"{output_path}_chart.csv", mode='w', newline='', encoding="utf-8")
writer_chart = csv.writer(csvfile_chart)
writer_chart.writerow(
    ["table_name", "query", "gpt4"])

csvfile_forecasting = open(f"{output_path}_forecasting.csv", mode='w', newline='', encoding="utf-8")
writer_forecasting = csv.writer(csvfile_forecasting)
writer_forecasting.writerow(
    ["table_name", "query", "gpt4"])



def read_jsonl(file_path):
    """
    读取 JSONL 文件并将其解析为一个包含 JSON 对象的列表。

    参数:
        file_path (str): JSONL 文件的路径。

    返回:
        list: 包含 JSON 对象的列表。
    """
    json_objects = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            json_objects.append(obj)
    return json_objects

error_cnt=0
input_gpt = read_jsonl(input_gpt_path)
output_gpt = read_jsonl(output_gpt_path)
for i in tqdm(range(len(input_gpt))):
    table_name = input_gpt[i]["table_name"]
    query = input_gpt[i]["query"]
    output = output_gpt[i]["samples"][0]
    if any(insight_dataset["query"].isin([query])):
        writer_insight.writerow([table_name, query, output])
    elif any(chart_dataset["query"].isin([query])):
        writer_chart.writerow([table_name, query, output])
    elif any(forecast_dataset["query"].isin([query])):
        writer_forecasting.writerow([table_name, query, output])
    else:
        print(f"error query {query}")
        error_cnt+=1
        continue

print(f"error_cnt {error_cnt}")
csvfile_chart.close()
csvfile_forecasting.close()
csvfile_insight.close()

