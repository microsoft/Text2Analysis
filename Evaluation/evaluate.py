# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import csv
import os
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from Common.example_bank import Example
from Common.taxonomy import AnaTaxonomy
from Evaluation.checker import Checker, get_L2_anatype, FORECASTING_GT

# gt_path = "subset_0808_3.csv"
# pred_path = "gpt4_0809_result.csv"
# pred_path = "starchat_0808_2.csv"
# pred_path = "starchat_0808_2_alpha.csv"
# pred_path = "tapex_0811.csv"
# pred_path="gpt3_0816_forecasting.csv"
# pred_path="gpt35_0816_forecasting.csv"
# pred_path="chatgpt_0816_forecasting.csv"
# pred_path = "codegen_insights_2.csv"
#
# gt_path = "subset_0809_chart.csv"
# pred_path = "gpt4_0809_chart.csv"
# pred_path= "starchat_0809_chart.csv"
# pred_path = "starchat_0809_chart_alpha.csv"
# pred_path="gpt35_0816_chart.csv"
# pred_path="chatgpt_0816_chart.csv"
# pred_path = "codegen_insights_chart.csv"


# gt_path="subset_0811_insights.csv"
# pred_path="starchat_0811_insights.csv"
# pred_path="gpt4_0811_insights.csv"
# pred_path="starchat_0811_insights_alpha.csv"
# pred_path="gpt35_0816_insight.csv"
# pred_path="chatgpt_0816_insight.csv"
# pred_path = "codegen_insights_insights.csv"

gt_path = "Tables/text2analysis_1220.csv"
# pred_path = "tapex_1219_tapex-large-finetuned-wtq.csv"
# pred_path = "starchat_1219_alpha.csv"
# pred_path = "gpt4_1219.csv"
# pred_path = "starchat_1219_beta.csv"
pred_path="codegen_1219.csv"

DATSET_PATH = {
    AnaTaxonomy.Chart: 'Tables/chart4annotation-202308',
    AnaTaxonomy.Forecasting: 'Tables/forecasting',
    AnaTaxonomy.Insights: "Tables/insight_dataset",
    AnaTaxonomy.L1: 'Tables/source',
    "chart": 'Tables/chart4annotation-202308',
    "pivot": 'Tables/pivot4annotation-202308',
}
if __name__ == '__main__':

    evaluation_dict = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))  # task -> unclear_taxo -> given_parameter -> [eval]

    gt = pd.read_csv(gt_path)
    pred = pd.read_csv(pred_path)  # , encoding='cp1252')
    table2examples = defaultdict(dict)  # table_name -> query -> example
    for index, row in gt.iterrows():
        example = Example.init_subset_query(row)
        example.python = eval(row["python"]) if str(row["python"]).startswith("[") else [str(row["python"])]
        example.python_res = eval(row["python_res"]) if str(row["python"]).startswith("[") else [str(row["python_res"])]
        example.ori_query = row["ori_query"]
        example.given_parameter = (eval(row["given_parameter"]) if row["given_parameter"].startswith("[") else [
            row["given_parameter"]]) if pd.notna(row["given_parameter"]) else [None]
        table2examples[example.table_name][example.query] = example


    for index, row in tqdm(pred.iterrows(), total=len(pred)):
        try:
            example = table2examples[row["table_name"]][row["query"]]
        except:
            continue  # TODO:Find reason
        task = get_L2_anatype(example.ana_taxonomy)
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
        if "code_result" in row and pd.notna(row["code_result"]):
            try:
                code_result = eval(row["code_result"]) if row["code_result"].startswith("[") else [row["code_result"]]
            except:
                code_result = [row["code_result"]]
        else:
            code_result = None
        eva_list = []
        print(f"[query]: {example.query}")
        if pd.isna(example.ori_query):
            example.ori_query = example.query
        if task == AnaTaxonomy.Forecasting and not any(
                FORECASTING_GT["query"].isin([example.ori_query.replace(" ", "").lower()])):
            print(f"[Not find ori forecasting query]: {example.ori_query}")
            continue

        for idx, py in enumerate(example.python):
            if "gpt" in pred_path:
                code = row["gpt4"]
                code_lines = code.strip().split("\n")
                if code_lines[0][0] != '#':
                    code_lines[0] = "#" + code_lines[0]
                for i, l in enumerate(code_lines):
                    if "Python code" in l or "python code" in l:
                        code_lines[i] = "#" + code_lines[i]
                code = '\n'.join(code_lines)
                code = "df = table\n" + code
                code=code.replace("tables[0]", "table")
                checker = Checker(table, code, py, example.python_res[idx], example.ana_taxonomy,
                                  example.ori_query, code_pre_result=code_result[idx] if code_result else None)
            elif "star" in pred_path:
                checker = Checker(table, row["starchat"], py, example.python_res[idx], example.ana_taxonomy,
                                  example.ori_query, code_pre_result=code_result[idx] if code_result else None)
            elif "tapex" in pred_path:
                checker = Checker(table, None, py, example.python_res[idx], example.ana_taxonomy,
                                  example.ori_query, code_pre_result=row["wtq"])
            elif "codegen" in pred_path:
                # Add # to the first space of each line
                lines = row["codegen"].split('\n')  # 按行分割字符串
                for i in range(len(lines)):
                    if lines[i].startswith(" ") and not lines[i].startswith("  "):
                        lines[i] = lines[i].replace(" ", "#", 1)
                row["codegen"] = '\n'.join(lines)

                checker = Checker(table, row["codegen"], py, example.python_res[idx], example.ana_taxonomy,
                                  example.ori_query, code_pre_result=code_result[idx] if code_result else None)
            else:
                raise NotImplementedError(f"Unknown model: {pred_path}")

            eva = checker.check_result()
            eva_list.append(eva)
        eva_sum = [e[1] + e[2] + (e[3] if type(e[3]) != dict else e[3]["R2"]) for e in eva_list]
        eva_sum_max = max(eva_sum)
        idx = eva_sum.index(eva_sum_max)
        for unclear in example.query_taxonomy:
            for para in example.given_parameter:
                evaluation_dict[task][unclear][para].append(eva_list[idx][1:])

    eval_path = "evaluation_result_1220.csv"
    csvfile = open(eval_path, mode='a', newline='', encoding="utf-8")
    writer = csv.writer(csvfile)
    writer.writerow(["file","task", "unclear_taxonomy", "given_query_taxonomy", "total","executable", "exact_match", 'CORR', 'R2', 'MSE', 'RMSE', 'MAE', 'MedAE', "forecast_metric_total"])
    print(evaluation_dict)
    for task in evaluation_dict:
        for unclear in evaluation_dict[task]:
            for para in evaluation_dict[task][unclear]:
                if task != AnaTaxonomy.Forecasting:
                    total = len(evaluation_dict[task][unclear][para])
                    executable = sum([e[0] for e in evaluation_dict[task][unclear][para]])
                    exact_match = sum([e[1] for e in evaluation_dict[task][unclear][para]])
                    print(f"task: {task} ,unclear_taxonomy: {unclear}, given_para: {para}")
                    print(f"total: {total}, executable: {executable / total}, exact_match: {exact_match / total}")
                    writer.writerow(
                        [pred_path, task.value, unclear, para, total, executable, exact_match, "", "", "", "", "", "",""])
                else:
                    total = len(evaluation_dict[task][unclear][para])
                    executable = sum([e[0] for e in evaluation_dict[task][unclear][para]])
                    exact_match = sum([e[1] for e in evaluation_dict[task][unclear][para]])
                    CORR = sum([e[2]["CORR"] for e in evaluation_dict[task][unclear][para] if type(e[2]) == dict])
                    R2 = sum([e[2]["R2"] for e in evaluation_dict[task][unclear][para] if type(e[2]) == dict])
                    MSE = sum([e[2]["MSE"] for e in evaluation_dict[task][unclear][para] if type(e[2]) == dict])
                    RMSE = sum([e[2]["RMSE"] for e in evaluation_dict[task][unclear][para] if type(e[2]) == dict])
                    MAE = sum([e[2]["MAE"] for e in evaluation_dict[task][unclear][para] if type(e[2]) == dict])
                    MedAE = sum([e[2]["MedAE"] for e in evaluation_dict[task][unclear][para] if type(e[2]) == dict])
                    print(f"task: {task} ,unclear_taxonomy: {unclear}, given_para: {para}")
                    print(f"total: {total}, executable: {executable / total}, exact_match: {exact_match / total}")
                    print(
                        f"CORR: {CORR / total}, R2: {R2 / total}, MSE: {MSE / total}, RMSE: {RMSE / total}, MAE: {MAE / total}, MedAE: {MedAE / total}")
                    writer.writerow(
                        [pred_path, task.value, unclear, para, total, executable, exact_match, CORR, R2, MSE, RMSE, MAE,
                         MedAE,len([e[2]["CORR"] for e in evaluation_dict[task][unclear][para] if type(e[2]) == dict])])
    csvfile.close()
