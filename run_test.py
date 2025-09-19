# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os
import numpy as np

import pandas as pd

# need to disable GPU usage explicitly for execution
# otherwise, multiprocessing tensorflow programs may hang indefinitely
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# disable tensorflow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from collections import defaultdict
from typing import List, Tuple, Union

from tqdm import tqdm, trange
from pathlib import Path

from argparse import ArgumentParser
import glob
from text2analysis import Text2AnalysisDataset
from Evaluation.checker import Checker, get_L2_anatype, FORECASTING_GT
import csv
from Common.taxonomy import AnaTaxonomy,QueryTaxonomy
import resource

# 设置软限制和硬限制为500MB  
rsrc = resource.RLIMIT_AS  
soft, hard = resource.getrlimit(rsrc)  
print('Soft limit starts as  :', soft)  
  
resource.setrlimit(rsrc, (6 * 1024 * 1024 * 1024, hard)) # 设置为500MB  

EVALUATION_PATTERN2PARA={
    (AnaTaxonomy.L1,QueryTaxonomy.Lacks_table_references,"sort_field"):"msr_field",
    (AnaTaxonomy.L1,QueryTaxonomy.Lacks_table_references,"msr_field"):"msr_field",
    (AnaTaxonomy.L1,QueryTaxonomy.Lacks_table_references,None):"msr_field",
    (AnaTaxonomy.L1,QueryTaxonomy.Everything_is_clear,None):"clear",
    (AnaTaxonomy.L1,QueryTaxonomy.Lacks_other_API_arguments,"agg_func"):"agg_func",
    (AnaTaxonomy.L1,QueryTaxonomy.Lacks_other_API_arguments,None):"agg_func",
    (AnaTaxonomy.Forecasting,QueryTaxonomy.Lacks_table_references,"forecast_field"):"forecast_field",
    (AnaTaxonomy.Forecasting,QueryTaxonomy.Lacks_other_API_arguments,"freq or steps or (freq and steps)"):"freq/steps",
    (AnaTaxonomy.Forecasting,QueryTaxonomy.Everything_is_clear,None):"clear",
    (AnaTaxonomy.Insights,QueryTaxonomy.Everything_is_clear,None):"clear",
    (AnaTaxonomy.Insights,QueryTaxonomy.Lacks_table_references,"dim"):"field",
    (AnaTaxonomy.Insights,QueryTaxonomy.Lacks_table_references,"msr"):"field",
    (AnaTaxonomy.Insights,QueryTaxonomy.Lacks_table_references,None):"field",
    (AnaTaxonomy.Chart,QueryTaxonomy.Everything_is_clear,"None"):"clear",
    (AnaTaxonomy.Chart,QueryTaxonomy.Lacks_table_references,"x_fields"):"x_field",
    (AnaTaxonomy.Chart,QueryTaxonomy.Lacks_table_references,"y_fields"):"y_field",
    (AnaTaxonomy.Chart,QueryTaxonomy.Lacks_other_API_arguments,"chart_type"):"chart_type",
}

killed_id={ # Those answer is killed by OOM, and can't be captured by try-except
    "gpt_outputs_0128":{"gpt-4-32k-0613":[244,267]},
    "gpt_retrieve_outputs_0128_2":{"gpt-4-32k-0613":[238, 245]},
    # "gpt_outputs_0129": {"gpt-4-32k-0613": [237,244,250,312]},
}

def test_model(
        text2analysis: Text2AnalysisDataset,
        model: str,
        output_dir: Union[str, Path] = "codex_greedy_outputs",
        args=None,
):
    """
    Use multiprocessing to test a certain model's generated codes, and dump the score records into json files

    `model` is a str, helping to specify the path where generated codes should be stored
    `mode` is a str, can be "Insertion" or "Completion"
    `num_procs` is the number of processes to set
    `output_dir` is the path where the generated codes are stored
    `source_dir` is the path where the dataset is stored

    a single generated code's path can be like `output_dir` / `model` / Pandas / `mode` / q0 / 0.py
    """
    score = defaultdict(list)
    problem_code_pairs = []
    evaluation_dict = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list)))  # task -> unclear_taxo -> given_parameter -> [eval]

    
    for problem_id in trange(len(text2analysis)):
        
        if problem_id in killed_id.get(output_dir,{}).get(model,[]):
            print(f"[Killed by OOM]: {problem_id}")
            problem_code_pairs.append((text2analysis[problem_id], [""]))
            continue
        problem=text2analysis[problem_id]
        if problem.task == AnaTaxonomy.Forecasting and not any(
                FORECASTING_GT["query"].isin([problem.ori_query.replace(" ", "").lower()])):
            print(f"[Not find ori forecasting query]: {problem.ori_query}")
            continue


        generated_code_path = (
                Path(output_dir) / model / ("q" + str(problem_id))
        )
        code_list = []


        for generated_code_sample_path in glob.glob(
                str(generated_code_path / "*.py")
        ):
            code = open(generated_code_sample_path, "r", encoding="UTF-8").read()
            code_list.append(code)
        problem_code_pairs.append((text2analysis[problem_id], code_list))

    result_cache_path = Path(output_dir) / f"{model}_result_cache"
    os.makedirs(result_cache_path, exist_ok=True)
    for problem_code_pair in tqdm(problem_code_pairs):
        problem, code_list = problem_code_pair
        eva_list=[]
        for idx, py in enumerate(problem.python):
            print(f"------------------------{problem.problem_id}------------------------")
            checker = Checker(problem.table, code_list[0], py, problem.python_res[idx], problem.ana_taxonomy,
                                problem.ori_query, code_pre_result=problem.code_result[idx] if problem.code_result else None)
            eva = checker.check_result()
            eva_list.append(eva)
        eva_sum = [e[1] + e[2] + (e[3] if type(e[3]) != dict else e[3]["R2"]) for e in eva_list]
        eva_sum_max = max(eva_sum)
        idx = eva_sum.index(eva_sum_max)
        for unclear in problem.query_taxonomy:
            for para in problem.given_parameter:
                evaluation_dict[problem.task][unclear][para].append(eva_list[idx][1:])
        with open(result_cache_path / f"{problem.problem_id}.json", "w") as f:
            json.dump(
                {
                    "id":problem.problem_id,
                    "executable":eva_list[idx][1],
                    "exact_match":eva_list[idx][2],
                    "forecast":eva_list[idx][2],
                    "task":problem.task.value,
                    "unclear_taxonomy":[i.value for i in problem.query_taxonomy],
                    "given_query_taxonomy":problem.given_parameter}, f)
            

    eval_path = "evaluation_result.csv"
    # csvfile = open(eval_path, mode='a', newline='', encoding="utf-8")
    # writer = csv.writer(csvfile)
    header=["task", "unclear_taxonomy", "given_query_taxonomy", "total","executable", "exact_match", 'CORR', 'R2', 'MSE', 'RMSE', 'MAE', 'MedAE', "forecast_metric_total"]
    df=pd.DataFrame(columns=header)
    # writer.writerow(["task", "unclear_taxonomy", "given_query_taxonomy", "total","executable", "exact_match", 'CORR', 'R2', 'MSE', 'RMSE', 'MAE', 'MedAE', "forecast_metric_total"])
    for task in evaluation_dict:
        for unclear in evaluation_dict[task]:
            for para in evaluation_dict[task][unclear]:
                if (task,unclear,para) in EVALUATION_PATTERN2PARA:
                    para_new=EVALUATION_PATTERN2PARA[(task,unclear,para)]
                else:
                    continue
                if task != AnaTaxonomy.Forecasting:
                    total = len(evaluation_dict[task][unclear][para])
                    executable = sum([e[0] for e in evaluation_dict[task][unclear][para]])
                    exact_match = sum([e[1] for e in evaluation_dict[task][unclear][para]])
                    # print(f"task: {task} ,unclear_taxonomy: {unclear}, given_para: {para}")
                    # print(f"total: {total}, executable: {executable / total}, exact_match: {exact_match / total}")
                    # writer.writerow(
                    #     [task.value, unclear, para_new, total, executable, exact_match, "", "", "", "", "", "",""])
                    df.loc[len(df)] = [task.value, unclear, para_new, total, executable, exact_match, "", "", "", "", "", "",""]
                else:
                    total = len(evaluation_dict[task][unclear][para])
                    executable = sum([e[0] for e in evaluation_dict[task][unclear][para]])
                    exact_match = sum([e[1] for e in evaluation_dict[task][unclear][para]])
                    CORR = sum([e[2]["CORR"] for e in evaluation_dict[task][unclear][para] if type(e[2]) == dict and e[2]["CORR"]!=None])
                    R2 = sum([e[2]["R2"] for e in evaluation_dict[task][unclear][para] if type(e[2]) == dict and e[2]["R2"]!=None])
                    MSE = sum([e[2]["MSE"] for e in evaluation_dict[task][unclear][para] if type(e[2]) == dict and e[2]["MSE"]!=None])
                    RMSE = sum([e[2]["RMSE"] for e in evaluation_dict[task][unclear][para] if type(e[2]) == dict and e[2]["RMSE"]!=None])
                    MAE = sum([e[2]["MAE"] for e in evaluation_dict[task][unclear][para] if type(e[2]) == dict and e[2]["MAE"]!=None])
                    MedAE = sum([e[2]["MedAE"] for e in evaluation_dict[task][unclear][para] if type(e[2]) == dict and e[2]["MedAE"]!=None])
                    # print(f"task: {task} ,unclear_taxonomy: {unclear}, given_para: {para}")
                    # print(f"total: {total}, executable: {executable / total}, exact_match: {exact_match / total}")
                    # print(
                    #     f"CORR: {CORR / total}, R2: {R2 / total}, MSE: {MSE / total}, RMSE: {RMSE / total}, MAE: {MAE / total}, MedAE: {MedAE / total}")
                    # writer.writerow(
                        # [task.value, unclear, para_new, total, executable, exact_match, CORR, R2, MSE, RMSE, MAE,
                        #  MedAE,len([e[2]["CORR"] for e in evaluation_dict[task][unclear][para] if type(e[2]) == dict])])
                    df.loc[len(df)] = [task.value, unclear, para_new, total, executable, exact_match, CORR, R2, MSE, RMSE, MAE,
                         MedAE,len([e[2]["CORR"] for e in evaluation_dict[task][unclear][para] if type(e[2]) == dict])]
    # csvfile.close()
    df.to_csv(eval_path, mode='a', index=False)

    # 1. 计算所有数据的ECR和pass rate  
    # ECR_all_data = df['executable'].sum() / np.float64(df['total'].sum())  
    ECR_all_data = df['executable'].sum() / 2425 
    pass_rate_all_data = df['exact_match'].sum() / np.float64(df['executable'].sum())  
    # pass_rate_all_data_2 = df['exact_match'].sum() / np.float64(df['total'].sum())  
    pass_rate_all_data_2 = df['exact_match'].sum() / 2425
    
    # 保存到DataFrame中  
    all_data_metrics = pd.DataFrame({  
        'total': [df['total'].sum()],
        'ECR': [ECR_all_data],  
        'pass_rate': [pass_rate_all_data],  
        'pass_rate_2': [pass_rate_all_data_2]
    })  

    print(all_data_metrics)
    
    # 2. 计算不同task的ECR和pass rate  
    task_grouped = df.groupby('task')  
    task_ECR_pass_rate = task_grouped.apply(lambda x: pd.Series({ 
        'total': x['total'].sum(),
        'ECR': x['executable'].sum() / np.float64(x['total'].sum()),  
        'pass_rate': x['exact_match'].sum() / np.float64(x['executable'].sum()),  
        'pass_rate_2': x['exact_match'].sum() / np.float64(x['total'].sum())
    })).reset_index()  

    print(task_ECR_pass_rate)
    
    # 3. 计算不同task不同given_query_taxonomy的ECR和pass rate  
    task_query_grouped = df.groupby(['task', 'given_query_taxonomy'])  
    task_query_ECR_pass_rate = task_query_grouped.apply(lambda x: pd.Series({  
        'total': x['total'].sum(),
        'ECR': x['executable'].sum() / np.float64(x['total'].sum()),  
        'pass_rate': x['exact_match'].sum() / np.float64(x['executable'].sum()),
        'pass_rate_2': x['exact_match'].sum() / np.float64(x['total'].sum())
    })).reset_index()  

    print(task_query_ECR_pass_rate)
    
    # 将结果保存到CSV文件  
    all_data_metrics.to_csv('all_data_metrics.csv', index=False)  
    task_ECR_pass_rate.to_csv('task_ECR_pass_rate.csv', index=False)  
    task_query_ECR_pass_rate.to_csv('task_query_ECR_pass_rate.csv', index=False)  

    return score


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4-32k-0613",
        choices=["codex-cushman", "codex001", "codex002", "incoder-1B", "gpt4","gpt-35-turbo-16k-0613","gpt-4-turbo","gpt-4-32k-0613"],
        help="Type of Codex Model to run",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="gpt_refined_outputs_0131_2",
        help="Path to which the Codex responses will be cached at",
    )
    parser.add_argument(
        "--source_dir",
        type=str,
        default="Tables/text2analysis_1220.csv",
        help="Path to the downloaded DS-1000 data",
    )

    args = parser.parse_args()

    text2analysis = Text2AnalysisDataset(source_dir=args.source_dir, is_eval=True)
    print("loaded dataset")
    test_model(
        text2analysis=text2analysis,
        model=args.model,
        output_dir=args.output_dir,
        args=args
    )
