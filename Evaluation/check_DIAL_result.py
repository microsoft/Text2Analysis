# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import csv
import json
import os
import pandas as pd
import traceback
from copy import deepcopy
from Common.safe_data import parse_literal
from Evaluation.checker import Checker, AnaTaxonomy
from Evaluation.evaluate import DATSET_PATH
def run_python(table,code, dtype=False):
    table_used_in_exec = deepcopy(table)
    local_var = {"tables": table_used_in_exec}
    function = code.split("```python")[1].split("```")[0] if "```" in code else code
    # import library. Note that the library must be imported before the function is executed.
    import_lib = [i for i in function.split("\n") if i.startswith("import ") or i.startswith("from ")]
    try:
        LIBVAR = locals()
        exec("\n".join(import_lib), globals(), LIBVAR)
        GLOBALVAR = globals()
        GLOBALVAR.update(LIBVAR)
        exec("\n".join([function]), GLOBALVAR, local_var)
        if "print" not in function:
            return None, function
        scripts = function.split("\n")
        for i in reversed(range(len(scripts))):
            script=scripts[i]
            if script.startswith("print") and script.strip().endswith(")"):
                result = script.split("print(")[1].split(",")[-1].split(")")[0].strip('\n ')
                break
            elif script.startswith("print"):
                result= scripts[i + 1].strip('\n ),')
                break
        else:
            return None, function
        if dtype:
            function+=f"""
result = pd.DataFrame({result}).T
print(result)
"""
            return run_python(table,function)
        exec(f"result = {result}", GLOBALVAR, local_var)
        return local_var["result"], function
    except Exception as e:
        print("[Exception]", "\n".join(traceback.format_exc().split("\n")[3:]))
        return "Error", function

def python_res2res(python_res):
    try:
        if type(python_res) is pd.DataFrame:
            python_res = python_res.astype(str).to_json(orient='split')
        elif type(python_res) is dict:
            for key in python_res:
                if type(python_res[key]) is pd.DataFrame:
                    python_res[key] = python_res[key].astype(str).to_json(orient='split')
            python_res= json.dumps(python_res)
        elif type(python_res) is list:
            for idx, item in enumerate(python_res):
                if type(item) is pd.DataFrame:
                    python_res[idx] = item.astype(str).to_json(orient='split')
            python_res = json.dumps(python_res)
        else:
            python_res = str(python_res)
    except:
        python_res = str(python_res)
    return python_res

if __name__ == '__main__':

    python = pd.read_csv(r"Python_0729_utf8.csv")
    # Output the correct results
    csvfile_right = open("Python_annotation_0811.csv", mode='w', newline='', encoding='utf-8')
    writer_right = csv.writer(csvfile_right)
    writer_right.writerow(
        ["table_name", "html", "query", "operations", "ambiguities", "dial_ori", "answer_ori", "dial_code",
         "dial_python", "answer_dial", "dial_ori_python", "python", "python_res", "ori_python", "ori_python_res", "correct",
         "python_anno"])
    # Output the wrong results
    csvfile = open("Python_0811.csv", mode='w', newline='', encoding='utf-8')
    writer = csv.writer(csvfile)
    writer.writerow(
        ["table_name", "html", "query", "operations", "ambiguities", "dial_ori", "answer_ori", "dial_code",
         "dial_python", "answer_dial", "dial_ori_python", "python", "python_res", "ori_python", "ori_python_res"])
    # Input annotation results
    annotation_path=['project-5-at-2023-08-07-13-25-d0038491.csv']
    annotation_table=[]
    for path in annotation_path:
        annotation_table.append(pd.read_csv(path))
    annotation_table=pd.concat(annotation_table,ignore_index=True)

    # For groundtruth
    right_num = 0
    no_answer_num = 0
    # For annotation
    out_of_scope_num = 0
    correct_num = 0
    incorrect_num = 0

    for idx, row in python.iterrows():
        # The query has groundtruth
        if not pd.isna(row["answer_ori"]):
            try:
                answer_ori = json.loads(row["answer_ori"])
            except:
                try:
                    if "\\\"" in row["answer_ori"]:
                        answer_ori= json.loads(row["answer_ori"].replace("\\\"", "\""))
                    else:
                        answer_ori = parse_literal(row["answer_ori"], "answer_ori")
                except:
                    answer_ori=row["answer_ori"]
                    print(row["answer_ori"])

            # load python_res
            if type(row["python_res"])==str and "Exception: KeyError:" in row["python_res"]:
                python = row["python"]
                if row["table_name"].endswith("csv"):
                    table = pd.read_csv(os.path.join(DATSET_PATH[AnaTaxonomy.L1], row["table_name"]))
                else:
                    table = pd.read_excel(os.path.join(DATSET_PATH[AnaTaxonomy.L1], row["table_name"]))
                python_res, python = run_python(table,python)
                row["python"] = python
                row["python_res"] = python_res2res(python_res)
            try:
                python_res = json.loads(row["python_res"])
            except:
                try:
                    if "\\\"" in row["python_res"]:
                        python_res = json.loads(row["python_res"].replace("\\\"", "\""))
                    else:
                        python_res = parse_literal(row["python_res"], "python_res")
                except:
                    python_res=row["python_res"]
                    if type(python_res)==str and  "dtype:" in python_res:
                        python=row["python"]
                        if row["table_name"].endswith("csv"):
                            table = pd.read_csv(os.path.join(DATSET_PATH[AnaTaxonomy.L1], row["table_name"]))
                        else:
                            table = pd.read_excel(os.path.join(DATSET_PATH[AnaTaxonomy.L1], row["table_name"]))
                        python_res,python = run_python(table, python,dtype=True)
                        row["python"]=python
                        row["python_res"]=python_res2res(python_res)
                    else:
                        print("!!!!!!!!!!!!!!!!!",row["python_res"])
                        print("answer_ori",row["answer_ori"])

            if type(answer_ori) == dict:
                answer_ori = pd.read_json(json.dumps(answer_ori))

            if type(python_res) == dict:
                try:
                    python_res = pd.read_json(json.dumps(python_res), orient='split')
                except:
                    print(row["python_res"])
            check = Checker('', "a='-----------------'\nprint(a)", "", "", [AnaTaxonomy.Aggregation])

            if check.isMatch(answer_ori, python_res):
                row["correct"] = "Correct"
                row["python_anno"] = ""
                writer_right.writerow(row)
                right_num += 1
                continue
            else:
                if row["table_name"] == "Employee.csv" and "Annual Salary" in row["python"] and "replace('$'" not in row["python"]:
                    row["python"] = """
# Convert 'Annual Salary' column to numeric by removing '$' and ',' characters and converting to float
tables['Annual Salary'] = tables['Annual Salary'].str.replace('$', '').str.replace(',', '').astype(float)
                            """ + row["python"]
                    if row["table_name"].endswith("csv"):
                        table = pd.read_csv(os.path.join(DATSET_PATH[AnaTaxonomy.L1], row["table_name"]))
                    else:
                        table = pd.read_excel(os.path.join(DATSET_PATH[AnaTaxonomy.L1], row["table_name"]))
                    python_res = run_python(table, row["python"])[0]
                    row["python_res"] = python_res2res(python_res)
                if check.isMatch(answer_ori, python_res):
                    row["correct"] = "Correct"
                    row["python_anno"] = ""
                    writer_right.writerow(row)
                    right_num += 1
                    continue
                else:
                    print(row["query"])
                    print(f"[Wrong]: answer_ori:{answer_ori}")
                    print(f"[Wrong]: python_res:{python_res}")

        # The query has groundtruth
        annotation = annotation_table.loc[
            (annotation_table['table_name'] == row["table_name"]) & (annotation_table["query"] == row[
                "query"]), ["correct", "python_anno", 'python', 'ambiguities_new','operations_new']].values
        if len(annotation) > 0:
            correct=annotation[0][0]
            python_anno=annotation[0][1]
            python=annotation[0][2]
            ambiguities_new=annotation[0][3]
            operations_new=annotation[0][4]
            if correct=="Out of scope":
                out_of_scope_num += 1
                continue
            elif correct=="Incorrect":
                writer.writerow(row)
                incorrect_num+= 1
                continue
            else:
                correct_num += 1
                if not pd.isna(python_anno):
                    if python_anno.startswith("{"):
                        python_anno=json.loads(python_anno)["text"][0]
                    python=python_anno
                if row["table_name"].endswith("csv"):
                    table = pd.read_csv(os.path.join(DATSET_PATH[AnaTaxonomy.L1], row["table_name"]))
                else:
                    table = pd.read_excel(os.path.join(DATSET_PATH[AnaTaxonomy.L1], row["table_name"]))
                python_res=run_python(table,python)[0]
                row["python"] = python
                row["python_res"] = python_res2res(python_res)
                row["correct"] = "Correct"
                row["python_anno"] = ""
                writer_right.writerow(row)
                continue
        else:
            writer.writerow(row)
            no_answer_num += 1
    print(f"right_num:{right_num}, no_answer_num:{no_answer_num}, out_of_scope_num:{out_of_scope_num}, correct_num:{correct_num}, incorrect_num:{incorrect_num}")
