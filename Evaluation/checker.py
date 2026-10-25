# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import re
import traceback
from typing import List
from copy import deepcopy
import numpy as np
import pandas as pd
from greykite.common.evaluation import calc_pred_err
from sklearn import metrics

from Common.taxonomy import AnaTaxonomy
from Common.safe_data import DataFormatError, MAX_CELL_BYTES, parse_literal

FLOAT_PRECISION = 2

# FORECASTING_GT = pd.read_csv(r"code2query 1.csv")
FORECASTING_GT = pd.read_csv(r"Tables/result 0815.csv")
FORECASTING_GT['query'] = FORECASTING_GT['query'].apply(lambda x: x.replace(" ", "").lower())


def get_L2_anatype(type_list):
    for i in type_list:
        if i in [AnaTaxonomy.Forecasting, AnaTaxonomy.Insights, AnaTaxonomy.Chart]:
            return i
    return AnaTaxonomy.L1


def to_str(value, floatPrecision):
    if type(value) == str:
        value = value.strip()
    if type(value) in (float, np.float32, np.float64, int, np.int32, np.int64):
        ret = ("{:.%df}" % floatPrecision).format(value)
    elif type(value) == str:
        try:
            num = float(value)
            ret = ("{:.%df}" % floatPrecision).format(num)
        except ValueError:
            ret = value
    else:
        ret = str(value)
    return ret.lower()

def inside_table(table1:np.ndarray,table2:np.ndarray):
    if table1.shape[1]==0 or table2.shape[1]==0 or table1.shape[0]!=table2.shape[0]:
        return False
    if table1.shape[1]>table2.shape[1]:
        big_table=table1
        small_table=table2
    else:
        big_table=table2
        small_table=table1
    for i in range(small_table.shape[1]):
        for j in range(big_table.shape[1]):
            if (small_table[:, i] == big_table[:, j]).all():
                break
        else:
            return False
    return True

def inside_table_record(table1:np.ndarray,table2:np.ndarray):
    if table1.shape[0]==0 or table2.shape[0]==0 or table1.shape[1]!=table2.shape[1]:
        return False
    if table1.shape[0]>table2.shape[0]:
        big_table=table1
        small_table=table2
    else:
        big_table=table2
        small_table=table1
    for i in range(small_table.shape[0]):
        for j in range(big_table.shape[0]):
            if (small_table[i,:] == big_table[ j,:]).all():
                break
        else:
            return False
    return True


def json2df(json):
    return pd.read_json(json, orient='split')


def remove_read_table(lines, replace_string=None):
    star_idx = -1
    end_idx = -1
    for idx, line in enumerate(lines):
        if "= pd.read_" in line and star_idx == -1:
            star_idx = idx if star_idx == -1 else star_idx
        if star_idx != -1 and ")" in line:
            end_idx = idx
            break
    else:
        star_idx = -1
    if star_idx != -1:
        lines[star_idx] = lines[star_idx].split("=")[0] + "=table" if replace_string is None else replace_string
        for idx in range(star_idx + 1, end_idx + 1):
            lines[idx] = ""
    return lines


def check_load_table(function):
    if "input(" in function:
        lines = function.split("\n")
        for idx, line in enumerate(lines):
            if "input(" in line:
                lines[idx] = re.sub(r'input\([^)]*\)', '\"\"', line)
        function = "\n".join(lines)
    # For StarChat chart
    """
# Read the HTML tables into a pandas DataFrame
tables = pd.read_html('your HTML table code here')

# Select the first table in the list of DataFrames
table = tables[0]
        """
    if """tables[0]""" in function and "tables = pd.read_" in function:
        # Remove lines starting with #
        filtered_lines = '\n'.join(
            [line.split("=")[0] for line in function.split('\n') if not line.strip().startswith('#')])

        # count_tables
        word_count = filtered_lines.lower().count('tables')

        if word_count == 1:
            lines = function.split("\n")
            lines = remove_read_table(lines, replace_string="")
            function = "\n".join(lines)
            function = function.replace("""tables[0]""", "table")

    # For chart in gpt4
    if """tables[0]""" in function:
        # Remove lines starting with #
        filtered_lines = '\n'.join(
            [line.split("=")[0] for line in function.split('\n') if not line.strip().startswith('#')])

        # count_tables
        word_count = filtered_lines.lower().count('tables')

        if word_count == 0:
            function = function.replace("""tables[0]""", "table")

    # For StarChat
    if "= pd.read_" in function:
        if "<" in function:
            pass
        lines = function.split("\n")
        lines = remove_read_table(lines)
        function = "\n".join(lines)

    return function


def eval_python(code, table,has_result=False):
    local_var = {"tables": table}
    try:
        # print(code)
        function = code.split("```python")[1].split("```")[0] if "```" in code else code.strip(".")
        function = check_load_table(function)
        # print(function)
        # import library. Note that the library must be imported before the function is executed.
        import_lib = [i for i in function.split("\n") if i.startswith("import ") or i.startswith("from ")]
        LIBVAR = locals()
        exec("\n".join(import_lib), globals(), LIBVAR)
        GLOBALVAR = globals()
        GLOBALVAR.update(LIBVAR)

        exec("\n".join([function]), GLOBALVAR, local_var)
        return local_var["result"]
    except:
        exc_info=traceback.format_exc()
        error_line = None
        if "File \"<string>\"," in exc_info:
            pattern = r'File "<string>", line (\d+)'  
            match = re.search(pattern, exc_info)  
            
            if match:  
                line_number = match.group(1)  
                error_line=function.split("\n")[int(line_number)-1]

        # print("[KeyError]", "\n".join(traceback.format_exc().split("\n")[3:]))
        # print(f"Error line: {error_line}")
        # print(f"Table: {table}")
        # print(f"Code: {function}")
        return "Error", traceback.format_exc(), error_line


def cut_forecast_table(table, delete_line, table_name=None):
    t_field = 0
    if table_name == "hourly_bikesharing.csv" or table_name == "passenger.csv":
        t_field = 1
    elif table_name == "carbon_dioxide.csv":
        t_field = 4
    table.iloc[:, t_field] = pd.to_datetime(table.iloc[:, t_field].astype(str))  # TODO: time is not the first column

    # The first field is time
    # Find the max date
    max_date = table.iloc[:, t_field].max()

    # # Calculate cutoff time
    if delete_line[-1] == "Y":
        cutoff_date = max_date - pd.DateOffset(years=int(delete_line[:-1]))
    elif delete_line[-1] == "M":
        cutoff_date = max_date - pd.DateOffset(months=int(delete_line[:-1]))
    elif delete_line[-1] == "D":
        cutoff_date = max_date - pd.DateOffset(days=int(delete_line[:-1]))
    elif delete_line[-1] == "W":
        cutoff_date = max_date - pd.DateOffset(weeks=int(delete_line[:-1]))
    elif delete_line[-1] == "H":
        cutoff_date = max_date - pd.DateOffset(hours=int(delete_line[:-1]))
    else:
        raise ValueError(f"Invalid time format: {delete_line}")

    table = table[table.iloc[:, t_field] <= cutoff_date]
    return table


class Checker:
    def __init__(self, table, code_pre, code_gt, ground_truth, task: List[AnaTaxonomy], ori_query=None,
                 code_pre_result=None):
        self.code_pre = code_pre
        self.code_gt = code_gt
        self.ground_truth = self.load_string_result(ground_truth, orient='records') if AnaTaxonomy.Forecasting in task \
            else self.load_string_result(ground_truth)
        self.task = task
        if AnaTaxonomy.Forecasting not in task:
            self.code_gt_result = self.ground_truth
        else:
            if "code_result" not in FORECASTING_GT:
                delete_line = FORECASTING_GT.loc[
                    FORECASTING_GT["query"] == ori_query.replace(" ", "").lower(), ["delete_lines"]].values[0][0]
                table = cut_forecast_table(table, delete_line)
                self.code_gt_result = eval_python(code_gt, table)
            else:
                code_gt_result = FORECASTING_GT.loc[
                    FORECASTING_GT["query"] == ori_query.replace(" ", "").lower(), ["code_result"]].values[0][0]
                self.code_gt_result = self.load_string_result(code_gt_result)

        if code_pre_result is None:
            result = eval_python(code_pre, table,True)
            if type(result)==tuple and len(result) == 3 and result[0]=="Error":
                self.code_pre_result = result[0]
                self.traceback = result[1]
                self.error_line=result[2]
            else:
                self.code_pre_result = result
                self.traceback = None
                self.error_line=None
        else:
            self.code_pre_result = self.load_string_result(
                code_pre_result) if task != AnaTaxonomy.Forecasting else self.load_string_result(code_pre_result,
                                                                                                 orient='both')

    def read_json(self, result,orient="split"):
        if orient!="both":
            result=pd.read_json(json.dumps(result), orient=orient)
        else:
            if "columns" in result and "data" in result:
                result = pd.read_json(json.dumps(result), orient="split")
            else:
                result = pd.read_json(json.dumps(result), orient="records")
        return result

    def load_string_result(self, result_string, orient='split'):  # TODO: dict of dataframe,orient='records'
        if isinstance(result_string, str) and len(result_string.encode("utf-8")) > MAX_CELL_BYTES:
            raise DataFormatError("result: text exceeds the size limit")
        try:
            result = parse_literal(result_string, "result")
        except DataFormatError:
            if not isinstance(result_string, str) or result_string.lstrip().startswith(("[", "{", "(")):
                raise
            result = result_string
        if type(result) == dict:
            try:
                result=self.read_json(result,orient)
            except:
                for key in result:
                    try:
                        result[key]=self.read_json(result[key],orient)
                    except:
                        pass
        elif type(result) == list:
            try:
                result=self.read_json(result,orient)
            except:
                for idx, item in enumerate(result):
                    try:
                        result[idx]=self.read_json(item,orient)
                    except:
                        pass
        if (type(result) == list and len(result) != 0 and type(result[0]) != pd.DataFrame) or \
                (type(result) == dict and len(result) != 0 and type(list(result.values())[0]) != pd.DataFrame):
            print(f"[This is not a DataFrame]: {result}")
        return result

    def check_result(self) -> (AnaTaxonomy, bool, bool, bool):
        if type(self.code_pre_result) == str and self.code_pre_result == "Error":
            executable_code_ratio = exact_match = accuracy = 0
        elif AnaTaxonomy.Forecasting in self.task:
            executable_code_ratio, exact_match, accuracy = self.check_forcasting()
        # elif AnaTaxonomy.Clustering in self.task:
        #     executable_code_ratio, exact_match, accuracy = self.check_clustering()
        elif AnaTaxonomy.Chart in self.task:
            executable_code_ratio, exact_match, accuracy = self.check_chart()
        else:
            executable_code_ratio = 1
            exact_match = accuracy = self.isMatch()
        if not exact_match:
            print(f"[!!!ground truth]: {self.code_gt_result}")
            print(f"[!!!code_pre_result]: {self.code_pre_result}")
        task = get_L2_anatype(self.task)
        return task, executable_code_ratio, exact_match, accuracy  # TODO: forecast & 0 how to compare

    def check_forcasting(self) -> (bool, bool, dict):
        if type(self.code_pre_result) == pd.DataFrame and (type(self.ground_truth) == pd.DataFrame or \
         (self.ground_truth.shape[0] == 1 and type(self.code_pre_result) in (float, np.float32, np.float64, int, np.int32, np.int64))): # TODO: what about dict
            if type(self.code_pre_result) != pd.DataFrame:
                fake_table = self.ground_truth
                fake_table.iloc[0, 1] = deepcopy(self.code_pre_result)
                self.code_pre_result = fake_table
            if self.ground_truth.shape[0] != self.code_pre_result.shape[0] or self.ground_truth.shape[1]<2 or self.code_pre_result.shape[1]<2:
                return 1, 0, 0
            print(f"[ground truth]: {self.ground_truth}")
            print(f"[code_pre_result]: {self.code_pre_result}")
            try:
                metrics = calc_pred_err(self.ground_truth.iloc[:, 1], self.code_pre_result.iloc[:, 1])
                max_value=max(self.ground_truth.iloc[:, 1].max(),self.code_pre_result.iloc[:, 1].max())
                for key in ["MSE","RMSE","MAE","MedAE"]:
                    metrics[key]=metrics[key]/max_value
            except:
                metrics = 0
        else:
            metrics = 0
        exact_match = self.isMatch(self.code_gt_result.iloc[:, :2] if type(self.code_gt_result)==pd.DataFrame else self.code_gt_result,
                                   self.code_pre_result.iloc[:, :2]  if type(self.code_pre_result)==pd.DataFrame else self.code_pre_result,
                                   can_inside_table_record=True)
        return 1, int(exact_match), metrics

    def check_clustering(self) -> (bool, bool, bool):
        '''
        code_pre_result:
        {
            "result": dict, # the result of clustering, dataframe.to_json(orient='split')
            "eval": {
                "davis_bouldin_score": float,
                "calinski_harabasz_score": float,
                "silhouette_score": float,
            },
            "measures": List[str], # the measure field names in the clustering
        }
        '''

        def evaluation_davis_bouldin_score(dimension, data, labels):
            score = 0
            for dim in dimension:
                score += -metrics.davies_bouldin_score([data[dim][record] for record in range(data[dim].shape[0])],
                                                       labels)
            return score

        def evaluation_calinski_harabasz_score(dimension, data, labels):
            score = 0
            for dim in dimension:
                score += metrics.calinski_harabasz_score([data[dim][record] for record in range(data[dim].shape[0])],
                                                         labels)
            return score

        def evaluation_silhouette_score(dimension, data, labels):
            from sklearn import metrics
            score = 0
            for dim in dimension:
                score += metrics.silhouette_score([data[dim][record] for record in range(data[dim].shape[0])], labels)
            return score

        pre_table = json2df(self.code_pre_result["result"])
        gt_table = json2df(self.result_gt["result"])
        executable_code_ratio = self.isMatch(pre_table, gt_table)
        correctness = self.isMatch(self.code_pre_result["measures"], self.result_gt['measures'])
        accuracy = self.isMatch(self.code_pre_result["eval"], self.result_gt["eval"])
        return executable_code_ratio, correctness, accuracy

    def check_chart(self) -> (bool, bool, bool):
        '''
        code_pre_result:
        {
            "chart_type": str, # choose from [lineChart, barChart, pieChart, scatterChart]
            "x_fields": str, # the x field name of the chart
            "y_fields": List[str], # the y field names of the chart
        }
        '''
        print(self.code_pre_result)
        if type(self.code_pre_result) != dict or \
                "chart_type" not in self.code_pre_result or "x_fields" not in self.code_pre_result or "y_fields" not in self.code_pre_result \
                or type(self.code_pre_result["chart_type"]) != str or type(self.code_pre_result["x_fields"]) != str or \
                type(self.code_pre_result["y_fields"]) != list or len(
            [i for i in self.code_pre_result["y_fields"] if type(i) != str]) != 0:
            return 1, False, False
        # if self.code_pre is not None and "matplotlib" not in self.code_pre: # TODO: Add or not add?
        #     return 1, False, False

        if self.code_pre_result["chart_type"] != self.ground_truth["chart_type"] or self.code_pre_result["x_fields"] != \
                self.ground_truth["x_fields"] or set(self.code_pre_result["y_fields"]) != set(
            self.ground_truth["y_fields"]):
            eval = False
        else:
            eval = True
        return 1, eval, eval

    def isMatch(self, result_pre=None, result_gt=None, has_sort=None,can_inside_table_record=False) -> bool:
        if result_pre is None:
            result_pre = self.code_pre_result
        if result_gt is None:
            result_gt = self.ground_truth

        def simplify(x):
            if type(x) == pd.Series:
                x = x.to_frame().T
            if type(x) == pd.DataFrame:
                x = x.astype(str)
            if type(x) == pd.DataFrame and x.shape[0] == 1 and x.shape[1] == 1:
                return x.iloc[0, 0]
            elif type(x) == pd.DataFrame and x.shape[1] == 1:
                return x.iloc[:, 0].tolist()
            elif type(x) == pd.DataFrame and (x.shape[0] == 0 or x.shape[1] == 0):
                return []
            elif type(x) == list and len(x) == 1:
                return x[0]
            elif type(x) == dict and len(x) == 1:
                return x[list(x.keys())[0]]
            else:
                return x

        result_gt = simplify(result_gt)
        result_pre = simplify(result_pre)

        if type(result_gt) in (float, np.float32, np.float64, int, np.int32, np.int64, str, bool, np.bool_) and \
                type(result_pre) in (float, np.float32, np.float64, int, np.int32, np.int64, str, bool, np.bool_):
            return to_str(result_gt, FLOAT_PRECISION) == to_str(result_pre, FLOAT_PRECISION)
        elif type(result_gt) == list and type(result_pre) == pd.DataFrame:
            return self.isMatch(result_gt, result_pre.iloc[:, 0].tolist())
        elif type(result_gt) == pd.DataFrame and type(result_pre) == list:
            return self.isMatch(result_gt.iloc[:, 0].tolist(), result_pre)
        elif type(result_gt) == pd.DataFrame and type(result_pre) in (
                float, np.float32, np.float64, int, np.int32, np.int64, str, bool, np.bool_):
            return any([self.isMatch(i, result_pre) for i in result_gt.iloc[:, 0].tolist()]) or any(
                [self.isMatch(i, result_pre) for i in result_gt.iloc[0, :].tolist()])
        elif type(result_gt) in (float, np.float32, np.float64, int, np.int32, np.int64, str, bool, np.bool_) and type(
                result_pre) == pd.DataFrame:
            return any([self.isMatch(result_gt, i) for i in result_pre.iloc[:, 0].tolist()]) or any(
                [self.isMatch(result_gt, i) for i in result_pre.iloc[0, :].tolist()])
        elif type(result_pre) == tuple:
            return any([self.isMatch(result_gt, i) for i in result_pre])
        elif type(result_pre) != type(result_gt):
            return False
        elif type(result_gt) == list and type(result_pre) == list:
            list_ans1 = [to_str(item, FLOAT_PRECISION) for item in result_gt]
            list_ans2 = [to_str(item, FLOAT_PRECISION) for item in result_pre]
            return list_ans1 == list_ans2
        elif type(result_gt) == pd.DataFrame and type(result_pre) == pd.DataFrame:
            ans1_norm = result_gt.applymap(lambda x: to_str(x, FLOAT_PRECISION))
            ans2_norm = result_pre.applymap(lambda x: to_str(x, FLOAT_PRECISION))

            if (has_sort is None and "sort" not in self.code_gt) or has_sort ==False:
                ans1_norm = ans1_norm.sort_values(by=list(ans1_norm.columns))
                ans2_norm = ans2_norm.sort_values(by=list(ans2_norm.columns))

            # Remove header
            data1 = ans1_norm.to_numpy()
            data2 = ans2_norm.to_numpy()

            # compare values
            if data1.shape[0] != data2.shape[0] and can_inside_table_record:
                return inside_table_record(data1, data2)
            elif data1.shape[0] != data2.shape[0]:
                return False
            else:
                return inside_table(data1, data2)
        elif type(result_gt) == dict and type(result_pre) == dict:
            for key in result_gt.keys():
                if key not in result_pre or not self.isMatch(result_gt[key], result_pre[key]):
                    return False
            return True
        else:
            raise Exception(f"Unknown type: {type(result_gt)}")


if __name__ == '__main__':
    check = Checker('', "a=1\nprint(a)", "", "", [AnaTaxonomy.Aggregation])
    A = pd.DataFrame({"Gender": ["Male", "Female"], "EEID": [482, 518], "test": [1, 2]})
    B = pd.DataFrame({"Gender": ["Female", "Male"], "Employee Count": [518, 482]})
    print(A)
    print(B)
    print(check.isMatch(A, B))
    check = Checker('', "a=1\nprint(a)", "", "", [AnaTaxonomy.Forecasting])
    check.result_gt = A.to_json(orient='split')
    check.code_pre_result = B.to_json(orient='split')
    print(check.check_forcasting())
