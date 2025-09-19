# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import traceback

import pandas as pd

from GCR.run import CodeLLM, Role


class AutoDebug(object):
    _max_retries: int = 5
    _rows_to_display: int = 5

    _PYTHONLIB = ["import pandas as pd", ]

    _DEBUGGING_INSTRUCTION = """You are a Python code debugging assistant.
You have to follow the rules:
1. You must correct the Python function and return a right one that fixes the given error.
2. You must generate the right function with the same signature as the given Python code.
3. You must take the previous messages as examplars.
4. The function must return one variable being one of pd.DataFrame, List[int], List[float], List[str], int, float and str.
5. There is already a DataFrame variable for `tables` in the environment, if the code reassigns `tables=pd.DataFrame(...)` in the code, please delete the it.
6. If there are any result, print the result at the end.
7. We want to get result without user input, so please do not use `input()` in the code.
"""
    _DEBUGGING_QUERY_TEMPLATE = """Pandas dataframe (tables) with {num_rows} rows and {num_columns}  columns.
The output of `print(tables.head({rows_to_display}).to_csv(index=False))` is:
<s>
{table_head}
</s>
tables columns:
<s>
{table_columns}
</s>
column types: 
<s>
{column_types}
</s>
The user input the following query:
<s>
{query}
</s>
You generated this python code:
<s>
{function}
</s>
fails with the following error:
<s>
{error_returned}
</s>
"""

    _DEBUGGING_EXAMPLES = [
        {
            Role.USER: """
Pandas dataframe (tables) with 10 rows and 4 columns.
The output of `print(tables.head(2).to_csv(index=False))` is:
<s>
Island,Nickname,Area,Population (as of 2010)
Hawai'i, The Big Island, 4,028.0 sq mi, 185,079
Maui, The Valley Isle, 727.2 sq mi, 144,444
</s>
tables columns:
<s>
[ 'Island', 'Nickname', 'Area', 'Population (as of 2010)' ]
</s>
column types: 
<s>
{"Island": "string", "Nickname": "string", "Area": "string", "Population (as of 2010)": "string"}
</s>
The user input the following query:
<s>
Can you tell me the total population?
</s>
You generated this python code:
<s>
def total_population(tables: pd.DataFrame) -> int:

    # Selecting only 'Population (as of 2010)' column
    population = tables[['Population (as of 2010)']]

    # Calculating total population
    total_population = tables['Population (as of 2010)'].sum()
    
    # Returning the total population
    return int(total_population)
    
total_population(tables)
</s>
fails with the following error:
<s>
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<stdin>", line 4, in total_population
ValueError: invalid literal for int() with base 10: '185,079144,444'
</s>
""",
            Role.ASSISTANT: """def total_population(tables: pd.DataFrame) -> int:

    # Converting the 'Population (as of 2010)' column from string to numeric
    tables['Population (as of 2010)'] = pd.to_numeric(tables['Population (as of 2010)'])

    # Selecting only 'Population (as of 2010)' column
    population = tables[['Population (as of 2010)']]

    # Calculating total population
    total_population = tables['Population (as of 2010)'].sum()
    
    # Returning the total population
    return int(total_population)
total_population(tables)
# END"""
        }
    ]

    _llm: CodeLLM = None

    def __init__(self, llm) -> None:
        self._llm = llm

    def _run_code(self, query, function: str, table: pd.DataFrame, show_code: bool = False, prompt: str = None):
        self._table_head = table.head(5).to_csv(index=False)
        if prompt is not None:
            self._DEBUGGING_INSTRUCTION += "\n" + prompt

        column_dtype = table.dtypes.to_dict()
        column_dtype = {str(k): str(v) for k, v in column_dtype.items()}
        self._column_dtype = json.dumps(column_dtype)
        self._query = query

        LIBVAR = locals()
        exec("\n".join(self._PYTHONLIB), globals(), LIBVAR)
        GLOBALVAR = globals()
        GLOBALVAR.update(LIBVAR)

        cnt = 0
        last_exception = None
        while cnt < self._max_retries:
            local_var = {"tables": table}
            try:
                function = function.split("```python")[1].split("```")[0] if "```" in function else function
                # import library. Note that the library must be imported before the function is executed.
                import_lib = [i for i in function.split("\n") if i.startswith("import ") or i.startswith("from ")]
                LIBVAR = locals()
                exec("\n".join(import_lib), globals(), LIBVAR)
                GLOBALVAR = globals()
                GLOBALVAR.update(LIBVAR)

                exec("\n".join([function]), GLOBALVAR, local_var)
                if "print" not in function:
                    return None, function
                scripts = function.split("\n")
                for script in reversed(scripts):
                    if script.startswith("print"):
                        result = script.split("print(")[1].split(",")[-1].split(")")[0]
                        break
                return local_var[result], function
            except Exception as e:
                cnt += 1
                last_exception = "\n".join(traceback.format_exc().split("\n")[3:])
                print("[Exception]", last_exception)
                print("[Auto Debugging]")
                function = self._debug_code(function, str(last_exception), table)

                if show_code:
                    print("Corrected Code".center(64, "-"))
                    print(function)

        print(f"Auto debugging excel max_retries = {self._max_retries}.")
        return "Exception: " + str(last_exception), function

    def _debug_code(self, function: str, observation: str, table: pd.DataFrame):
        debug_prompt = self._DEBUGGING_QUERY_TEMPLATE.format(num_rows=table.shape[0],
                                                             num_columns=table.shape[1],
                                                             rows_to_display=self._rows_to_display,
                                                             table_head=self._table_head,
                                                             table_columns=str(list(table.columns)),
                                                             column_types=self._column_dtype,
                                                             query=self._query,
                                                             function=function,
                                                             error_returned=observation
                                                             )

        return self._llm._completion(self._DEBUGGING_INSTRUCTION, self._DEBUGGING_EXAMPLES, debug_prompt)


if __name__ == '__main__':
    code = """
```python
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Add 1 to the year in 'datetime' column
tables['datetime'] = tables['datetime'].str.replace('0000', '2000')

# Convert 'datetime' column to pandas datetime format
tables['datetime'] = pd.to_datetime(tables['datetime'], format='%d-%b-%Y %H:%M:%S', errors='coerce')

# Drop rows with invalid datetime values
tables = tables.dropna(subset=['datetime'])

# GroupBy and Aggregate
volatility_by_year = tables.groupby(tables['datetime'].dt.year)["data_mean_global"].std().reset_index().rename(columns={"datetime": "year", "data_mean_global": "Volatility"})

# Evaluate Trend
x = volatility_by_year["year"].values.reshape(-1, 1)
y = volatility_by_year["Volatility"].values
reg = LinearRegression().fit(x, y)
volatility_trend = reg.coef_[0]

# Forecast
forecast_steps = 10
years_to_forecast = np.arange(x[-1] + 1, x[-1] + 1 + forecast_steps).reshape(-1, 1)
volatility_forecast_values = reg.predict(years_to_forecast)
volatility_forecast = pd.DataFrame({"year": years_to_forecast.squeeze(), "Volatility": volatility_forecast_values})

# Print results
print("Volatility Trend:", volatility_trend)
print("Volatility Forecast:")
print(volatility_forecast)
# END
```
"""
    query = "Can you provide guidance on how to achieve a {20%} enhancement in {median_house_value} specifically for {county} {Alameda}?"
    df = pd.read_excel(
        r"C:\Users\v-xinyihe\Documents\Repo\azure\DIAL2\DIAL\Query2DIALPython\bin\Debug\net7.0\TestData\clif_train.xlsx")

    print(df)

    llm = CodeLLM()
    auto_pd = AutoDebug(llm)

    res = auto_pd._run_code(query, code, df, show_code=True)
    print(res)

    print("result".center(64, '-'))
    print(res)
