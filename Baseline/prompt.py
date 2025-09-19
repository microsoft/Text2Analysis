# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from GCR.run import Role


def get_examples(need_role=False):
    output = []
    for example in examples:
        if not need_role:
            output.append("\r\n".join([
                "<example>",
                "Given the HTML tables:",
                example["html"],
                "The user query is:",
                example["query"],
                "==>",
                example["code"],
                "</example>"
            ]))
        else:
            output.append({Role.USER: "\r\n".join([
                "Given the HTML tables:",
                example["html"],
                "The user query is:",
                example["query"],
            ]),
                Role.ASSISTANT: example["code"]
            })
    if not need_role:
        return "\r\n".join(output)
    else:
        return output


examples = [
    {
        "html": """
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Year</th>
      <th>Brand</th>
      <th>Category</th>
      <th>Model</th>
      <th>Sales</th>
      <th>Costs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2007</td>
      <td>BMW</td>
      <td>Compact</td>
      <td>BMW 3-Series</td>
      <td>142490</td>
      <td>99743</td>
    </tr>
    <tr>
      <td>2008</td>
      <td>BMW</td>
      <td>Compact</td>
      <td>BMW 3-Series</td>
      <td>112464</td>
      <td>78725</td>
    </tr>
    <tr>
      <td>2009</td>
      <td>BMW</td>
      <td>Compact</td>
      <td>BMW 3-Series</td>
      <td>90960</td>
      <td>63672</td>
    </tr>
    <tr>
      <td>2010</td>
      <td>BMW</td>
      <td>Compact</td>
      <td>BMW 3-Series</td>
      <td>100910</td>
      <td>70637</td>
    </tr>
    <tr>
      <td>2011</td>
      <td>BMW</td>
      <td>Compact</td>
      <td>BMW 3-Series</td>
      <td>94371</td>
      <td>66060</td>
    </tr>
  </tbody>
</table>""",
        "query": "forecast the cost data of BMW in next coming year.",
        "code": """
The output code is :
```python
import pandas as pd
from prophet import Prophet
temporal_field = "Year"
forecast_field = "Sales"
confidence_level = 0.95
freq = "Y"
steps = 1
df_G1 = table[(table["Brand"] == "BMW")]
df_G1 = df_G1.groupby(temporal_field)[forecast_field].sum().reset_index()
df_G1[temporal_field] = pd.to_datetime(df_G1[temporal_field], format="%Y")
df_G1[temporal_field] = pd.to_datetime(df_G1[temporal_field])
df_G1.rename(columns={temporal_field: "ds", forecast_field: "y"}, inplace=True)
m = Prophet(interval_width=confidence_level)
m.fit(df_G1)
future = m.make_future_dataframe(periods=steps, freq=freq)
future_forecast = m.predict(future)[["ds", "yhat"]].tail(steps)
future_forecast['yhat'] = future_forecast['yhat'].apply(lambda x: '{:.0f}'.format(x))
print(future_forecast)
```
"""
    },
    {
        "html": """
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>DH</th>
      <th>HH</th>
      <th>Frequency</th>
      <th>Cumulative %</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>do</td>
      <td>5500.0</td>
      <td>5</td>
      <td>5.95%</td>
    </tr>
    <tr>
      <td>5500</td>
      <td>6500.0</td>
      <td>6</td>
      <td>13.10%</td>
    </tr>
    <tr>
      <td>6500</td>
      <td>7500.0</td>
      <td>11</td>
      <td>26.19%</td>
    </tr>
    <tr>
      <td>7500</td>
      <td>8500.0</td>
      <td>15</td>
      <td>44.05%</td>
    </tr>
    <tr>
      <td>8500</td>
      <td>9500.0</td>
      <td>16</td>
      <td>63.10%</td>
    </tr>
</tbody>
</table>""",
        "query": "Help me create a line chart to visualize the 'Cumulative %' field for the 'HH' field.\nplease print the result dictionary at the end.",
        "code": """
The output code is :
```python
import pandas as pd
import matplotlib.pyplot as plt

# Assuming 'table' is a DataFrame

# Get the categories and values from the DataFrame
categories = table['HH'].tolist()
values = table['Cumulative %'].tolist()

# Create the chart
plt.plot(categories, values)

# Add title and labels for the axes
plt.title('line Chart')
plt.xlabel('HH')
plt.ylabel('Cumulative %')

# Show the plot
plt.show()

result = {'x_fields': 'HH', 'y_fields': ['Cumulative %'], 'chart_type': 'lineChart'}
print(result)

```
"""
    },
    {
        "html": """
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Name</th>
      <th>Color</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>A</td>
      <td>Red</td>
    </tr>
    <tr>
      <td>A</td>
      <td>Red</td>
    </tr>
    <tr>
      <td>A</td>
      <td>Red</td>
    </tr>
    <tr>
      <td>A</td>
      <td>Red</td>
    </tr>
    <tr>
      <td>A</td>
      <td>Red</td>
    </tr>
    <tr>
      <td>B</td>
      <td>Red</td>
    </tr>
  </tbody>
</table>""",
        "query": "For Red, does B appear most often?.",
        "code": """
The output code is :
```python
aimport pandas as pd

insight_type = "Rank"
table = table[table["Color"] == "Red"]
insight_result = insight_evaluate(table, insight_type, "Name", "count", "Name")
if insight_result[0]:
    result = insight_result[1]["First"] == "B"
else:
    result = False
    print(result)
```
"""
    },
]
