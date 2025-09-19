# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pandas as pd
from Common.example_bank import Example
dataset1 = pd.read_csv("subset_0808_2.csv")
dataset2 = pd.read_csv("subset_0811_insights.csv")
dataset3 = pd.read_csv("subset_0809_chart.csv")
dataset4 = pd.read_csv("new_table_final_1220.csv")
print(len(dataset3))
print(len(dataset4))
dataset = pd.concat([dataset1, dataset2, dataset3.head(int(len(dataset3)/2)), dataset4])
print(dataset.columns)
# for i in range(0,len(dataset),100):
#     print(dataset.iloc[i]["python_res"])
print(len(dataset))

# 创建一个新的空DataFrame，用于存储符合条件的行
new_df = pd.DataFrame(columns=dataset.columns)

# 遍历 old_df 的每一行，应用条件函数，并将符合条件的行添加到 new_df
for index, row in dataset.iterrows():
    try:
        example = Example.init_subset_query(row)
        if pd.notna(example.query) and pd.notna(row["python"]) and pd.notna(row["python_res"]):
            new_df = pd.concat([new_df, pd.DataFrame(row).transpose()], ignore_index=True)
    except:
        pass
print(len(new_df))
new_df.to_csv("Tables/text2analysis_1220.csv", index=False)
