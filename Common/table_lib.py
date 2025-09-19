# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

import pandas as pd

# tables in usertestingJune24
path = r"OPAL"
names = ["Formatting", "Analysis", "Fundraising", "Publishing"]
tables_html = [pd.read_excel(os.path.join(path, f"{i}.xlsx")) for i in names]
query_lib = pd.read_excel(
    r"OPAL/query/usertestingJune24.xlsx",
    sheet_name='Utterances')

# tables in label studio
# json_path = r"C:/Users/v-xinyihe/Documents/Repo/azure/DIAL/QueryBank/Annotation/project-1-at-2023-07-06-06-08-7a017cea.json"
# label_studio = load_json(json_path)
label_studio_path = r"project-1-at-2023-07-26-18-25-d7225031.csv"
label_studio = pd.read_csv(label_studio_path)
