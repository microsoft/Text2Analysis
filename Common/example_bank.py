# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json

from Common.safe_data import parse_list_cell
from Common.table_lib import *
from Common.taxonomy import *
from GCR.run import Role


class Example:
    JOIN_SYMBOL = "\r\n"
    TABLE_HEAD = 5

    def __init__(self):
        self.table = None
        self.query = None
        self.ana_taxonomy = None
        self.query_taxonomy = None
        self.table_name = None
        self.html = None

        # for taxonomy example
        self.reason = None

        # for unclear query example
        self.given_query_taxonomy = None
        self.given_parameter = None
        self.parameter_grounding_from_query = None
        self.new_query = None
        self.given_ana_taxonomy = None

        # for subset dataset
        self.python = None
        self.python_res = None
        self.source = None  # the annotation csv file path
        self.ori_query = None

    @classmethod
    def init_query_annotation(cls, content):
        example = cls()
        example.table = pd.read_html(content["html"])[0]
        example.query = content["query"]
        example.ana_taxonomy = [AnaTaxonomy.from_raw_str(tax[-1]) for tax in json.loads(content["operations"])[0]["taxonomy"]] if pd.notna(content["operations"]) else []
        example.query_taxonomy = [QueryTaxonomy.from_raw_str(tax[-1]) for tax in
                                  json.loads(content["ambiguities"])[0]["taxonomy"]] if pd.notna(content["ambiguities"]) else []
        example.table_name = content["table_name"]
        example.html = content["html"]
        return example

    @classmethod
    def init_unclear_parameter_query_example(cls, content):
        example = cls()
        example.query = content["query"]
        example.given_query_taxonomy = content["given_query_taxonomy"]
        example.given_parameter = content["given_parameter"]
        example.parameter_grounding_from_query = content["parameter_grounding_from_query"]
        example.given_ana_taxonomy = content["given_ana_taxonomy"]
        example.new_query = content["new_query"]
        annotation = label_studio.loc[
            (label_studio['table_name'] == content["tables"]) & (label_studio["query"] == content[
                "query"]), ["html", "operations", 'ambiguities']].values
        example.table = pd.read_html(annotation[0][0])[0]
        example.ana_taxonomy = [AnaTaxonomy(tax[-1]) for tax in json.loads(annotation[0][1])[0]["taxonomy"]]
        example.query_taxonomy = [QueryTaxonomy.from_raw_str(tax[-1]) for tax in
                                  json.loads(annotation[0][2])[0]["taxonomy"]]
        example.table_name = content["tables"]
        example.html = annotation[0][0]
        return example

    @classmethod
    def init_unclear_parameter_query_annotation(cls, content, without_annotation=False):
        example = cls()
        example.table = pd.read_html(content["html"])[0]
        example.query = content["query"]
        example.ana_taxonomy = [AnaTaxonomy.from_raw_str(tax) for tax in parse_list_cell(content["operations"], "operations", require_list=True)]
        example.query_taxonomy = [QueryTaxonomy.from_raw_str(tax[-1]) for tax in
                                  json.loads(content["ambiguities_new"])[0]["taxonomy"]] \
            if not (without_annotation or pd.isna(content["ambiguities_new"])) \
            else [QueryTaxonomy.from_raw_str(tax) for tax in parse_list_cell(content["ambiguities"], "ambiguities", require_list=True)]
        example.table_name = content["table_name"]
        example.html = content["html"]
        example.given_ana_taxonomy = AnaTaxonomy(content["given_anatype"])
        example.given_query_taxonomy = [QueryTaxonomy.from_raw_str(content["given_ambiguities"])]
        example.given_parameter = content["given_parameter"]
        example.parameter_grounding_from_query = None
        new_query = content["new_query_anno"] if not (without_annotation or pd.isna(content["new_query_anno"])) else \
            content["new_query"]
        example.new_query = new_query.replace("\n", ", ").strip("\'").split("\', \'") \
            if new_query.startswith("\'") and new_query.endswith("\'") \
            else new_query.split('\n')  # The first time annotation / The second time annotation
        example.correct = content["correct"] if not without_annotation else None
        return example

    @classmethod
    def init_usertesting(cls, content):
        example = cls()
        example.query = content
        example.table = tables_html[
            names.index(query_lib.loc[query_lib['Utterance'] == example.query, 'Study'].values[0])]
        return example

    @classmethod
    def init_usertesting_taxonomy_example(cls, content):
        example = cls()
        example.query = content["query"]
        if "tables" in content:
            example.table = tables_html[names.index(content["tables"])]
        else:
            example.table = tables_html[
                names.index(query_lib.loc[query_lib['Utterance'] == example.query, 'Study'].values[0])]
        example.ana_taxonomy = content["ana_taxonomy"]
        example.query_taxonomy = content["query_taxonomy"]
        example.reason = content["reason"]
        return example

    @classmethod
    def init_unclear_task_query_example(cls, content):
        example = cls()
        annotation = label_studio.loc[
            (label_studio['table_name'] == content["tables"]), ["html"]].values
        example.table = pd.read_html(annotation[0][0])[0]

        example.given_query_taxonomy = content["given_query_taxonomy"]
        example.new_query = content["new_query"]
        example.table_name = content["tables"]
        example.html = annotation[0][0]
        return example

    @classmethod
    def init_unclear_task_query_annotation(cls, content, without_annotation=False):
        example = cls()
        example.table = pd.read_html(content["html"])[0]
        example.table_name = content["table_name"]
        example.html = content["html"]
        example.given_query_taxonomy = [QueryTaxonomy(content["given_ambiguities"])]
        new_query = content["new_query_anno"] if not(without_annotation or pd.isna(content["new_query_anno"])) else content["new_query"]
        example.new_query = new_query.replace("\n", ", ").strip("\'").split("\', \'") \
            if new_query.startswith("\'") and new_query.endswith("\'") \
            else new_query.split('\n')  # The first time annotation / The second time annotation
        example.correct = content["correct"] if not without_annotation else None
        return example

    @classmethod
    def init_subset_query(cls, content):
        example = cls()
        example.table = pd.read_html(content["html"])[0]
        example.query = content["query"]
        example.ana_taxonomy = [AnaTaxonomy.from_raw_str(tax) for tax in parse_list_cell(content["operations"], "operations", require_list=True)] if pd.notna(
            content["operations"]) else []
        example.query_taxonomy = [QueryTaxonomy.from_raw_str(tax) for tax in parse_list_cell(content["ambiguities"], "ambiguities", require_list=True)] if pd.notna(
            content["ambiguities"]) else []
        example.table_name = content["table_name"]
        example.html = content["html"]
        return example

    def generate_table_query(self):
        return self.JOIN_SYMBOL.join([
            "Given the HTML tables:",
            self.table.head(self.TABLE_HEAD).to_html(index=False),
            "The user query is:",
            str(self.query),
        ])

    def generate_taxonomy(self, query_taxonomy=None):
        if query_taxonomy == None:
            query_taxonomy = self.query_taxonomy
        return f"Taxonomy: {str({'Analysis Operations taxonomy': [i.value for i in self.ana_taxonomy], 'Query Unspecified taxonomy': [i.value for i in query_taxonomy]})}"

    def generate_example4taxonomy(self):
        return self.JOIN_SYMBOL.join([
            "<example>",
            self.generate_prompt4taxonomy(),
            self.generate_taxonomy(),
            f"Reason: {self.reason}"
            "</example>"
        ])

    def generate_prompt4taxonomy(self):
        return self.JOIN_SYMBOL.join([
            self.generate_table_query(),
            "==>",
        ])

    def generate_prompt4unclear_parameter_query(self, query_taxonomy, ana_taxonomy, parameter):
        return self.JOIN_SYMBOL.join([
            self.generate_table_query(),
            f"This query's {self.generate_taxonomy()}",
            f"You need to change this query to the given 'Query Unspecified taxonomy': '{query_taxonomy[0].value}' (remove the '{parameter}' parameter of '{ana_taxonomy.value}' operation from the query).",
            "==>",
        ])

    def generate_example4unclear_parameter_query(self, need_role=False):
        if not need_role:
            return self.JOIN_SYMBOL.join([
                "<example>",
                self.generate_prompt4unclear_parameter_query(self.given_query_taxonomy, self.given_ana_taxonomy,
                                                             self.given_parameter),
                f"Parameter grounding from query: {self.parameter_grounding_from_query}. This is optional, because some examples do not have this information. But when you complement it, you must include this information.",
                f"New queries: {self.new_query}",
                "</example>"
            ])
        else:
            return {Role.USER: self.generate_prompt4unclear_parameter_query(self.given_query_taxonomy,
                                                                            self.given_ana_taxonomy,
                                                                            self.given_parameter),
                    Role.ASSISTANT: self.JOIN_SYMBOL.join([
                        f"Parameter grounding from query: {self.parameter_grounding_from_query}. You should not answer None!!! This can be None in examples, because some examples do not have this information.",
                        f"New queries: {self.new_query}"
                    ])}

    def generate_prompt4unclear_task_query(self):
        return self.JOIN_SYMBOL.join([
            "Given the HTML tables:",
            self.table.head(self.TABLE_HEAD).to_html(index=False),
            f"You need to generate user queries according to the given 'Query Unspecified taxonomy': '{QueryTaxonomy.Unspecified_tasks.value}'.",
            "==>",
        ])

    def generate_example4unclear_task_query(self, need_role=False):
        if not need_role:
            return self.JOIN_SYMBOL.join([
                "<example>",
                self.generate_prompt4unclear_task_query(),
                f"New queries: {self.new_query}",
                "</example>"
            ])
        else:
            return {Role.USER: self.generate_prompt4unclear_task_query(),
                    Role.ASSISTANT: self.JOIN_SYMBOL.join([
                        f"New queries: {self.new_query}"
                    ])}


examples4taxonomy = [
    {
        "tables": "Formatting",
        "query": "Make the tables more visually appealing.",
        "ana_taxonomy": [AnaTaxonomy.Formatting],
        "query_taxonomy": [QueryTaxonomy.Lacks_table_references, QueryTaxonomy.Lacks_other_API_arguments],
        "reason": "It's clear that the intent is to format the tables, but it's not clear which part of the tables and into what format."
    }
]

examples4unclear_parameter_query = [
    {  # Forecasting, forecast_field
        "tables": "CarSales.csv",
        "query": "Forecast the sales and cost data of different brands, categories and models of cars in 2012. ",
        "given_query_taxonomy": [QueryTaxonomy.Lacks_table_references],
        "given_ana_taxonomy": AnaTaxonomy.Forecasting,
        "given_parameter": "forecast_field",
        "parameter_grounding_from_query": "sales and cost data",
        "new_query": [
            "Forecast the data of different brands, categories and models of cars in 2012.",
            "What are the forecasts for different brands, categories and models of cars in 2012?",
            "What will be for different brands, categories and models of cars in 2012?",
            "Can you help me predict the data for various car brands, categories, and models in 2012?",
            "I need assistance in forecasting the 2012 data for different car brands, categories, and models.",
            "Could you please assist me in predicting the 2012 data for cars of various brands, categories, and models?",
        ],
    },
    {  # Forecasting, freq or steps or (freq and steps)
        "tables": "CarSales.csv",
        "query": "Forecast the sales and cost data of different brands, categories and models of cars in 2012. ",
        "given_query_taxonomy": [QueryTaxonomy.Lacks_other_API_arguments],
        "given_ana_taxonomy": AnaTaxonomy.Forecasting,
        "given_parameter": "freq or steps or (freq and steps)",
        "parameter_grounding_from_query": "2012",
        "new_query": [
            "Forecast the sales and cost data of different brands, categories and models of cars.",
            "What will be the sales and cost data of different brands, categories and models of cars.",
            "Can you do me a favor and predict the sales and cost data for various car brands, categories, and models?",
            "Can we make an estimation of the sales and cost data for various car brands, categories, and models?",
            "What are the anticipated sales and cost data for different car brands, categories, and models?",
        ]
    },
    {  # Clustering, groupby
        "tables": "stock.csv",
        "query": "Can you group the stocks according to the open and close price trend?",
        "given_query_taxonomy": [QueryTaxonomy.Lacks_table_references],
        "given_ana_taxonomy": AnaTaxonomy.Clustering,
        "given_parameter": "groupby",
        "parameter_grounding_from_query": "stocks",
        "new_query": [
            "Can you group according to the open and close price trend?",
            "Can you categorize based on the trends of open and close prices?"
            "Analyze and cluster by the open and close price trends, please.",
        ]
    },
    {  # Clustering, measures
        "tables": "stock.csv",
        "query": "Can you group the stocks according to the open and close price trend?",
        "given_query_taxonomy": [QueryTaxonomy.Lacks_table_references],
        "given_ana_taxonomy": AnaTaxonomy.Clustering,
        "given_parameter": "measures",
        "parameter_grounding_from_query": "open and close price",
        "new_query": [
            "Can you group the stocks based on their trends?",
            "Show me the groupings of stocks by their trends.",
            "I want to see the stocks clustered by their trends.",
        ]
    },
    {  # Aggregation, msr_field
        "tables": "Few Shot Examples.xlsx",
        "query": "Which brand has the highest total sales in 2023?",
        "given_query_taxonomy": [QueryTaxonomy.Lacks_table_references],
        "given_ana_taxonomy": AnaTaxonomy.Aggregation,
        "given_parameter": "msr_field",
        "parameter_grounding_from_query": "sales",
        "new_query": [
            "Which brand had the best overall in 2023?",
            "Looking back at 2023, which brand can be considered the best overall?",
            "Among the brands, which one performed the strongest overall in 2023?"
        ]
    },
    {  # Aggregation, agg_func
        "tables": "Few Shot Examples.xlsx",
        "query": "Which brand has the highest total sales in 2023?",
        "given_query_taxonomy": [QueryTaxonomy.Lacks_other_API_arguments],
        "given_ana_taxonomy": AnaTaxonomy.Aggregation,
        "given_parameter": "agg_func",
        "parameter_grounding_from_query": "total",
        "new_query": [
            "Which brand has the highest sales in 2023?",
            "Which brand achieved the highest sales in 2023?",
            "Can you tell me the brand with the highest sales in 2023?"
        ]
    },
    {  # Sort, sort_field
        "tables": "Few Shot Examples.xlsx",
        "query": "Which brand has the highest total sales in 2023?",
        "given_query_taxonomy": [QueryTaxonomy.Lacks_table_references],
        "given_ana_taxonomy": AnaTaxonomy.Sort,
        "given_parameter": "sort_field",
        "parameter_grounding_from_query": "sales",
        "new_query": [
            "Which brand had the best overall in 2023?",
            "Looking back at 2023, which brand can be considered the best overall?",
            "Among the brands, which one performed the strongest overall in 2023?"
        ]
    }
]
unclear_type=[i["given_ana_taxonomy"] for i in examples4unclear_parameter_query]

examples4unclear_task_query = [
    {
        "tables": "CarSales.csv",
        "given_query_taxonomy": [QueryTaxonomy.Unspecified_tasks],
        "new_query": [
            "Analyze this data sheet",
            "I think the AI chat will analyze all the data and will give the result.",
            "Help me analyze the 'Cost' field."
        ],
    }]
