# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from enum import Enum


class AnaTaxonomy(Enum):
    '''
    AnaTaxonomy.Formatting.value = "Formatting"
    AnaTaxonomy('Formatting') = AnaTaxonomy.Formatting
    AnaTaxonomy.from_raw_str('3.2 Formatting') = AnaTaxonomy.Formatting
    '''
    # L1
    L1= "L1"
    Pivot = "Pivot/groupby"
    Aggregation = "Aggregation"
    Filter = "Filter"
    Sort = "Sort"
    Add_virtual_field = "Add virtual field"
    Calculation="Calculation"
    Add_field="Add field"
    # L2
    Clustering = "Clustering"
    Forecasting = "Forecasting"
    What_if = "What-if"
    Insights = "Basic Insights"
    # Visualization
    Visualization = "Visualization" # TODO:delete it
    Chart = "Chart"
    Formatting = "Formatting"

    @classmethod
    def from_raw_str(cls, string: str):
        string = string.split(".")[-1].strip()
        if string == "Group":
            return cls.Pivot
        return cls(string)

    # @classmethod
    # def from_raw_str(cls, string: str):
    #     string = string.split(".")[-1]
    #     if string == "Pivot/groupby":
    #         return cls.Pivot
    #     elif string == "Aggregation":
    #         return cls.Aggregation
    #     elif string == "Filter":
    #         return cls.Filter
    #     elif string == "Sort":
    #         return cls.Sort
    #     elif string == "Add virtual field":
    #         return cls.Add_virtual_field
    #     elif string == "Clustering":
    #         return cls.Clustering
    #     elif string == "Forecasting":
    #         return cls.Forecasting
    #     elif string == "What-if":
    #         return cls.What_if
    #     elif string == "Basic Insights":
    #         return cls.Insights
    #     elif string == "Chart":
    #         return cls.Chart
    #     elif string == "Formatting":
    #         return cls.Formatting
    #     else:
    #         raise ValueError("Unexpected analysis type: {}".format(string))


class QueryTaxonomy(Enum):
    Everything_is_clear = "Everything is clear"
    Unspecified_tasks = "Unspecified tasks"
    # Unspecified_parameters = "Unspecified parameters"
    Lacks_table_references = "Lacks tables references"
    Lacks_other_API_arguments = "Lacks other API arguments"

    # @classmethod
    # def from_raw_str(cls, string: str):
    #     string = string.split(".")[-1].strip()
    #     return cls(string)

    @classmethod
    def from_raw_str(cls, string: str):
        string = string.split(".")[-1]
        if string == "Everything is clear" or string == "Unspecified Algorithm":
            return cls.Everything_is_clear
        elif string == "Unspecified tasks":
            return cls.Unspecified_tasks
        elif string == "Lacks tables references" or string=="Lacks table references" or string=="ambiguous table references":
            return cls.Lacks_table_references
        elif string == "Lacks other API arguments":
            return cls.Lacks_other_API_arguments
        else:
            print(string)
            raise ValueError("Unexpected query type: {}".format(string))
