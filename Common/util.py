# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json


def load_json(path, encoding=None):
    with open(path, 'r', encoding=encoding) as f:
        info = json.load(f)
    return info

def dump_json(file,path, encoding=None):
    with open(path, 'w', encoding=encoding) as f:
        json.dump(file, f)
