import ast
import csv
import importlib.util
import json
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Common.safe_data import DataFormatError, MAX_CELL_BYTES, parse_example_fields, parse_list_cell, parse_literal


class SafeDataTests(unittest.TestCase):
    def test_json_and_legacy_literals(self):
        self.assertEqual(parse_literal('["hello", null, true, -2]'), ["hello", None, True, -2])
        self.assertEqual(parse_literal("['hello', None, True, -2]"), ["hello", None, True, -2])
        self.assertEqual(parse_literal("{'value': (1, 2)}"), {"value": (1, 2)})

    def test_report_payload_never_executes(self):
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / "executed.txt"
            payload = "[__import__('pathlib').Path(" + repr(str(sentinel)) + ").write_text('executed')]"
            with self.assertRaises(DataFormatError):
                parse_list_cell(payload, "code_result")
            self.assertFalse(sentinel.exists())

    def test_each_reported_column_never_executes(self):
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / "executed.txt"
            payload = "[__import__('pathlib').Path(" + repr(str(sentinel)) + ").write_text('executed')]"
            for field in ("python", "python_res", "given_parameter", "code_result"):
                row = {"python": "['result = 1']", "python_res": "['1']", "given_parameter": "field", "code_result": "['1']"}
                row[field] = payload
                with self.subTest(field=field), self.assertRaises(DataFormatError):
                    parse_example_fields(row)
                self.assertFalse(sentinel.exists())

    def test_taxonomy_columns_never_execute(self):
        for field in ("operations", "ambiguities"):
            with self.subTest(field=field), self.assertRaises(DataFormatError):
                parse_list_cell("[__import__('os').getcwd()]", field, require_list=True)

    def test_no_executable_eval_calls_remain(self):
        root = Path(__file__).resolve().parents[1]
        for source in root.rglob("*.py"):
            if "tests" in source.parts:
                continue
            tree = ast.parse(source.read_text(encoding="utf-8-sig"))
            calls = [node.lineno for node in ast.walk(tree)
                     if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                     and node.func.id == "eval"]
            self.assertEqual(calls, [], str(source))

    def test_executable_expressions_are_rejected(self):
        for payload in ("[print(1)]", "[item for item in [1]]", "[object.__class__]", "[lambda: 1]", "[1] + [2]", "[0] * 1000000"):
            with self.subTest(payload=payload), self.assertRaises(DataFormatError):
                parse_literal(payload)

    def test_size_depth_and_type_limits(self):
        for payload in (" " * (MAX_CELL_BYTES + 1), "[" * 70 + "0" + "]" * 70, '[NaN]', "{1, 2}", "b'bytes'", "[" + ",".join("0" for index in range(20001)) + "]"):
            with self.subTest(payload=payload[:40]), self.assertRaises(DataFormatError):
                parse_literal(payload)

    def test_scalar_and_multiple_alternatives(self):
        scalar = {"python": "result = [1, 2]", "python_res": "[1, 2]", "given_parameter": float("nan")}
        result = parse_example_fields(scalar)
        self.assertEqual(result["python_res"], ["[1, 2]"])
        self.assertEqual(result["given_parameter"], [None])
        multiple = {"python": "['result = 1', 'result = 2']", "python_res": "['1', '2']", "given_parameter": '["field", null]', "code_result": '["1", "2"]'}
        result = parse_example_fields(multiple)
        self.assertEqual(result["python"], ["result = 1", "result = 2"])
        self.assertEqual(result["code_result"], ["1", "2"])

    def test_invalid_shapes_fail_closed(self):
        for updates in ({"python": "[1]"}, {"python": "[]"}, {"python_res": "['1', '2']"}, {"code_result": "[]"}, {"given_parameter": "[{}]"}):
            row = {"python": "['result = 1']", "python_res": "['1']", "given_parameter": "field"}
            row.update(updates)
            with self.subTest(updates=updates), self.assertRaises(DataFormatError):
                parse_example_fields(row)


@unittest.skipUnless(importlib.util.find_spec("pandas"), "Loader tests require pandas; parser tests use only the standard library.")
class LoaderTests(unittest.TestCase):
    def setUp(self):
        import pandas as pd
        from Common.taxonomy import AnaTaxonomy
        self.pd = pd
        self.root = Path(__file__).resolve().parents[1]
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        table_module = types.ModuleType("Common.table_lib")
        table_module.pd = pd
        gcr = types.ModuleType("GCR.run")
        gcr.Role = types.SimpleNamespace(USER="user", ASSISTANT="assistant")
        checker = types.ModuleType("Evaluation.checker")
        checker.Checker = object
        checker.get_L2_anatype = lambda taxonomy: AnaTaxonomy.L1
        checker.FORECASTING_GT = pd.DataFrame({"query": []})
        progress = types.ModuleType("tqdm")
        progress.tqdm = lambda iterable, **kwargs: iterable
        self.modules = patch.dict(sys.modules, {"Common.table_lib": table_module, "GCR.run": gcr, "Evaluation.checker": checker, "tqdm": progress})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.html = patch.object(pd, "read_html", return_value=[pd.DataFrame({"value": [1]})])
        self.html.start()
        self.addCleanup(self.html.stop)
        self.dataset_class = runpy.run_path(str(self.root / "text2analysis.py"), run_name="loader_test")["Text2AnalysisDataset"]
        import Common.example_bank
        self.example_class = Common.example_bank.Example

    def row(self):
        return {"html": "<table></table>", "query": "query", "table_name": "table.csv", "operations": "[]", "ambiguities": "[]", "python": "['result = 1']", "python_res": "['1']", "given_parameter": "field", "code_result": "['1']", "ori_query": "query", "source": "fixture"}

    def write_csv(self, row):
        path = self.directory / "fixture.csv"
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        return path

    def test_dataset_loads_valid_csv(self):
        source = self.write_csv(self.row())
        read_csv = self.pd.read_csv
        def read(path, *args, **kwargs):
            return read_csv(path, *args, **kwargs) if Path(path) == source else self.pd.DataFrame({"value": [1]})
        with patch.object(self.pd, "read_csv", side_effect=read):
            dataset = self.dataset_class(source, is_eval=True)
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0].python, ["result = 1"])
        self.assertEqual(dataset[0].code_result, ["1"])

    def test_dataset_rejects_all_malicious_csv_fields(self):
        sentinel = self.directory / "executed.txt"
        payload = "[__import__('pathlib').Path(" + repr(str(sentinel)) + ").write_text('executed')]"
        for field in ("python", "python_res", "given_parameter", "code_result", "operations", "ambiguities"):
            row = self.row()
            row[field] = payload
            source = self.write_csv(row)
            read_csv = self.pd.read_csv
            def read(path, *args, **kwargs):
                return read_csv(path, *args, **kwargs) if Path(path) == source else self.pd.DataFrame({"value": [1]})
            with self.subTest(field=field), patch.object(self.pd, "read_csv", side_effect=read), self.assertRaises(DataFormatError):
                self.dataset_class(source, is_eval=True)
            self.assertFalse(sentinel.exists())

    def test_evaluation_script_rejects_csv_before_scoring(self):
        for field in ("python", "python_res", "given_parameter", "operations", "ambiguities"):
            row = self.row()
            row[field] = "[__import__('os').getcwd()]"
            with self.subTest(field=field), patch.object(self.pd, "read_csv", return_value=self.pd.DataFrame([row])), self.assertRaises(DataFormatError):
                runpy.run_path(str(self.root / "Evaluation" / "evaluate.py"), run_name="__main__")

    def test_evaluation_script_rejects_prediction_result(self):
        ground_truth = self.pd.DataFrame([self.row()])
        prediction = self.pd.DataFrame([{"table_name": "table.csv", "query": "query", "code_result": "[__import__('os').getcwd()]"}])
        with patch.object(self.pd, "read_csv", side_effect=[ground_truth, prediction, self.pd.DataFrame({"value": [1]})]), self.assertRaises(DataFormatError):
            runpy.run_path(str(self.root / "Evaluation" / "evaluate.py"), run_name="__main__")

    def test_checker_result_fallback_never_executes(self):
        evaluation = types.ModuleType("greykite.common.evaluation")
        evaluation.calc_pred_err = lambda *args, **kwargs: None
        with patch.dict(sys.modules, {"greykite.common.evaluation": evaluation}), patch.object(self.pd, "read_csv", return_value=self.pd.DataFrame({"query": []})):
            checker_type = runpy.run_path(str(self.root / "Evaluation" / "checker.py"), run_name="checker_test")["Checker"]
        checker = checker_type.__new__(checker_type)
        sentinel = self.directory / "executed.txt"
        payload = "__import__('pathlib').Path(" + repr(str(sentinel)) + ").write_text('executed')"
        self.assertEqual(checker.load_string_result(payload), payload)
        self.assertFalse(sentinel.exists())
        self.assertEqual(checker.load_string_result("True"), True)
        self.assertEqual(checker.load_string_result("Increasing"), "Increasing")

    def test_checker_rejects_invalid_structured_results(self):
        evaluation = types.ModuleType("greykite.common.evaluation")
        evaluation.calc_pred_err = lambda *args, **kwargs: None
        with patch.dict(sys.modules, {"greykite.common.evaluation": evaluation}), patch.object(self.pd, "read_csv", return_value=self.pd.DataFrame({"query": []})):
            checker_type = runpy.run_path(str(self.root / "Evaluation" / "checker.py"), run_name="checker_test")["Checker"]
        checker = checker_type.__new__(checker_type)
        for payload in ("[__import__('os').getcwd()]", "{'value': print(1)}", "(print(1),)", "[" * 70 + "0" + "]" * 70):
            with self.subTest(payload=payload), self.assertRaises(DataFormatError):
                checker.load_string_result(payload)


if __name__ == "__main__":
    unittest.main()