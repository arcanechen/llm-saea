import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
from llm_saea import Problem, Plan, benchmark, optimize
from llm_saea.llm import OpenAIDesigner, DeepSeekDesigner
from llm_saea.policy import export_program
from llm_saea.surrogate import Portfolio


class CoreTests(unittest.TestCase):
    def test_exact_budget_bounds_and_reproducibility(self):
        for method in ["random","de","saea","adaptive"]:
            calls = []
            def f(x):
                calls.append(x.copy())
                return float(x@x)
            p = Problem("counted",np.full(3,-2.),np.full(3,3.),f)
            a = optimize(p,budget=31,seed=17,method=method,n_initial=12)
            self.assertEqual(len(calls),31)
            self.assertTrue(np.all(a.x>=-2) and np.all(a.x<=3))
            self.assertTrue(np.all(np.diff(np.minimum.accumulate(a.y))<=0))
            b = optimize(p,budget=31,seed=17,method=method,n_initial=12)
            np.testing.assert_array_equal(a.y,b.y)
            np.testing.assert_array_equal(a.x,b.x)

    def test_common_initial_design_and_small_budget(self):
        p = benchmark("sphere",3)
        results = [optimize(p,budget=7,seed=2,method=m) for m in ["random","de","saea","adaptive"]]
        for r in results:
            self.assertEqual(len(r.y),7)
            np.testing.assert_array_equal(r.x,results[0].x)

    def test_replay_matches_all_dynamic_decisions(self):
        p = benchmark("rastrigin",3)
        a = optimize(p,budget=45,seed=2,method="adaptive",n_initial=12,design_interval=8)
        b = optimize(p,budget=45,seed=2,method="replay",n_initial=12,replay_events=a.events)
        np.testing.assert_array_equal(a.x,b.x)
        np.testing.assert_array_equal(a.y,b.y)

    def test_quadratic_generalization(self):
        rng = np.random.default_rng(4)
        x, test = rng.random((50,3)), rng.random((20,3))
        f = lambda z: np.sum((z-.3)**2,axis=1)
        model = Portfolio().fit(x,f(x),"auto")
        self.assertEqual(model.kind,"quadratic")
        self.assertLess(np.max(np.abs(model.model.predict(test)-f(test))),1e-4)

    def test_invalid_values(self):
        for kwargs in [{"exploration":float("nan")},{"population":True},{"surrogate":"evil"},{"mutation":9}]:
            with self.assertRaises(ValueError): Plan(**kwargs)
        with self.assertRaises(ValueError): Plan.from_dict({"mutation":.7})
        with self.assertRaises(ValueError): Problem("bad",[1],[0],lambda x:0)
        with self.assertRaises(ValueError):
            optimize(Problem("nan",[0,0],[1,1],lambda x:float("nan")),budget=4)
        with self.assertRaises(ValueError): optimize(benchmark("sphere",2),budget=0)

    def test_generated_code_uses_literal_data(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp)/"generated.py"
            export_program(file,Plan(reason="'\n__import__('os').system('not-executed')"),{"source":"unit_test"})
            namespace = {"__name__":"generated"}
            exec(compile(file.read_text(encoding="utf-8"),str(file),"exec"),namespace)
            self.assertEqual(namespace["PLAN"].population,40)
            self.assertEqual(len(namespace["solve"](benchmark("sphere",2),budget=8).y),8)
            with self.assertRaises(FileExistsError): export_program(file,Plan(),{})

    def test_save_contains_measured_history(self):
        with tempfile.TemporaryDirectory() as temp:
            r=optimize(benchmark("sphere",2),budget=10)
            r.save(temp)
            data=json.loads((Path(temp)/"result.json").read_text(encoding="utf-8"))
            self.assertEqual(data["evaluations"],10)
            self.assertFalse(data["metadata"]["online_llm"])
            self.assertEqual(len((Path(temp)/"history.csv").read_text().splitlines()),11)

    def test_failed_objective_preserves_journal(self):
        calls=[]
        def objective(x):
            calls.append(x)
            if len(calls)==3:
                raise RuntimeError("Simulation failure")
            return float(x@x)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"journal.jsonl"
            with self.assertRaises(RuntimeError):
                optimize(Problem("fails",[0,0],[1,1],objective),budget=10,journal_path=path)
            entries=[json.loads(s) for s in path.read_text().splitlines()]
            self.assertEqual([e["status"] for e in entries],["ok","ok","failed"])
            self.assertEqual(len(calls),3)


class LLMContractTests(unittest.TestCase):
    """Offline contract tests use a fake client; they do not verify live API access."""
    def designer(self,directory,text,status="completed"):
        response=SimpleNamespace(id="unit-test",model="fake-model",status=status,usage=None,output_text=text)
        self.requests=[]
        def create(**kwargs):
            self.requests.append(kwargs)
            return response
        return OpenAIDesigner("fake-model",directory,max_calls=1,
                              client=SimpleNamespace(responses=SimpleNamespace(create=create)))

    def test_valid_structured_plan_and_request_limit(self):
        with tempfile.TemporaryDirectory() as temp:
            designer=self.designer(temp,json.dumps(Plan().to_dict()))
            plan=designer.design({"dimension":3})
            self.assertEqual(plan,Plan())
            self.assertTrue(self.requests[0]["text"]["format"]["strict"])
            self.assertFalse(self.requests[0]["store"])
            with self.assertRaises(RuntimeError): designer.design({})
            self.assertEqual(len(self.requests),1)
            self.assertIn("validated_plan",json.loads((Path(temp)/"call_001.json").read_text()))

    def test_bad_response_does_not_become_offline_result(self):
        for text,status in [("not json","completed"),("","incomplete"),(json.dumps({"code":"arbitrary"}),"completed")]:
            with tempfile.TemporaryDirectory() as temp:
                d=self.designer(temp,text,status)
                with self.assertRaises(RuntimeError): d.design({})
                data=json.loads((Path(temp)/"call_001.json").read_text())
                self.assertIn("error_type",data)
                self.assertNotIn("validated_plan",data)


class DeepSeekContractTests(unittest.TestCase):
    def fake(self, directory, finish_reason="stop", text=None):
        self.requests=[]
        response=SimpleNamespace(id="test-deepseek",model="deepseek-flash",usage=None,
            choices=[SimpleNamespace(finish_reason=finish_reason,
                message=SimpleNamespace(content=json.dumps(Plan().to_dict()) if text is None else text))])
        def create(**kwargs):
            self.requests.append(kwargs)
            return response
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        return DeepSeekDesigner("deepseek-flash",directory,max_calls=1,client=client)

    def test_deepseek_json_contract_and_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            designer=self.fake(temp)
            result=optimize(benchmark("sphere",2),budget=25,method="llm",designer=designer,max_designs=1)
            self.assertEqual(len(self.requests),1)
            self.assertEqual(self.requests[0]["response_format"],{"type":"json_object"})
            self.assertEqual(self.requests[0]["extra_body"]["thinking"]["type"],"disabled")
            self.assertEqual(self.requests[0]["max_tokens"],768)
            self.assertEqual(result.events[0]["source"],"deepseek")
            self.assertEqual(result.metadata["llm_provider"],"deepseek")
            log=json.loads((Path(temp)/"call_001.json").read_text())
            self.assertEqual(log["source"],"deepseek")
            self.assertEqual(log["api_family"],"chat_completions")

    def test_deepseek_rejects_truncated_or_invalid_json(self):
        for finish,text in [("length","{}"),("stop",""),("stop",'{"code":"arbitrary"}')]:
            with tempfile.TemporaryDirectory() as temp:
                designer=self.fake(temp,finish,text)
                with self.assertRaises(RuntimeError):designer.design({})
                self.assertEqual(len(self.requests),1)

    def test_deepseek_requires_its_own_key(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temp:
            with patch.dict("os.environ",{"OPENAI_API_KEY":"test-openai-only","DEEPSEEK_API_KEY":""}):
                with self.assertRaisesRegex(RuntimeError,"DEEPSEEK_API_KEY is missing"):
                    DeepSeekDesigner("deepseek-flash",temp)


if __name__ == "__main__":
    unittest.main()
