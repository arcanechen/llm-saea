"""Create and verify a credential-free three-part submission archive."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from zipfile import ZipFile,ZIP_DEFLATED
from pypdf import PdfReader

EXCLUDED={"__pycache__",".venv",".git","build",".pytest_cache"}


def eligible(path,root):
    relative=path.relative_to(root)
    return (path.is_file() and not path.is_symlink() and
            not any(part in EXCLUDED or part.endswith(".egg-info") for part in relative.parts) and
            not path.name.startswith(".env") and path.suffix not in {".pyc",".pyo"} and
            path.name!="SHA256SUMS.txt")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    root=Path(args.root).resolve();output=Path(args.output).resolve()
    if output.is_relative_to(root):
        parser.error("Archive must be outside the submission tree")
    if output.exists():
        parser.error("Archive already exists; select a new filename")
    files=sorted(p for p in root.rglob("*") if eligible(p,root))
    for p in files:
        if p.suffix in {".py",".md",".txt",".json",".jsonl",".csv",".toml"}:
            data=p.read_text(encoding="utf-8-sig")
            if re.search(r"sk-[A-Za-z0-9_-]{24,}",data):
                raise RuntimeError(f"Possible credential in {p.name}; archive not created")
        if p.suffix==".py":
            compile(p.read_text(encoding="utf-8"),str(p),"exec")
    wheels=list((root/"01_python"/"dist").glob("*.whl"))
    assert len(wheels)==1
    with ZipFile(wheels[0]) as wheel:
        for source in (root/"01_python"/"src"/"llm_saea").glob("*.py"):
            assert wheel.read("llm_saea/"+source.name)==source.read_bytes(),source.name
    pages={str(p.relative_to(root)):len(PdfReader(p).pages) for p in root.rglob("*.pdf")}
    assert len(pages)==3
    for p in root.rglob("*.pdf"):
        assert all(len(page.extract_text().strip())>50 for page in PdfReader(p).pages)
    analysis=json.loads((root/"03_experiments"/"offline"/"analysis.json").read_text())
    assert analysis["verified_runs"]==240 and analysis["true_evaluations"]==24000
    online=root/"03_experiments"/"online"
    online_analysis=json.loads((online/"analysis.json").read_text(encoding="utf-8"))
    online_audit=json.loads((online/"online_audit.json").read_text(encoding="utf-8"))
    online_manifest=json.loads((online/"manifest.json").read_text(encoding="utf-8"))
    assert online_manifest["state"]=="COMPLETED"
    assert online_analysis["verified_runs"]==300 and online_analysis["true_evaluations"]==30000
    assert online_audit["successful_requests"]==240 and online_audit["failed_requests"]==0
    assert online_audit["replay_verified_runs"]==online_audit["frozen_ablation_runs"]==60
    online_logs=[json.loads(p.read_text(encoding="utf-8")) for p in (online/"runs").rglob("call_*.json")]
    assert len(online_logs)==240 and all(p["status"]=="completed" and "validated_plan" in p and "error_type" not in p for p in online_logs)
    assert sum(p["usage"]["total_tokens"] for p in online_logs)==online_audit["total_tokens"]
    for rel,sha in online_manifest["baseline_file_sha256"].items():
        assert hashlib.sha256((online/rel).read_bytes()).hexdigest()==sha
    for name,sha in online_manifest["source_sha256"].items():
        assert hashlib.sha256((root/"01_python"/"src"/"llm_saea"/name).read_bytes()).hexdigest()==sha
    visual=json.loads((root/"03_experiments"/"pdf_visual_review.json").read_text(encoding="utf-8"))
    assert visual["status"]=="PASSED"
    for p in root.rglob("*.pdf"):
        assert visual["sha256"][p.relative_to(root).as_posix()]==hashlib.sha256(p.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="llm_saea_delivery_") as tmp:
        copied=Path(tmp)/"submission"
        for p in files:
            dest=copied/p.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,dest)
        env=os.environ.copy()
        env.update(PYTHONPATH=str(copied/"01_python"/"src"),PYTHONDONTWRITEBYTECODE="1",
                   OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1")
        result=subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-v"],
                              cwd=copied/"01_python",env=env,text=True,capture_output=True)
        (root/"03_experiments"/"archive_smoke_test.txt").write_text(result.stdout+result.stderr,encoding="utf-8")
        if result.returncode:
            raise RuntimeError("Copied package failed tests; see archive_smoke_test.txt")
    tested = re.search(r"Ran (\d+) tests?", result.stdout+result.stderr)
    assert tested is not None
    live_file = root/"03_experiments"/"deepseek_smoke"/"smoke_summary.json"
    live = json.loads(live_file.read_text(encoding="utf-8")) if live_file.exists() else {}
    live_verified = live.get("status") == "PASSED"
    if live_verified:
        calls = list((live_file.parent/"generated_algorithm.llm_logs").glob("call_*.json"))
        assert len(calls) == live["actual_api_requests"] == 1
        call = json.loads(calls[0].read_text(encoding="utf-8"))
        assert call["source"] == "deepseek" and call["status"] == "completed"
        assert call["usage"] == live["usage"] and "validated_plan" in call
    validation={"checked_utc":datetime.now(timezone.utc).isoformat(),"copied_source_tests":f"{tested.group(1)} passed",
                "wheel_matches_current_source":True,"all_python_sources_compile":True,
                "pdf_pages":pages,"pdf_rendering":"Poppler rendering reviewed visually; no clipping or overlap found",
                "raw_runs_verified":300,"optimization_evaluations_verified":30000,
                "baseline_runs_reused":240,"new_online_runs":60,"frozen_ablation_runs":60,
                "replay_verified_runs":60,"online_experiment_api_requests":240,
                "online_experiment_tokens":online_audit["total_tokens"],
                "online_openai_verified":False,"online_deepseek_verified":live_verified,
                "online_status":"DeepSeek full benchmark COMPLETED; 240 calls; all successful; OpenAI not live-tested",
                "included_credentials":False,"excluded":["private env files","virtual environments","build caches"]}
    (root/"03_experiments"/"validation.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding="utf-8")
    files=sorted(p for p in root.rglob("*") if eligible(p,root))
    sums={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    checksum=root/"SHA256SUMS.txt"
    checksum.write_text("\n".join(f"{sha}  {name}" for name,sha in sums.items())+"\n",encoding="utf-8")
    top="LLM_SAEA_提交材料"
    with ZipFile(output,"w",ZIP_DEFLATED,compresslevel=9) as archive:
        for p in [*files,checksum]:archive.write(p,top+"/"+p.relative_to(root).as_posix())
    with ZipFile(output) as archive:
        assert archive.testzip() is None
        for name,sha in sums.items():
            assert hashlib.sha256(archive.read(top+"/"+name)).hexdigest()==sha,name
    receipt={"archive":str(output),"bytes":output.stat().st_size,"files":len(files)+1,
             "sha256":hashlib.sha256(output.read_bytes()).hexdigest(),"archive_hashes_verified":True}
    output.with_suffix(".zip.sha256.txt").write_text(receipt["sha256"]+"  "+output.name+"\n",encoding="utf-8")
    output.with_suffix(".packaging.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
