"""Config dataclass round-trip — pure logic, CPU-safe."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hephaestus.config import HephaestusConfig


def test_from_yaml_cve(tmp_path):
    cfg_text = """
task_name: cve-analysis
system_prompt: "classify severity"
model:
  name: "Qwen/Qwen2.5-3B-Instruct"
  dtype: "float16"
evaluation:
  quality_gate: 0.95
  task_type: "multiclass"
  classes: ["CRITICAL","HIGH","MEDIUM","LOW"]
"""
    p = tmp_path / "cve.yaml"
    p.write_text(cfg_text)
    cfg = HephaestusConfig.from_yaml(str(p))
    assert cfg.task_name == "cve-analysis"
    assert cfg.model.name == "Qwen/Qwen2.5-3B-Instruct"
    assert cfg.evaluation.task_type == "multiclass"
    assert cfg.evaluation.classes == ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    assert cfg.evaluation.quality_gate == 0.95


def test_to_dict_keys():
    cfg = HephaestusConfig()
    d = cfg.to_dict()
    for key in ("task_name", "model", "training", "evaluation", "export"):
        assert key in d
