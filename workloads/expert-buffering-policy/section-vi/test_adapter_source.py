"""CPU syntax/placement checks on the privately patched source; no imports of it."""
import argparse
import ast
import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent
FROZEN = HERE.parents[1] / "finemoe/deps/FineMoE-EuroSys26"
SOURCE = None
QWEN = "finemoe/models/modeling_qwen/modeling_qwen2_moe.py"


def forward(path):
    tree = ast.parse(path.read_text())
    block = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SyncQwen2MoeSparseMoeBlock")
    return next(n for n in block.body if isinstance(n, ast.FunctionDef) and n.name == "forward")


class SourceChecks(unittest.TestCase):
    def test_python_syntax_without_imports(self):
        for path in (QWEN, "finemoe/runtime/model_offload.py", "op_builder/prefetch.py",
                     "finemoe/ops/op_builder/prefetch.py"):
            ast.parse((SOURCE / path).read_text(), filename=path)
        for path in ("prepare_adapter.py", "build_adapter.py"):
            ast.parse((HERE / path).read_text(), filename=path)

    def test_expert_compute_loop_is_unchanged(self):
        loops = []
        for tree in (forward(FROZEN / QWEN), forward(SOURCE / QWEN)):
            loops.append(next(n for n in tree.body if isinstance(n, ast.For) and
                              isinstance(n.target, ast.Name) and n.target.id == "expert_idx"))
        self.assertEqual(ast.dump(loops[0], include_attributes=False),
                         ast.dump(loops[1], include_attributes=False))

    def test_actual_counts_begin_loop_end_order(self):
        tree = forward(SOURCE / QWEN)
        calls = {n.func.attr: n.lineno for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        loop = next(n for n in tree.body if isinstance(n, ast.For) and
                    isinstance(n.target, ast.Name) and n.target.id == "expert_idx")
        self.assertLess(calls["begin_expert_buffering"], loop.lineno)
        self.assertGreater(calls["end_expert_buffering"], loop.end_lineno)
        text = (SOURCE / QWEN).read_text()
        self.assertIn('expert_mask.sum(dim=(1, 2)).to(device="cpu", dtype=torch.int64).tolist()', text)
        self.assertIn('if not getattr(self, "eb_section_vi", False):', text)

    def test_worker_uses_one_policy_and_completion_hook(self):
        text = (SOURCE / "core/prefetch/task_scheduler.cpp").read_text()
        self.assertIn("if (eb_state_ && node->is_sparse) return;", text)
        self.assertIn("eb_state_ ? RemoveExpertBufferingNode(task)", text)
        self.assertIn("if (eb_state_ && node->is_sparse) CopyExpertBufferingNode(task);", text)
        self.assertIn("CompleteDemand(node->mutex, node->cv, node->state", text)
        body = (HERE / "adapter_live.inc").read_text().split("void ArcherTaskPool::CopyExpertBufferingNode", 1)[1]
        self.assertLess(body.index("CanAdmit("), body.index("node->SetDevice("))
        self.assertLess(body.index("node->SetDevice("), body.index("Admitted("))

    def test_old_runtime_is_rejected_and_both_build_paths_include_adapter(self):
        text = (SOURCE / "finemoe/runtime/model_offload.py").read_text()
        self.assertIn('"expert_buffering_runtime_revision", None) != "section-vi-private-adapter-v1"', text)
        for filename in ("op_builder/prefetch.py", "finemoe/ops/op_builder/prefetch.py"):
            builder = (SOURCE / filename).read_text()
            self.assertIn("core/eb_section_vi/adapter_state.cpp", builder)
            self.assertIn("return ['-ldl']", builder)

    def test_package_aliases_stay_inside_private_copy(self):
        for name in ("core", "op_builder"):
            alias = SOURCE / "finemoe/ops" / name
            self.assertTrue(alias.is_symlink())
            self.assertEqual(alias.resolve(), SOURCE / name)
        loop = SOURCE / "core/core"
        self.assertFalse(loop.exists() or loop.is_symlink())

    def check_owned_cleanup(self, exit_leader):
        spec = importlib.util.spec_from_file_location("eb_build_wrapper_cpu_test", HERE / "build_adapter.py")
        wrapper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wrapper)  # Only stdlib; main/build/torch never run.
        child = "import time; time.sleep(30)"
        code = ("import subprocess, sys, time; "
                f"p = subprocess.Popen([sys.executable, '-c', {child!r}], "
                "stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
                "print(p.pid, flush=True); " + ("" if exit_leader else "time.sleep(30)"))
        process = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE,
                                   text=True, start_new_session=True)
        try:
            descendant = int(process.stdout.readline())
            if exit_leader:
                self.assertEqual(process.wait(timeout=2), 0)
            self.assertIn(descendant, wrapper.group_members(process.pid))
            wrapper.stop_owned(process)
            self.assertEqual(wrapper.group_members(process.pid), [])
        finally:
            wrapper.stop_owned(process)
            process.stdout.close()

    def test_cpu_build_group_cleanup_with_live_leader(self):
        self.check_owned_cleanup(False)

    def test_cpu_build_group_cleanup_after_leader_exit(self):
        self.check_owned_cleanup(True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    SOURCE = args.source.resolve()
    unittest.main(argv=[__file__] + remaining, verbosity=2)
