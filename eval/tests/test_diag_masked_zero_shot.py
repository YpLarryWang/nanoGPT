from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class MaskedZeroShotIdentityTest(unittest.TestCase):
    def test_supplementary_random_draw_uses_seed_qualified_symlink(self):
        variant = "bl10m-d512L32-do0.1-gate-attnres8-offdev-test"
        seed = "20260719"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nano = root / "nano"
            eval_root = root / "eval"
            hf_root = root / "hf"
            run_dir = nano / "out-babylm" / variant
            tokenizer = nano / "data/babylm_officialdev/tokenizer/bpe-16000.json"
            modeling = nano / "eval/hf_nanogpt/modeling_nanogpt.py"
            for path in (run_dir, tokenizer.parent, modeling.parent, eval_root, hf_root):
                path.mkdir(parents=True, exist_ok=True)
            checkpoint = run_dir / "ckpt.pt"
            checkpoint.write_bytes(b"checkpoint")
            (run_dir / "checkpoint_manifest.json").write_text(json.dumps({
                "roles": {"final": checkpoint.name},
            }))
            tokenizer.write_text("{}")
            modeling.write_text("# model implementation\n")

            shared = hf_root / f"{variant}--maskrandom_count_matched"
            shared.mkdir()
            (shared / "checkpoint_source.json").write_text(json.dumps({
                "filename": checkpoint.name,
            }))
            (shared / "modeling_nanogpt.py").write_text(modeling.read_text())

            fake_python = root / "fake-python"
            fake_python.write_text("""#!/usr/bin/env python3
import os
from pathlib import Path
import sys
if sys.argv[1] == '-c' and 'runpy.run_module' not in sys.argv[2]:
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
args = sys.argv[3:]
model = Path(args[args.index('--model_path_or_name') + 1])
task = args[args.index('--task') + 1]
leaf = 'blimp_filtered' if task == 'blimp' else task
out = Path(os.environ['EVAL_REPO']) / 'results' / model.name / 'main/zero_shot/causal' / task / leaf
out.mkdir(parents=True, exist_ok=True)
(out / 'best_temperature_report.txt').write_text('### AVERAGE ACCURACY\\n50.0\\n')
(out / 'predictions.json').write_text('{}\\n')
""")
            fake_python.chmod(0o755)
            env = {
                **os.environ,
                "NANO_REPO": str(nano),
                "EVAL_REPO": str(eval_root),
                "EVAL_PY": str(fake_python),
                "HF_ROOT": str(hf_root),
            }
            subprocess.run(
                [
                    "bash", str(ROOT / "eval/diag_masked_zero_shot.sh"),
                    variant, "random_count_matched", seed, "blimp",
                ],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            result_name = f"{variant}--maskrandom_count_matched-seed{seed}"
            self.assertTrue((eval_root / "results" / result_name).is_dir())
            self.assertFalse((hf_root / result_name).exists())
            self.assertTrue(shared.is_dir())


if __name__ == "__main__":
    unittest.main()
