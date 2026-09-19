import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from scripts.release_files import release_files
from src.models.external_codec import load_external_codec
from src.utils.fvd import load_fvd_detector


class ExternalDependencyTests(unittest.TestCase):
    def test_external_codec_is_loaded_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "encoder.ts"
            model = nn.Linear(4, 3).eval()
            x = torch.randn(2, 4)
            torch.jit.trace(model, x).save(str(path))
            loaded = load_external_codec(path)
            torch.testing.assert_close(loaded(x), model(x))
            self.assertFalse(loaded.training)
            self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))
            with self.assertRaisesRegex(FileNotFoundError, "not bundled"):
                load_external_codec(Path(tmp) / "missing.ts")

    def test_detector_checksum_checked_before_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "detector.pt"
            torch.jit.trace(nn.Linear(3, 2).eval(), torch.ones(1, 3)).save(str(path))
            with patch("torch.jit.load") as loader:
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    load_fvd_detector(path, "cpu")
                loader.assert_not_called()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            model, actual = load_fvd_detector(path, "cpu", digest)
            self.assertEqual(actual, digest)
            self.assertFalse(model.training)
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))

    def test_release_has_no_gpe_implementation(self):
        root = Path(__file__).resolve().parents[1]
        files = release_files(root)
        self.assertFalse((root / "src/models/gpe.py").exists())
        for path in files:
            self.assertNotIn("external", path.relative_to(root).parts)
            if "src" in path.relative_to(root).parts:
                content = path.read_text(encoding="utf-8")
                self.assertNotIn("class GPEEncoder", content)
                self.assertNotIn("class GPEDecoder", content)
        with tempfile.TemporaryDirectory() as tmp:
            forbidden = Path(tmp) / "src/models/gpe.py"
            forbidden.parent.mkdir(parents=True)
            forbidden.touch()
            with self.assertRaisesRegex(ValueError, "must not be included"):
                release_files(tmp)


if __name__ == "__main__":
    unittest.main()
