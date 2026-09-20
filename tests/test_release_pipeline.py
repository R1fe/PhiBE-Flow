import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import torch

from scripts.audit_release import audit, inspect_text
from scripts.export_release import export
from scripts.generate_acrobot import generate
from scripts.prepare_kth import convert as convert_kth, local_video
from scripts.prepare_nse import convert as convert_nse
from scripts.release_files import release_files
from src.datasets.acrobot_angles import load_acrobot_pickle
from src.datasets.kth import discover_kth_videos, KTHDataset
from src.datasets.nse import NSEForecastDataset, compute_nse_normalization, resolve_nse_files
from src.utils.download import download_file

ROOT = Path(__file__).resolve().parents[1]


class ReleasePipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_demo_npz_is_deterministic_and_has_exact_frame_spacing(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp)/"a.npz", Path(tmp)/"b.npz"
            generate(first, trajectories=5, frames=6)
            generate(second, trajectories=5, frames=6)
            a, b = load_acrobot_pickle(first), load_acrobot_pickle(second)
            self.assertEqual(len(a), 5)
            np.testing.assert_allclose(np.diff(a[0]["t"]), 1/30)
            for x, y in zip(a, b):
                np.testing.assert_array_equal(x["y"], y["y"])
            with self.assertRaises(FileExistsError):
                generate(first)

    def test_nse_conversion_and_lazy_normalization_match_pt(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp)/"raw", Path(tmp)/"converted"
            source.mkdir()
            data = torch.arange(2*8*8*8).reshape(2, 8, 8, 8).float()/500
            torch.save((data, None), source/"one.pt")
            convert_nse(source, output)
            pt = NSEForecastDataset([source/"one.pt"], 8, 8, 2.0, center=0.25, center_data=True)
            mapped = NSEForecastDataset([output/"one.npy"], 8, 8, 2.0, center=0.25, center_data=True)
            for index in range(len(pt)):
                for actual, expected in zip(mapped[index], pt[index]):
                    torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(mapped.get_trajectory(1), data[1]/2 - 0.25)
            self.assertAlmostEqual(compute_nse_normalization([output/"one.npy"]),
                                   float(data.square().mean((-2, -1)).sqrt().mean()), places=6)
            self.assertEqual(mapped.trajectories[0].data_ptr(), mapped.trajectories[0].contiguous().data_ptr())
            torch.save(data, output/"one.pt")
            with self.assertRaisesRegex(ValueError, "both"):
                resolve_nse_files(output)
            # Windows cannot remove an NPY while the dataset retains its mapping.
            del pt, mapped

    def test_avi_conversion_has_valid_shapes_and_is_non_destructive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/"avi"
            source.mkdir()
            for index in range(2):
                writer = cv2.VideoWriter(str(source/f"clip{index}.avi"), cv2.VideoWriter_fourcc(*"MJPG"), 25, (16, 16))
                self.assertTrue(writer.isOpened())
                for step in range(6):
                    writer.write(np.full((16, 16, 3), step*20+index, dtype=np.uint8))
                writer.release()
            output = root/"videos.hdf5"
            manifest = convert_kth(source, output, image_size=8, min_frames=4)
            self.assertEqual(len(manifest["videos"]), 2)
            dataset = KTHDataset(discover_kth_videos(output), frames_per_sample=4, image_size=8)
            self.assertEqual(dataset[0][0].shape, (4, 3, 8, 8))
            before = output.read_bytes()
            with self.assertRaises(FileExistsError):
                convert_kth(source, output)
            self.assertEqual(output.read_bytes(), before)
            partial = root/"other.hdf5.part"
            partial.write_bytes(b"keep existing partial")
            with self.assertRaises(FileExistsError):
                convert_kth(source, root/"other.hdf5", min_frames=4)
            self.assertEqual(partial.read_bytes(), b"keep existing partial")

    def test_zip_paths_are_never_extracted_verbatim(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"video.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("../../outside.avi", b"not a real video")
            with local_video(path, "../../outside.avi") as extracted:
                self.assertEqual(extracted.name, "video.avi")
                self.assertEqual(extracted.read_bytes(), b"not a real video")
            self.assertFalse((Path(tmp)/"outside.avi").exists())

    def test_download_verifies_hash_and_does_not_replace_existing_file(self):
        class Response(io.BytesIO):
            def geturl(self):
                return "https://example.invalid/asset"

        content = b"fixture content"
        digest = hashlib.sha256(content).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"asset.bin"
            with patch("src.utils.download.urlopen", return_value=Response(content)):
                self.assertEqual(download_file("https://example.invalid/asset", output, digest), digest)
            with self.assertRaisesRegex(ValueError, "existing"):
                download_file("https://example.invalid/asset", output, "0"*64)
            self.assertEqual(output.read_bytes(), content)
            with patch("src.utils.download.urlopen", return_value=Response(content)):
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    download_file("https://example.invalid/asset", Path(tmp)/"bad.bin", "0"*64)
            self.assertFalse((Path(tmp)/"bad.bin").exists())
            self.assertFalse((Path(tmp)/"bad.bin.part").exists())
            with self.assertRaises(ValueError):
                download_file("http://example.invalid/asset", output)

    def test_identity_scanner_reports_location_without_echoing_secret(self):
        private_path = "Q:" + "/Users/" + "private-example/data"
        token = "synthetic-identity-token"
        findings = inspect_text(private_path+"\n"+token, [token])
        self.assertEqual({item["line"] for item in findings}, {1, 2})
        self.assertNotIn(token, json.dumps(findings))
        self.assertNotIn(private_path, json.dumps(findings))

    def test_release_export_excludes_assets_history_and_uses_fixed_timestamps(self):
        report = audit(ROOT)
        self.assertEqual(report["findings"], [])
        self.assertIn("nse_dataset", report["assets_without_direct_download"])
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp)/"anonymous-code"
            manifest = export(ROOT, destination)
            self.assertEqual(manifest["schema_version"], 1)
            self.assertNotIn("status", manifest)
            names = set(manifest["files"])
            self.assertNotIn("scripts/demo.py", names)
            self.assertNotIn("IMPLEMENTATION_STATUS_zh-CN.md", names)
            self.assertTrue(all(not name.startswith("experiments/") for name in names))
            self.assertTrue(all(not name.startswith("data/") or name == "data/README.md" for name in names))
            self.assertFalse((destination/".git").exists())
            with zipfile.ZipFile(destination.with_suffix(".zip")) as archive:
                for info in archive.infolist():
                    self.assertEqual(info.date_time, (2000, 1, 1, 0, 0, 0))
            for name, digest in manifest["files"].items():
                self.assertEqual(hashlib.sha256((destination/name).read_bytes()).hexdigest(), digest)
            self.assertEqual(audit(destination)["findings"], [])
            with self.assertRaises(FileExistsError):
                export(ROOT, destination)


if __name__ == "__main__":
    unittest.main()
