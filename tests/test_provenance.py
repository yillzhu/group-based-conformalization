import hashlib
import tempfile
import unittest
from pathlib import Path

from conformal_cl.provenance import _hash_paths


class ProvenanceTests(unittest.TestCase):
    def test_hashes_paths_outside_project_root(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "external-input.txt"
            path.write_bytes(b"external input")

            hashes = _hash_paths([path])

            self.assertEqual(
                hashes,
                {
                    str(path.resolve()): hashlib.sha256(b"external input").hexdigest()
                },
            )


if __name__ == "__main__":
    unittest.main()
