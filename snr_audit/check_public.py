"""Verify sampled gains and analytical filter responses using the public source tree."""
from pathlib import Path
import json

from check import responses, test_gain


class SourceTree:
    def __init__(self, root):
        self.root = root

    def namelist(self):
        roots = ("front_end", "emg_release", "decoding_inference", "deap_decoding_and_weights")
        return sorted(path.relative_to(self.root).as_posix()
                      for folder in roots for path in (self.root / folder).rglob("*.py"))

    def read(self, name):
        return (self.root / name).read_bytes()


if __name__ == "__main__":
    tree = SourceTree(Path(__file__).resolve().parents[1])
    gain = test_gain(tree)
    print(json.dumps({"gain_cases": gain["sample_cases_tested"], "responses_db": responses(tree)},
                     indent=2))
