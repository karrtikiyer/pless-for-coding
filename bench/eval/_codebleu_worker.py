"""Child-process entry point for deterministic self-CodeBLEU.

Started by ``bench.eval.metrics.pairwise_codebleu_fixed_seed`` with a fixed ``PYTHONHASHSEED``.
Reads a JSON list of code groups on stdin, writes the per-group diversity values as JSON to the path
given in argv[1] (a file, not stdout, so library prints cannot corrupt the result).
"""
import json
import sys

from bench.eval.metrics import pairwise_codebleu_in_process


def main() -> None:
    groups = json.loads(sys.stdin.read())
    result = pairwise_codebleu_in_process(groups)
    with open(sys.argv[1], "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
