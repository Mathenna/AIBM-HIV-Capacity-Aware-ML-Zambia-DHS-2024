#!/usr/bin/env python3
"""Compatibility layer for the public reproducibility release.

The recovered final-analysis archive contained a much larger historical core script.
For the public repository, shared functions required by the verified primary,
capacity, subgroup and domain analyses are provided by ``analysis_common.py``.

The complete historical retuning/SHAP driver is retained in the authors' restricted
analysis archive while it is being audited for release. It is not silently recreated
here. Running this file directly therefore fails explicitly rather than pretending to
perform the historical full-retuning workflow.
"""
from analysis_common import *  # noqa: F401,F403


if __name__ == "__main__":
    raise SystemExit(
        "The historical full-retuning/SHAP driver is not part of this audited public "
        "release. Run `python src/analysis_pipeline.py` for the verified publication "
        "reproduction. See docs/REPRODUCIBILITY_NOTES.md."
    )
