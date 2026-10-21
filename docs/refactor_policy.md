# Refactor policy and verification

The original sibling folder `student-enroll` is the record. This new folder is an
organization-only refactor. The migration deliberately does not fix the findings.

Allowed changes made:

- Split files by responsibility and move experiment entry points out of libraries.
- Rewire imports, remove manual sys.path edits, and use repository-root paths.
- Change the working result directory from `results/student_enroll` to
  `results/enrollment`; quadratic runs use `results/quadratic`.
- Put the quadratic sweep under a main guard so importing it does not launch runs.
- Add scientific-role docstrings and issue comments; preserve original inner
  function/class docstrings even where they contain inaccurate claims.
- Add packaging, navigation documents, reorganized data/results, and new aggregation code.
- Remove the machine-specific conda prefix and declare missing dependencies in a
  new portable environment file. The superseded environment file remains only in
  the original sibling `student-enroll` folder.

Not changed: model architecture, hyperparameters, random-seeding behavior, score
definitions, signs, group state/caching, fallback/stopping behavior, solver settings,
training splits, prediction targets, tuple/CSV schemas, and legacy defects.
No shared algorithm or calibration rewrite was introduced: the separate historical
TensorFlow and PyTorch implementations remain separate. No corrective code is
hidden behind warnings or wrappers. Import-time library dependency sets are smaller
because files were split; this is structural, not a numerical-method change.

Verification performed during construction:

1. Every moved function/class (32 definitions) was compared by Python AST to its
   source after only approved import/path reference substitutions. Comments do not
   affect this check. Original main bodies were preserved with path/import wiring.
2. Copied data/result hashes were compared to the original files.
3. All new Python modules and notebook cells were syntax checked.
4. Aggregation cells were executed in order with NumPy/pandas; Table 4.3 was checked
   against original per-applicant files, saved summary, and thesis values. Synthetic
   quadratic fixtures exercised the missing-data path, flag parsing, denominator
   rules, duplicate detection, and timing aggregation.
5. Original-folder file hashes were checked for unintended changes.

The ML/solver dependencies are not installed in the review runtime, so training,
optimization, and numerical equivalence of full solver runs have not been tested.
The notebook cells were checked in sequential Python execution; this does not claim
validation of every Jupyter frontend or a clean installation of the full environment.

`refactor_manifest.json` records symbol destinations and source/data/result hashes.
The original sibling `student-enroll` folder is the sole historical archive; no
duplicate archive is kept inside this repository.
All changes proposed under `findings.md` remain future work.
