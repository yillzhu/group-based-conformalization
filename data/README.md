# Enrollment data (§4.6)

| File | Rows | Role |
| --- | ---: | --- |
| `college_student_enroll-s1-1.csv` | 20,000 | Historical classifier training, test, calibration inputs |
| `college_applications6000.csv` | 6,000 | Applicant cohort for optimized merit offers |

Original filenames and bytes are retained. Columns used are SAT, GPA, merit,
enroll, and StudentID. Merit is in thousands of dollars. The history CSV includes
an unnamed index column; the applicant CSV has a UTF-8 BOM. The original sibling
`student-enroll` folder remains the historical record.

The corrected pipeline uses a deterministic 70/20/10 split, fits preprocessing on
training rows only, records split IDs, and uses the same calibration rows in global
and group-based calculations. Historical code did not consistently do so. Quantile
targets remain classifier predictions, not observed true probabilities (ISSUE-05).

The JANOS paper and public replication repository say these records were randomly
generated for the solver experiments; no data-generating function is included in
the repository. No redistribution permission beyond the source repository's license
was established by this local refactor. Generated quadratic inputs are described in
`conformal_cl.quadratic.problem` and do not need the enrollment datasets.
