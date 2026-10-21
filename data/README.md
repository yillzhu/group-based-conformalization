# Enrollment data (§4.6)

| File | Rows | Role |
| --- | ---: | --- |
| `college_student_enroll-s1-1.csv` | 20,000 | Historical classifier training, test, calibration inputs |
| `college_applications6000.csv` | 6,000 | Applicant cohort for optimized merit offers |

Original filenames and bytes are retained. Columns used are SAT, GPA, merit,
enroll, and StudentID. Merit is in thousands of dollars. The history CSV includes
an unnamed index column; the applicant CSV has a UTF-8 BOM. Existing loaders are
unchanged. The original sibling `student-enroll` folder remains the historical record.

Training's random 70/20/10 split differs from RCC's last-2,000 selection (ISSUE-04).
The scaler is fit before the split. Quantile targets are classifier predictions,
not observed true probabilities (ISSUE-05). No source/redistribution permission was
established by this local refactor. Generated quadratic inputs are described in
`conformal_cl.quadratic.problem` and do not need the enrollment datasets.
