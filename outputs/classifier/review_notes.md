# Baseline review notes

This review does not change model settings or select a model using test results.

## Observed features

- Strong GRANTED features include `paid`, `settled`, `resolved`, and `provided documentation`.
- Strong DENIED features include `unpaid`, `unresolved`, `delinquent`, and `provided no`.
- `credibly` and `credibly testified` can encode the adjudicator's retrospective interpretation. Exhibit abbreviations `ae` and `ax` may reflect document conventions rather than applicant behavior.
- Direct verdict phrases are not among the reported top 20 features for either class. That is a limited feature inspection, not proof of leakage absence.

## Error counts

| Partition | Stored DENIED predicted GRANTED | Stored GRANTED predicted DENIED |
|---|---:|---:|
| validation | 444 | 251 |
| test | 548 | 285 |

Validation predictions are saved locally with row IDs for human error review. They can be joined back to the original data. No manual ground-truth adjudication, fairness audit, calibration, or prospective validation has been completed. Test errors are reported descriptively; they were not used to change preprocessing or model parameters.
