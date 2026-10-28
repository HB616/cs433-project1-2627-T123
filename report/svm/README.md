# SVM report package

Upload this entire folder to Overleaf, preserving the `assets/` subfolder.
Compile `svm-report.tex` with LuaLaTeX. The reusable `svm-section.tex` contains
the SVM material that can later be copied into or included from the team's
final two-page report.

The CSV and JSON files in `assets/` are inspectable source data for the three
figures. Regenerate all statistics and images from the raw course dataset with:

```bash
python3 generate_svm_report_assets.py \
  --data-dir ../../dataset \
  --max-rows 0 \
  --output-dir report/svm/assets
```

The standalone report deliberately contains more detail and figures than the
final group report can accommodate. During integration, retain the compact
results table, the main methodological paragraph, and at most one SVM figure.
