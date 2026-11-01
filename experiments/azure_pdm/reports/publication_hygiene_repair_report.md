# Publication hygiene repair report

## Result

The branch is cleaner for selective GitHub publication, but staging is still intentionally pending explicit authorization.

## Repairs

- `.gitignore` now covers Python caches, pytest cache, virtual environments, `.env` files, generated numeric/model artifacts (`.npy`, `.npz`, `.joblib`, `.pkl`, `.parquet`, `.pt`, `.pth`, `.ckpt`), and root-level raw/derived/phase-output directories.
- Required source code, tests, fixture CSVs, contracts, reports, and documentation remain trackable.
- Personal absolute paths were replaced in the Azure contracts, Day 1/Day 2 reports, Phase 4–8 documentation, and quickstart with `${IOT_FUSION_REPO_ROOT}`, `${AZURE_PDM_ROOT}`, or `${FD001_RECOVERY_ROOT}` placeholders.
- Existing tracked artifacts remain tracked; ignore rules do not untrack them.

## Verification

- Relevant tests: `187 passed, 1 warning`.
- JSON validation: 15 repository JSON files parsed successfully.
- `git diff --check`: clean.
- Secret scan: no real credentials, keys, tokens, or `.env` files found.
- Fixture CSV and required source examples remain trackable under `git check-ignore`.
- Protected Phase 5 and Phase 7 hashes are unchanged.

## Remaining risks

- Branch has no upstream configured; this was not changed.
- Existing tracked large artifacts remain in the repository: 34.50 MiB, 27.64 MiB, 6.86 MiB, 4.33 MiB, 3.37 MiB, and 2.14 MiB files. They were not deleted, migrated, or rewritten.
- The local FD001 frozen checkpoint is still absent even though an exact read-only copy exists in the sibling original worktree.
- The full FD001 inference test remains blocked by that missing local checkpoint.

## Publication recommendation

**Conditional GO after review of the proposed selective staging list.** Do not use `git add .`; stage only the reviewed source, test, contract, report, documentation, and fixture files. Do not stage raw/generated data, caches, model binaries, or external outputs.
