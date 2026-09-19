# Pre-Release Checklist and Outstanding Work

This is a preparation checklist, not double-blind compliance certification.
Automated scans detect common file-level identifiers; they cannot establish
that accounts, commit history, data hosts or submission practices are anonymous.

## Required Follow-Up

| Item | Current status | Before release |
| --- | --- | --- |
| Venue policy | ICLR 2027 is the intended venue | Check current author guidance for anonymous pages, external links, supplementary material and applicable AI disclosures |
| Project-wide license | GPE codec definitions removed from the current tree; readers obtain them separately | Address historical copies; retain third-party licenses; obtain rights-holder confirmation for the overall project license |
| NSE experimental data | Zenodo 10939479, five PT files, approximately 26.2 GB | Verify schema, simulation metadata, SHA-256 and the first/third-file checksum duplication; see DATASETS.md |
| Original Acrobot benchmark | Original-style generator recipe and portable entrypoint available | Confirm paper seeds/settings; noisy RK45 is not a standard SDE discretization; default sampling interval is 8/239, not 1/30 |
| Reported FVD | Server uses TorchScript I3D; OneDrive reference appears in commented code | Detector hash and 10+30-frame protocol recorded; still align split, clip starts and paper results; see FVD.md |
| Experiment reproduction | Full-budget convergence and paper metrics not reproduced | Record final commands, configs, seeds, hardware, timing and metrics and compare with each reported result |
| Acrobot images/GPE | MLP, external GPE source/T-S weights and TorchScript switching connected | Pipeline runs through image input, latent PhiBE training, rollout, decoding and visualization; geometric GPE training recipe, public compatible weights and Acrobot FVD remain outstanding |

Review the [ICLR 2027 author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines)
before submission. Venue requirements and third-party license obligations are
separate. Do not assume that selecting GPL for the project grants permission to
redistribute GPE or overrides its separate terms. See LICENSE_STATUS.md and the
retained third-party license texts; obtain appropriate review where necessary.

Two choices in the earlier image preprocessing require particular attention:
`smooth.py` smooths an entire trajectory, and `2d24d.py` uses forward differences
to construct a current latent velocity. These operations may use frames after
the forecast origin. Using them as conditioning can leak future targets.
Neither copying this preprocessing without disclosure nor silently replacing it
with a causal variant establishes reproduction of the original reported numbers.

## Completed Preparation

- Configurations use portable project-relative paths instead of personal directories.
- Dependencies, explicit KTH downloads, hashes, AVI-to-HDF5 conversion and setup checks are provided.
- Acrobot demonstration generation is labeled as such; NSE shards can be converted to memory-mapped NPY.
- Unified entrypoints cover four datasets. Acrobot images support MLP and reader-owned GPE/TorchScript codecs. Evaluation defaults to last.pt rather than test-selected best.pt.
- A positive source allowlist excludes data, weights, logs, retired entrypoints, private notes and all Git history.
- ZIP timestamps are fixed and file hashes are recorded; regression tests and CPU CI are included.
- Third-party copyright notices are retained and must not be removed for anonymity.

## Anonymous Publication Procedure

1. Run `python scripts/audit_release.py`. Add repeated `--deny-token` options locally for names, affiliations or account handles; reports do not echo those tokens.
2. Run `python scripts/audit_release.py --require-ready`. Missing required assets, checksums or a project LICENSE must not be bypassed and then described as a complete release.
3. Run `python scripts/export_release.py`. Review its `.release/anonymous-code/` output rather than uploading the whole working directory.
4. If providing a direct anonymous GitHub link, create a fresh repository under an unlinked anonymous account. Do not fork, mirror or copy an old .git directory. Preserve third-party attribution.
5. For that fresh repository, use a non-identifying local Git name/email and avoid personally identifying signing keys.
6. Inspect commit authors, remotes, staged files and repository/account metadata, including avatars, biography, organizations, sponsorship and linked activity.
7. Review linked data-host accounts, shared-file ownership, PDF metadata, image EXIF, notebook output, TensorBoard files and embedded checkpoint paths. Do not publish unreviewed binaries.
8. Test the clean export and README commands, then inspect Actions. CI logs can disclose paths, hosts, accounts or configuration.
9. Follow venue policy for Issues, Discussions, Actions and review-period updates. Avoid personal-account interactions that identify an anonymous repository.

Steps 4-6 describe direct anonymous GitHub hosting. With an anonymization proxy,
the source repository may remain private under a personal account. Submit only
the anonymous URL and inspect both its logged-out view and downloadable files.
The proxy needs appropriate access and redistribution permission; it does not
resolve source licensing automatically.

The artifact remains a **draft**. A clean source scan cannot replace these manual checks.
