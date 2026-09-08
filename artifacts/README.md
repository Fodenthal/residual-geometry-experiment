# Artifacts

Large generated experiment artifacts are not stored in Git. Compact reports and
decision summaries from the supervised audit are tracked in `semantic_audit/`.

Expected policy for public release:

- Put large run outputs on a durable external host such as Zenodo, Hugging Face,
  S3, or institutional storage.
- Record the URL, checksum, config hash, and exact commit/tag used to produce
  each artifact bundle.
- Keep this directory for small manifests only.

Suggested manifest format:

```text
artifact_name:
  url:
  sha256:
  produced_by_commit:
  config:
  config_hash:
  notes:
```
