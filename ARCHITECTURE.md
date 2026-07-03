# AudioTrove Architecture

Rules:

- AudioDocument is the canonical object.
- Filters return bool.
- Transformers return AudioDocument.
- Executors manage execution.
- Blocks stay stateless.
