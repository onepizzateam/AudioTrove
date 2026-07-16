import hashlib


def make_doc_id(path: str) -> str:
    return hashlib.sha256(path.encode()).hexdigest()[:16]
