import sys
import os
sys.path.insert(0, os.getcwd())

from pathlib import Path
from app import KnowledgeBaseService

service = KnowledgeBaseService()

kb_id = "73493908"
index_path = service._get_kb_index_path(kb_id)
upload_dir = service._get_kb_upload_dir(kb_id)

print(f'kb_id: {kb_id}', file=sys.stderr)
print(f'index_path: {index_path}', file=sys.stderr)
print(f'upload_dir: {upload_dir}', file=sys.stderr)
print(f'index exists: {Path(index_path).exists()}', file=sys.stderr)
print(f'upload dir exists: {Path(upload_dir).exists()}', file=sys.stderr)

if Path(upload_dir).exists():
    files = list(Path(upload_dir).glob('*'))
    print(f'Files in upload dir: {[f.name for f in files]}', file=sys.stderr)

try:
    kb = service.ensure_loaded(index_path)
    print(f'Loaded KB documents count: {len(kb.documents)}', file=sys.stderr)
    for doc in kb.documents:
        print(f'  - {doc.path} (chunk_id={doc.chunk_id}, text_len={len(doc.text)})', file=sys.stderr)
except Exception as e:
    print(f'Error loading: {type(e).__name__}: {e}', file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)