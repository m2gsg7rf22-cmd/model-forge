# Literal-only journal truncation control; no actual data or arrays.
repo=__import__('pathlib').Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository')
out=__import__('pathlib').Path('/private/tmp/modelforge-joint-audit-storage-jobs-review')
import sys,tempfile
from pathlib import Path
sys.path.insert(0, str(repo/'examples/isolated_glu'))
from joint_response_audit_storage import AuditStore,CHECKPOINTS
with tempfile.TemporaryDirectory(prefix='literal-journal-',dir=str(out)) as td:
    store=AuditStore(Path(td)/'case')
    for step in CHECKPOINTS: store.append({'step':step})
    store.path.write_bytes(b'')
    error=None
    try: store.complete({'checkpoints':store.rows,'status':'agreed','certificate':False,'private_gate':'closed'})
    except Exception as exc: error=type(exc).__name__+': '+str(exc)
    result=dict(error=error,done=store.done,failed=store.failed,journal_actual_bytes=store.path.stat().st_size,memory_committed_records=len(store.rows),result_exists=(store.root/'result.json').exists())
    assert error is not None and store.failed and not store.done and not result['result_exists']
