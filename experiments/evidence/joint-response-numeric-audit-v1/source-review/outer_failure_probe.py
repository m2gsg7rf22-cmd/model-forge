out=__import__("pathlib").Path('/private/tmp/modelforge-joint-numeric-audit-full-source-review')
import json,tempfile
from pathlib import Path
from unittest.mock import patch
import joint_response_audit_controller as m
with tempfile.TemporaryDirectory(dir=str(out)) as td:
    repo=Path(td)/'repo'; repo.mkdir()
    output=Path(td)/'experiments'/m.POLICY['id']; output.parent.mkdir()
    def engine(*args,**kw):
        output.mkdir(); (output/'controller-report.json').write_text('{"status":"completed"}')
        return dict(status='completed',memory_limit_mb=512,process_monitor_scope='process_and_children',memory_samples=2,peak_sampled_rss_bytes=1024)
    original=m.save
    once=[]
    def saved(path,value,*args):
        original(path,value,*args)
        if path.name=='outer-monitor.json' and value.get('status')=='completed' and not once:
            once.append(True); raise OSError('literal injected after-write failure')
    with patch.object(m,'verify_source',return_value=({},'a'*40)),patch.object(m,'layout',return_value={}),patch.object(m,'save',side_effect=saved):
        error=None
        try: m.launch(dict(repo=str(repo),output=str(output),config=str(repo/'c'),review=str(repo/'r')),engine=engine)
        except Exception as exc: error=type(exc).__name__
    result=dict(error=error,saved_outer=json.loads((output/'outer-monitor.json').read_text()),actual_data_reads=0,rng_calls=0,fits=0)
print(json.dumps(result))
