import sys,json,tempfile
from pathlib import Path
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');sys.path[:0]=[str(R/'src'),str(R/'examples/isolated_glu')]
import rank_analysis as m
p=Path(tempfile.mkdtemp(dir='/private/tmp/modelforge-rank-analysis-review'))/'choices-sealed.json'
orig=Path.write_bytes
try:
 def short(self,data):orig(self,data[:-1]);return len(data)-1
 Path.write_bytes=short
 try:m.save_json(p,{'selections':[]})
 except OSError:pass
 else:raise AssertionError('short write accepted')
 assert not p.exists() and p.with_suffix('.tmp').exists()
finally:Path.write_bytes=orig
result={'passed':True,'short_write_rejected_before_final_seal_exists':True,'actual_data_reads':0,'queries':0,'fits':0,'RNG':0,'source_sha256':m.sha(R/'examples/isolated_glu/rank_analysis.py')}
Path('/private/tmp/modelforge-rank-analysis-review/short-write-probe-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
