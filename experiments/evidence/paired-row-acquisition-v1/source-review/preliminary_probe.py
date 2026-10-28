from pathlib import Path
import sys,json,hashlib
repo=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');sys.path[:0]=[str(repo/'examples/isolated_glu'),str(repo/'src')]
import row_acquisition_controller as c
c.public_boundary=lambda *a:None
c.read_json=lambda *a:{'mapping':[{'id':f'c{i:02d}','width':w} for i,w in enumerate((16,48,96))]}
c.exact_store=lambda *a:None
bad={'cases':[{'case':'duplicate','probes':16,'methods':[{'method':'gap','qualifies':False},{'method':'stable','qualifies':False}]}]*12,'rule_passes':{'gap':True,'stable':True},'by_budget':{'17':{'gap':True,'stable':True}},'paired':[],'certificate':False,'new_queries':0,'new_fits':0,'private_gate':'closed'}
c.validate_final(Path('/literal-only-unread'),{}, {},bad)
r={'status':'counterexample_reproduced','claim':'Controller validates no evaluation names/order/derived aggregate binding;12duplicates with failing cases but success aggregate accepted under isolated IO stubs.','validation_io_stubbed':True,'new_queries_RNG_forward_fits':0,'controller_sha256':hashlib.sha256((repo/'examples/isolated_glu/row_acquisition_controller.py').read_bytes()).hexdigest()}
p=Path(__file__).with_name('preliminary-probe-results.json');p.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
