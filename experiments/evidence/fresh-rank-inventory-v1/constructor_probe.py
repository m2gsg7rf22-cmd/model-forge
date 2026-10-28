import sys,json,hashlib
from pathlib import Path
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');P=Path('/private/tmp/modelforge-fresh-inventory-review');sys.path.insert(0,str(R/'examples/isolated_glu'))
import fresh_rank_inventory as m
checks={}
def rejected(key,fields):
 try:m.ShardSpec('truth','literal',fields,1)
 except ValueError:checks[key]=True
 else:checks[key]=False
rejected('mutable_list_shape_rejected',{'x':[1]})
for field in ['file','allow_pickle','x'*65,'x/../bad','שם']:rejected('invalid_name_'+field[:10],{field:(1,)})
for shape in [(),(True,),(0,),(-1,)]:rejected('invalid_shape_'+str(shape),{'x':shape})
d={'x':(1,)};s=m.ShardSpec('truth','literal',d,1);d['x']=(10**9,);assert s.payload_bytes==8
try:s.fields['x']=(2,)
except TypeError:checks['copied_mapping_immutable']=True
else:raise AssertionError('mapping')
long=(1,)*2000;hdr=repr({'descr':'<f8','fortran_order':False,'shape':long}).encode();assert len(hdr)>4096
rejected('overlong_shape_rank_rejected',{'x':long})
result={'checks':checks,'all_passed':all(checks.values()),'literal_header_counterexample_bytes':len(hdr),'source_sha256':hashlib.sha256((R/'examples/isolated_glu/fresh_rank_inventory.py').read_bytes()).hexdigest(),'array_or_RNG_or_query_or_fit':False}
(P/'constructor-probe-results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result))
