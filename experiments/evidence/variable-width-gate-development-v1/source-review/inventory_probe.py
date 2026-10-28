"""Independent literal maximal-shape inventory reconstruction. No array/RNG creation."""
from pathlib import Path
import sys,json,hashlib
from math import prod
R=Path('/Users/adamfaruhi123/.local/share/modelforge-research/repository');P=Path(__file__).parent;sys.path[:0]=[str(R/'examples/isolated_glu'),str(R/'src')]
import variable_gate_inventory as actual
shapes={}
for job in range(6):
 prefix=f'job{job:02d}'
 def add(suffix,fields):
  name=prefix+'-'+suffix;assert name not in shapes;shapes[name]=fields
 state={'initial':(512,),'rescue':(512,),'initial_length':(1,),'rescue_length':(1,)}
 add('basis',{'basis':(192,528)});add('rng-before',state);add('rng-after',state)
 add('initial-raw',{'raw':(256,32)});add('initial-starts',{'starts':(256,32)})
 pre={'starts':(256,32),'optimized':(256,32),'before_losses':(256,)}
 add('initial-pre-refine',pre);add('initial-refine',dict(pre,refined=(256,32),losses=(256,)))
 add('initial-merge',{k:(256,) for k in ('order','decision','max_cosine','nearest_index','losses')})
 add('final',{'directions':(607,32),'losses':(607,),'cosines':(607,607),'count':(1,)})
 add('view-residuals',{'losses':(3,607),'count':(1,)})
 chunk={'refined':(8,32),'losses':(8,),'count':(1,),'attempted':(1,),'complete':(1,),'index':(1,)}
 intent={'index':(1,),'requested':(1,)}
 for i in range(32):
  add(f'initial-refine-intent{i:03d}',intent);add(f'initial-refine-chunk{i:03d}',chunk)
 for round in range(4):
  rp=f'r{round}-'
  add(rp+'input',{'known':(95,32)})
  add(rp+'geometry',{'known':(95,32),'residual':(192,528),'geometry':(3,),'tangent_singular':(95,3)})
  add(rp+'mixtures',{'raw':(256,190),'normalized':(256,190)})
  add(rp+'eigen',{'values':(256,32),'vectors':(256,32,32),'indices':(256,2)})
  pr={'starts':(512,32),'before_losses':(512,)};add(rp+'pre-refine',pr);add(rp+'refine',dict(pr,refined=(512,32),losses=(512,)))
  add(rp+'merge',{k:(512,) for k in ('order','decision','max_cosine','nearest_index','losses')})
  for i in range(32):
   add(rp+f'mix-intent{i:03d}',intent)
   add(rp+f'mix-pre{i:03d}',{'raw':(8,190),'normalized':(8,190),'index':(1,)})
   add(rp+f'eigen-chunk{i:03d}',{'values':(8,32),'vectors':(8,32,32),'indices':(8,2),'starts':(16,32),'count':(1,),'attempted':(1,),'complete':(1,),'index':(1,)})
  for i in range(64):add(rp+f'refine-intent{i:03d}',intent);add(rp+f'refine-chunk{i:03d}',chunk)
assert len(shapes)==5988
assert set(shapes)=={n for j in range(6) for n in actual.trace_names(f'job{j:02d}')}
assert all(actual.upper_fields(name)==fields for name,fields in shapes.items())
def archive(fields):return 8*sum(prod(s) for s in fields.values())+4096*len(fields)
def line(name,size):return len((json.dumps({'name':name+'.npz','bytes':size,'sha256':'0'*64},sort_keys=True,separators=(',',':'))+'\n').encode())
sizes={name:archive(fields) for name,fields in shapes.items()};manifest=sum(line(n,v) for n,v in sizes.items());traces=sum(sizes.values())+manifest
unique={json.dumps(f,sort_keys=True):f for f in shapes.values()};assert len(unique)==20
actual_unique={json.dumps(f,sort_keys=True) for f in actual.transport_schemas().values()};assert set(unique)==actual_unique
preflight=sum(archive(f)+line('family00',archive(f)) for f in unique.values())
plan={'trace_shards':len(shapes),'trace_bytes_upper_bound':traces,'manifest_bytes_upper_bound':manifest,'largest_shard_upper_bound':max(sizes.values()),'preflight_shards':len(unique),'preflight_bytes_upper_bound':preflight,'metadata_reserve_bytes':8*1024**2,'parent_retained_bytes_upper_bound':534980568,'bytes_upper_bound':traces+preflight+8*1024**2+534980568,'minimum_free_disk_bytes':2*1024**3}
assert plan==actual.inventory();assert manifest<1024**2 and plan['bytes_upper_bound']<1024**3
out={'status':'passed','independent_literal_shape_plan':plan,'every5988shape_and20family_match':True,'source_sha256':hashlib.sha256((R/'examples/isolated_glu/variable_gate_inventory.py').read_bytes()).hexdigest(),'arrays_RNG_query_or_fit_created':False}
(P/'inventory-probe-results.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
