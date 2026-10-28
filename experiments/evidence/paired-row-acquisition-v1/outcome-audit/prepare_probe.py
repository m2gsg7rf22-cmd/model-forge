from pathlib import Path
s=Path('/private/tmp/modelforge-fresh-width-outcome-audit/probe.py').read_text()
s=s.replace("P=Path('/private/tmp/modelforge-fresh-width-outcome-audit');O=R.parent/'experiments/fresh-variable-width-rank-v1-20261028-bb46845'","P=Path('/private/tmp/modelforge-row-acquisition-outcome-audit');O=R.parent/'experiments/paired-row-acquisition-v1-20261028-07235ac'")
s=s.replace("SOURCE='bb46845d1efccb80bdfad057d85dd0292e9bf03a'","SOURCE='07235acd0ecaa576d57b37eefb2b4e6fc3aa0c6a'")
s=s.replace('def schemas(n,m):','def schemas(n,m,budget=16):').replace('q=min(16*n,p);count=min(q,p-1)','q=min(budget*n,p);count=min(q,p-1)')
s=s.replace('probes=(16,n)','probes=(17,n)').replace('preactivations=(16,m)','preactivations=(17,m)')
a=s.index('def Hload(');b=s.index('\ndef best(',a)
s=s[:a]+'''def Hload(directory,case,kind,n):
 return np.concatenate([npz(directory,f'{case}-{kind}-h{j:02d}.npz',{'hessian':((8 if j<2 else 1),n,n,n)})['hessian'] for j in range(3)])
'''+s[b:]
s=s.replace('fresh-variable-width-rank-v1/source-review','paired-row-acquisition-v1/source-review').replace('8a27edccec53b1c544c774745e2024b6a9a08dcb09395d44bc81c86190cd4f01','6a819f3654a696cacd9297d32a5897f0106e670c3168dc95dea4fcd84a069acf').replace("review['full_fresh_protocol_approved'] and len(review['source_sha256'])==26","review['full_row_acquisition_protocol_approved'] and len(review['source_sha256'])==31")
s=s.replace('preregistrations/fresh-variable-width-rank-v1.json','preregistrations/paired-row-acquisition-v1.json').replace('ada5e3dcfaa50078f152b030d1808e7f8bd03f4c9388fcb1ac13b871455b5221','028a7997327b6c1d307ebfed9c8dc63d996a74e8611cece61eb9b4ae769fe84d').replace("len(config['frozen_dependencies'])==17","len(config['frozen_dependencies'])==24")
s=s.replace("meta['schema']=='modelforge-fresh-rank-observations-v1'","meta['schema']=='modelforge-row-acquisition-observations-v1'")
a=s.index(" cases=meta['cases'];expected_cases=[]");b=s.index(' assert sha(estdir/',a)
s=s[:a]+''' basecases=meta['cases'];expected_base=[]
 for t in range(3):
  for suffix,channel in [('fp64','fp64'),('fp32','fp32_compute')]:expected_base.append(dict(name=f'c{t:02d}-{suffix}',n=32,outputs=32,probes=17,compute=channel))
 assert basecases==expected_base
 cases=[dict(base,name=base['name']+f'-p{budget}',probes=budget) for base in basecases for budget in (16,17)]
 assert len(seal['selections'])==12 and inp['inputs_unchanged']
 public_records=manifest(pubdir);est_records=manifest(estdir/'arrays');assert len(public_records)==54 and len(est_records)==24
'''+s[b:]
s=s.replace("name,n=case['name'],case['n'];p=n*(n+1)//2;q=min(16*n,p);count=min(q,p-1);floor=64*np.finfo(float).eps*max(16*n,p);sch=schemas(n,1)","name,n=case['name'],case['n'];base=basecases[index//2]['name'];budget=case['probes'];p=n*(n+1)//2;q=min(budget*n,p);count=min(q,p-1);floor=64*np.finfo(float).eps*max(budget*n,p);sch=schemas(n,1,budget)")
s=s.replace("h=Hload(pubdir,name,kind,n);assert np.array_equal(h,h.swapaxes(-1,-2));mats.append(pack(h).reshape(16*n,p))","h=Hload(pubdir,base,kind,n)[:budget];assert np.array_equal(h,h.swapaxes(-1,-2));mats.append(pack(h).reshape(budget*n,p))")
s=s.replace('all24_public_choices_verified_before_truth_reveal','all12_public_choices_verified_before_truth_reveal')
a=s.index(" for n,widths in [(4,(2,4,8))");b=s.index(' assert mapping==expected_mapping',a)
s=s[:a]+''' order=sorted((16,48,96),key=lambda m:hashlib.sha256(seed+f'/row-acquisition-v1/assignment/{m}'.encode()).digest())
 for m in order:expected_mapping.append(dict(id=f'c{len(expected_mapping):02d}',n=32,width=m,kind='generic'))
'''+s[b:]
s=s.replace("len(tr)==1608 and len(evrec)==24","len(tr)==432 and len(evrec)==12").replace("len(evreport['cases'])==24","len(evreport['cases'])==12")
s=s.replace("return bool(not sensitive and r>0 and ((k is None) if r==p else k==r and all(v is not None and np.isfinite(v) for v in [lost,extra,gm,rm]) and 0<=lost<=.01 and 0<=extra<=.01 and 0<=gm<=.01 and rm>gm))","return bool(not sensitive and 0<r<p and k==r and all(v is not None and np.isfinite(v) for v in [lost,extra,gm,rm]) and 0<=lost<=.01 and 0<=extra<=.01 and 0<=gm<=.01 and rm>gm)")
a=s.index(' for index,item in enumerate(cached):');b=s.index("  for key in ['g','u','d']:",a)
s=s[:a]+''' for baseindex,case in enumerate(basecases):
  name,n=case['name'],case['n'];teacher=mapping[baseindex//2];m=teacher['width'];p=n*(n+1)//2;sch=schemas(n,m);ctx=npz(truthdir,name+'-context.npz',sch['context']);expected_truth.add(name+'-context.npz')
'''+s[b:]
s=s.replace('if index%2==0:pair_ctx=', 'if baseindex%2==0:pair_ctx=')
s=s.replace('for j in range(2):expected_truth.add', 'for j in range(3):expected_truth.add').replace('for j in range(2):expected_public.add', 'for j in range(3):expected_public.add')
a=s.index('  reference=pack(href)');b=s.index(' assert set(tr)==expected_truth',a)
segment=s[a:b]
segment=segment.replace('reference=pack(href).reshape(16*n,p);observed=pack(savedH[\'richardson\']).reshape(16*n,p)',"reference=pack(href[:budget]).reshape(budget*n,p);observed=pack(savedH['richardson'][:budget]).reshape(budget*n,p)")
segment=segment.replace("record['case']==name and record['width_for_evaluation_only']", "record['case']==condition['name'] and record['base_case']==name and record['probes']==budget and record['shared_context_sha256']==tr[name+'-context.npz']['sha256'] and record['width_for_evaluation_only']")
segment=segment.replace("  elif r==p:\n    if k is not None:reason.append('full_reference_requires_null')", "  elif r==p:reason.append('unexpected_full_reference')")
segment=segment.replace("dict(case=name,n=n,width=m", "dict(case=condition['name'],base_case=name,probes=budget,n=n,width=m")
segment=segment.replace('  del ctx,savedH,fd,href,analytic,reference,observed,e','  del reference,observed,e')
# Nest the condition metrics under each saved base context. Every value already selected public-only.
s=s[:a]+'''  for local,budget in enumerate((16,17)):
   index=2*baseindex+local;item=cached[index];condition=item['case'];q=item['q'];sch=schemas(n,m,budget)
'''+''.join(' '+line+'\n' for line in segment.splitlines())+'''  del ctx,savedH,fd,href,analytic
'''+s[b:]
s=s.replace('vectors==545280 and batches==768','vectors==430848 and batches==204')
a=s.index(' actualrules={rule:all(');b=s.index(" progress('final_caps_and_hashes')",a)
s=s[:a]+''' by_budget={str(b):{rule:all(next(m['qualifies'] for m in row['methods'] if m['method']==rule) for row in table if row['probes']==b) for rule in ['gap','stable']} for b in (16,17)}
 paired=[]
 for i,base in enumerate(basecases):
  a,b=table[2*i:2*i+2]
  paired.append(dict(base_case=base['name'],reference_ranks16=a['reference_ranks'],reference_ranks17=b['reference_ranks'],reference_sensitive16=a['reference_sensitive'],reference_sensitive17=b['reference_sensitive'],rules={rule:dict(pass16=next(m['qualifies'] for m in a['methods'] if m['method']==rule),pass17=next(m['qualifies'] for m in b['methods'] if m['method']==rule)) for rule in ['gap','stable']}))
 assert evreport['by_budget']==by_budget and evreport['paired']==paired
 actualrules=by_budget['17'];assert evreport['rule_passes']==controller['rule_passes']==actualrules
'''+s[b:]
a=s.index(" transport=O/'transport/arrays'");b=s.index(" storemap=",a)
s=s[:a]+''' transport=O/'transport/arrays';tman=manifest(transport);sch=schemas(32,96);transportschemas={
 'context.npz':sch['context'],'inputs0.npz':sch['inputs'],'responses0.npz':sch['responses'],
 'reference.npz':{'hessian':(8,32,32,32)},'reference_tail.npz':{'hessian':(1,32,32,32)},
 'bases16.npz':sch['bases'],'choices16.npz':sch['choices'],
 'bases17.npz':schemas(32,96,17)['bases'],'choices17.npz':schemas(32,96,17)['choices'],'evaluation16.npz':sch['evaluation']};assert set(tman)==set(transportschemas)
 for name,sc in transportschemas.items():npz(transport,name,sc)
'''+s[b:]
s=s.replace("marker=O.parent/'fresh-variable-width-rank-v1-attempt.json'","marker=O.parent/'paired-row-acquisition-v1-attempt.json'")
s=s.replace('scientific==1824','scientific==522').replace('signed_source_files_verified=26,runtime_pins=17,all24_choices_recomputed_before_truth_reveal=True','signed_source_files_verified=31,runtime_pins=24,all12_choices_recomputed_before_truth_reveal=True').replace('transport_archives_schema_checked=7','transport_archives_schema_checked=10').replace('rule_passes=actualrules,cases=table','rule_passes=actualrules,by_budget=by_budget,paired=paired,cases=table').replace("status='passed',cases=24","status='passed',cases=12")
Path('/private/tmp/modelforge-row-acquisition-outcome-audit/probe.py').write_text(s)
launch=Path('/private/tmp/modelforge-fresh-width-outcome-audit/launcher.py').read_text().replace('/private/tmp/modelforge-fresh-width-outcome-audit','/private/tmp/modelforge-row-acquisition-outcome-audit')
Path('/private/tmp/modelforge-row-acquisition-outcome-audit/launcher.py').write_text(launch)
