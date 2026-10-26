import ast, collections, hashlib, json, os, re, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT)); sys.stdout.reconfigure(encoding='utf-8',errors='replace')
from pypdf import PdfReader
from ingestion.src import metadata, discovery, validation
from evaluation import dataset

def save(name,value): (OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
records=[]
for src in discovery.discover(ROOT/'Data')[0]:
    pages=[p.extract_text() or '' for p in PdfReader(src.path).pages] if src.format=='pdf' else [src.path.read_text(encoding='utf-8-sig')]
    md=metadata.build_metadata(src,pages)
    issues=validation.validate(src,md)
    records.append({'path':'Data/'+src.rel_path,'pages':len(pages),'characters':sum(map(len,pages)),'metadata':md.values,'issues':[i.to_dict() for i in issues]})
save('data-inspection.json',records)
docs=[json.loads(s) for s in (ROOT/'ingestion/processed/documents/documents.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
chunks=[json.loads(s) for s in (ROOT/'ingestion/processed/chunks/chunks.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
gd=dataset.load(); cat=dataset.PermissionCatalog()
summary={'data_files':len(records),'pdfs':sum(r['path'].endswith('.pdf') for r in records),'documents':len(docs),'chunks':len(chunks),'max_tokens':max(c['token_count'] for c in chunks),'categories':dict(collections.Counter(r['path'].split('/')[2] if r['path'].startswith('Data/documents/') else r['path'].split('/')[1] for r in records)),'classifications':dict(collections.Counter(c['classification'] for c in docs)),'golden':gd.summary(),'golden_validation':dataset.validate(gd,cat),'permission_counts':{u:len(cat.authorized_documents(u)) for u in dataset.USERS},'data_metadata_errors':sum(i['severity']=='ERROR' for r in records for i in r['issues']),'data_metadata_warnings':sum(i['severity']=='WARNING' for r in records for i in r['issues'])}
save('data-summary.json',summary);print(json.dumps(summary,indent=2))
runs=[]
for p in [ROOT/'evaluation/results.json',*sorted((ROOT/'evaluation/runs').glob('*.json')),ROOT/'ingestion/reports/ingestion_summary.json',ROOT/'indexing/reports/indexing_report.json']:
    d=json.loads(p.read_text(encoding='utf-8')); item={'path':p.relative_to(ROOT).as_posix(),'kind':type(d).__name__}
    if isinstance(d,dict):
        item['keys']=list(d)
        for k in ['run_id','generated_at','timestamp','summary','aggregate','aggregates','metrics','totals','overall','partial','selected_ids','selected_cases','complete','baseline','phase6','phase7','phase8','phase9','results_summary','checks','failures','errors','schema_version','cases','config','configurations']:
            if k in d:
                v=d[k]
                if isinstance(v,list): item[k+'_count']=len(v)
                elif len(json.dumps(v))<20000:item[k]=v
        if 'results' in d and isinstance(d['results'],list):item['result_count']=len(d['results'])
    else:item['count']=len(d)
    runs.append(item)
save('saved-reports-summary.json',runs)
env=[]
for p in [ROOT/'.env.example',ROOT/'.env']:
    if p.exists():env.append({'path':p.name,'names':re.findall(r'^\s*([A-Z][A-Z0-9_]*)\s*=',p.read_text(encoding='utf-8'),re.M)})
save('env-names.json',env)
contracts=[]
for p in sorted((ROOT/'evaluation').glob('*.py'))+sorted((ROOT/'evaluation/tests').glob('*.py')):
    t=p.read_text(encoding='utf-8'); tree=ast.parse(t)
    contracts.append({'path':p.relative_to(ROOT).as_posix(),'doc':ast.get_docstring(tree),'functions':[{'name':n.name,'line':n.lineno,'doc':ast.get_docstring(n),'calls':sorted(set(ast.unparse(x.func) for x in ast.walk(n) if isinstance(x,ast.Call)))} for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef))],'cli':[ast.unparse(n) for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='add_argument']})
save('evaluation-contracts.json',contracts)
hist=[]
for commit in subprocess.check_output(['git','rev-list','--all','--reverse'],cwd=ROOT,text=True).splitlines():
    patch=subprocess.check_output(['git','show','--format=fuller','--no-ext-diff',commit,'--','*.py','requirements.txt','.gitignore','.env.example','run.bat'],cwd=ROOT,text=True,encoding='utf-8',errors='replace')
    (OUT/'history').mkdir(exist_ok=True)
    (OUT/'history'/f'{commit[:7]}.diff').write_text(patch,encoding='utf-8')
    hist.append({'commit':commit,'python_patch_lines':len(patch.splitlines())})
save('history-coverage.json',hist)
print('Read all 63 data files for metadata; all saved JSON reports parsed; evaluation AST contracts saved; source Git patches saved.')
