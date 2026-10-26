import ast, hashlib, json, os, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def git(*args):
    p = subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace')
    return p.stdout

def inventory():
    rows=[]
    for base,dirs,files in os.walk(ROOT):
        dirs[:]=[d for d in dirs if d not in {'.git','.venv','__pycache__','.project_overview'}]
        for name in files:
            p=Path(base)/name; rel=p.relative_to(ROOT).as_posix()
            row={'path':rel,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            if p.suffix=='.py':
                txt=p.read_text(encoding='utf-8-sig'); tree=ast.parse(txt)
                row['lines']=len(txt.splitlines())
                row['symbols']=[{'name':n.name,'line':n.lineno,'end':n.end_lineno,'type':type(n).__name__} for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))]
                row['imports']=[ast.unparse(n) for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
            rows.append(row)
    (OUT/'inventory.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    for suffix,args in [('status',['status','--short','--branch']),('history',['log','--all','--reverse','--format=fuller','--stat']),('refs',['show-ref']),('diffstat',['diff','--stat'])]:
        (OUT/f'git-{suffix}.txt').write_text(git(*args),encoding='utf-8')
    print('Inventory',len(rows),'files;',sum('symbols' in r for r in rows),'Python files;',sum(r.get('lines',0) for r in rows),'Python lines')

def read(paths, compact=False):
    for rel in paths:
        p=ROOT/rel
        print('\nFILE',rel)
        txt=p.read_text(encoding='utf-8-sig'); skip=set()
        if compact and p.suffix=='.py':
            for n in ast.walk(ast.parse(txt)):
                if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value,str):
                    skip.update(range(n.lineno,n.end_lineno+1))
        for i,line in enumerate(txt.splitlines(),1):
            if compact and (i in skip or not line.strip() or line.lstrip().startswith('#')): continue
            print(f'{i:4} {line}')

if __name__=='__main__':
    if len(sys.argv)==1: inventory()
    elif sys.argv[1]=='read': read(sys.argv[2:])
    elif sys.argv[1]=='compact': read(sys.argv[2:],True)
    elif sys.argv[1]=='symbols':
        for r in json.loads((OUT/'inventory.json').read_text()):
            if 'symbols' in r: print(r['path'],r['lines'], '; '.join(f"{s['name']}:{s['line']}" for s in r['symbols']))
