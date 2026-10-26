import ast, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
OUT=Path(__file__).resolve().parent; ROOT=OUT.parent; MIRROR=OUT/'verification_repo'
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
if '--child' in sys.argv:
    sys.dont_write_bytecode=True
    # Mocked OpenJEV transport still checks for a configured key first.
    # This dummy is deliberately unusable and never reaches a network.
    if '--dummy-provider-config' in sys.argv:
        os.environ['OPENJEV_API_KEY']='documentation-test-dummy'
        os.environ['OPENJEV_ENABLED']='false'
    suffix='configured' if '--dummy-provider-config' in sys.argv else 'clean'
    sys.path.insert(0,str(MIRROR));os.chdir(MIRROR)
    import socket, unittest, logging
    logging.disable(logging.CRITICAL)
    def denied(*a,**k):raise RuntimeError('Documentation verification blocks external network')
    socket.socket.connect=denied;socket.create_connection=denied
    suite=unittest.defaultTestLoader.discover(str(MIRROR/'evaluation/tests'),pattern='test_*.py',top_level_dir=str(MIRROR))
    with (OUT/f'tests-{suffix}.log').open('w',encoding='utf-8') as handle:
        result=unittest.TextTestRunner(stream=handle,verbosity=2).run(suite)
    summary={'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),'successful':result.wasSuccessful(),'network':'blocked','execution':'isolated source mirror; no .env copied'}
    (OUT/f'test-summary-{suffix}.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary))
    sys.exit(0 if result.wasSuccessful() else 1)
rows=json.loads((OUT/'inventory.json').read_text())
MIRROR.mkdir(exist_ok=True)
for row in rows:
    rel=row['path']
    if rel=='.env' or rel.endswith('.log') or rel.startswith('images/') or rel.endswith('audit.jsonl'):continue
    src=ROOT/rel;dst=MIRROR/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
proc=subprocess.run([sys.executable,'-B',str(__file__),'--child',*sys.argv[1:]],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
(OUT/'test-process.log').write_text(proc.stdout+'\n'+proc.stderr,encoding='utf-8')
changed=[]
for row in rows:
    p=ROOT/row['path']
    if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:changed.append(row['path'])
(OUT/'source-integrity.json').write_text(json.dumps({'preexisting_files_changed':changed,'checked_files':len(rows)},indent=2))
print(proc.stdout);print('Exit',proc.returncode,'preexisting files changed',len(changed))
