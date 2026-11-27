"""Loopback-only prediction application. Run: python -m ml_projects.app --port 8501."""
import argparse,csv,io,json,logging,threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from urllib.parse import urlsplit
from .advanced import ROOT,OUT,PROJECT,load_artifact,predict

LOG=logging.getLogger(__name__)

def csv_records(text):
    if not isinstance(text,str): raise ValueError('CSV must be text')
    reader=csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    if not reader.fieldnames or len(set(reader.fieldnames))!=len(reader.fieldnames): raise ValueError('CSV requires unique column headers')
    records=[]
    for row in reader:
        if None in row or any(value is None for value in row.values()): raise ValueError('CSV row does not match its headers')
        records.append(row)
        if len(records)>5000: raise ValueError('Maximum 5000 rows per request')
    return records

def create_server(port=8501, demo=False):
    mode='synthetic_demo' if demo else 'trained_model'
    if demo:
        from .demo import demo_bundle, demo_predict
        schema, example, summary, example_csv=demo_bundle(PROJECT)
        sample=[example]
        run_prediction=lambda records, threshold: demo_predict(PROJECT,schema,example,records,threshold)
    else:
        try:
            artifact=load_artifact(); summary=json.loads((OUT/'summary.json').read_text())
            example_csv=(OUT/'example_inputs.csv').read_bytes()
        except FileNotFoundError as exc:
            raise FileNotFoundError('Trained artifacts are missing. Run with --demo for the fictional simulator, or follow GITHUB_PREPARATION.md to train models.') from exc
        schema=artifact['schema']; sample=csv_records(example_csv.decode('utf-8'))
        run_prediction=lambda records, threshold: predict(artifact,records,threshold)
    lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,format,*args): LOG.info('%s %s',self.command,self.path)
        def reply(self,status,body,content_type='application/json'):
            if not isinstance(body,bytes): body=json.dumps(body,allow_nan=False).encode()
            self.send_response(status); self.send_header('Content-Type',content_type); self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store'); self.send_header('X-Content-Type-Options','nosniff'); self.send_header('X-Frame-Options','DENY'); self.end_headers(); self.wfile.write(body)
        def allowed(self):
            host=self.headers.get('Host','')
            allowed={f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}
            origin=self.headers.get('Origin')
            if host not in allowed or (origin and origin not in {'http://'+h for h in allowed}):
                self.reply(403,{'error':'This application accepts local same-origin requests only'}); return False
            return True
        def do_GET(self):
            if not self.allowed(): return
            path=urlsplit(self.path).path
            if path=='/api/health': self.reply(200,{'status':'ok','project':PROJECT,'mode':mode})
            elif path=='/api/schema': self.reply(200,{'project':PROJECT,'schema':schema,'sample':sample[0],'summary':summary,'mode':mode})
            elif path=='/example.csv': self.reply(200,example_csv,'text/csv; charset=utf-8')
            elif path in {'/','/app.js','/demo.js','/style.css'}:
                name={'/':'index.html','/app.js':'app.js','/demo.js':'demo.js','/style.css':'style.css'}[path]
                mime={'/':'text/html','/app.js':'text/javascript','/demo.js':'text/javascript','/style.css':'text/css'}[path]
                self.reply(200,(ROOT/'web'/name).read_bytes(),mime+'; charset=utf-8')
            else: self.reply(404,{'error':'Not found'})
        def do_POST(self):
            if not self.allowed(): return
            if urlsplit(self.path).path!='/api/predict': self.reply(404,{'error':'Not found'}); return
            try:
                length=int(self.headers.get('Content-Length','0'))
                if length<=0 or length>2_000_000: self.reply(413,{'error':'Request must be between 1 byte and 2 MB'}); return
                if self.headers.get_content_type()!='application/json': self.reply(415,{'error':'Send application/json'}); return
                payload=json.loads(self.rfile.read(length))
                if not isinstance(payload,dict): raise ValueError('Request must be an object')
                records=csv_records(payload['csv']) if 'csv' in payload else payload.get('records')
                with lock: result=run_prediction(records,payload.get('threshold'))
                self.reply(200,result)
            except (ValueError,TypeError,KeyError,OverflowError) as exc: self.reply(422,{'error':str(exc)})
            except Exception:
                LOG.exception('Prediction failed'); self.reply(500,{'error':'Prediction failed; inspect the local server log'})
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--port',type=int,default=8501)
    parser.add_argument('--demo',action='store_true',help='Run the fictional simulator without datasets or trained models')
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(levelname)s %(message)s')
    server=create_server(args.port,demo=args.demo)
    print('SYNTHETIC DEMO — invented formulas, not a trained model' if args.demo else 'TRAINED MODEL mode',flush=True)
    print(f'Open http://127.0.0.1:{server.server_port}',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
