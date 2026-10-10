
import hashlib, json, os, re, sys, time
def _f1_run(path):
    # F1b: every payload cell is materialised as a real file and compiled with that path, so code that needs
    # a source file (numba @njit(cache=True), inspect.getsource, tracebacks) behaves as it did in the kernel.
    payload = json.load(open(path, encoding='utf-8'))
    g = sys.modules['__main__'].__dict__
    cells_dir = os.path.join(os.getcwd(), 'cells')  # cwd = this run's own fresh f1_run directory
    os.makedirs(cells_dir, exist_ok=False)
    for cell in payload['cells']:
        src, label = cell['source'], cell['label']
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', label):
            raise RuntimeError('F1 payload label not a safe file name: ' + label)
        if hashlib.sha256(src.encode('utf-8')).hexdigest() != cell['sha256']:
            raise RuntimeError('F1 payload cell drift: ' + label)
        cell_path = os.path.join(cells_dir, label + '.py')
        with open(cell_path, 'x', encoding='utf-8', newline='') as f:
            f.write(src)
        with open(cell_path, 'rb') as f:
            if hashlib.sha256(f.read()).hexdigest() != cell['sha256']:
                raise RuntimeError('F1 materialised cell differs: ' + label)
        t = time.monotonic()
        print('F1_CELL_START', label, flush=True)
        exec(compile(src, cell_path, 'exec'), g)
        print('F1_CELL_DONE', label, round(time.monotonic() - t, 1), flush=True)
    print('F1_PAYLOAD_COMPLETE', flush=True)
_f1_run(sys.argv[1])
