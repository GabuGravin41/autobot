
# RDKit is not in the Kaggle image and internet is off for code competitions, so install the
# wheel from an attached dataset. Only the fragmentation channel needs it.
import subprocess, sys, glob
whl = glob.glob('/kaggle/input/**/rdkit-*.whl', recursive=True)
if whl:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', whl[0]], check=False)
try:
    from rdkit import Chem
    HAVE_RDKIT = True
except Exception:
    HAVE_RDKIT = False
print('RDKit available:', HAVE_RDKIT, '(fragmentation channel is optional - the notebook runs without it)')
