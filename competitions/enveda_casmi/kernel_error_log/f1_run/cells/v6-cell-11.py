
# ===================================================================================
#  Candidate pool = COCONUT (attached, CC-BY) U training structures (rebuilt here).
#  Only the COCONUT half is redistributable, so the other half is computed at runtime.
# ===================================================================================
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator, MACCSkeys
from rdkit.Chem.Descriptors import ExactMolWt
from multiprocessing import Pool as MPool
RDLogger.DisableLog('rdApp.*')

BITS = np.load(find('fp_bits.npy'))
_g = {}
def _fp_init():
    _g['m2'] = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
    _g['m3'] = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096)
    _g['rk'] = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048, maxPath=6)

def fp_and_mass(smi):
    if not _g: _fp_init()
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    try:
        fp = np.concatenate([_g['m2'].GetFingerprintAsNumPy(m).astype(np.uint8),
                             _g['m3'].GetFingerprintAsNumPy(m).astype(np.uint8),
                             _g['rk'].GetFingerprintAsNumPy(m).astype(np.uint8),
                             np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.uint8)])[BITS]
        return fp, float(ExactMolWt(m))
    except Exception:
        return None

class Pool:
    """Candidates + fingerprints, sorted by exact mass."""
    def __init__(s, fp, mass, keys, smiles, nbits):
        o = np.argsort(mass)
        s._fp = fp[o]; s.mass = mass[o]
        s.keys = np.asarray(keys, dtype=object)[o]
        s.smiles = np.asarray(smiles, dtype=object)[o]
        s.nbits = nbits
        s.k2i = {k: i for i, k in enumerate(s.keys)}
    def window(s, t, ppm):
        a = np.searchsorted(s.mass, t*(1-ppm/1e6), 'left')
        b = np.searchsorted(s.mass, t*(1+ppm/1e6), 'right')
        return np.arange(a, b)
    def fps(s, idx):
        return np.unpackbits(np.asarray(s._fp[idx]), axis=1)[:, :s.nbits]

def build_pool():
    t0 = time.time()
    d = os.path.dirname(find('coco_fp.npy'))
    cm = pickle.load(open(d + '/coco_meta.pkl', 'rb'))
    co_fp = np.load(d + '/coco_fp.npy'); co_mass = np.load(d + '/coco_mass.npy')
    co_keys = np.asarray(cm['keys'], dtype=object); co_smis = np.asarray(cm['smiles'], dtype=object)
    print(f'COCONUT: {len(co_mass):,} structures', flush=True)

    # ChEBI + LIPID MAPS: small, high-precision, and covers mammalian/lipid metabolites that a
    # plant/microbe-focused NP database misses. +8.8% candidates for +7-19% coverage of the
    # structures in the public spectral libraries (see the pool section).
    if CFG.USE_BIO_DB:
        try:
            bd = os.path.dirname(find('bio_fp.npy'))
            bm = pickle.load(open(bd + '/bio_meta.pkl', 'rb'))
            bi_fp = np.load(bd + '/bio_fp.npy'); bi_mass = np.load(bd + '/bio_mass.npy')
            co_fp = np.vstack([co_fp, bi_fp]); co_mass = np.concatenate([co_mass, bi_mass])
            co_keys = np.concatenate([co_keys, np.asarray(bm['keys'], dtype=object)])
            co_smis = np.concatenate([co_smis, np.asarray(bm['smiles'], dtype=object)])
            print(f'+ ChEBI/LIPID MAPS: {len(bi_mass):,} structures', flush=True)
        except FileNotFoundError:
            print('ChEBI/LIPID MAPS dataset not attached - skipping', flush=True)

    tr = pq.read_table(TRAIN, columns=['inchikey14', 'normalized_smiles']).to_pandas()
    tr = tr.dropna().drop_duplicates('inchikey14')
    tr = tr[~tr.inchikey14.isin(set(co_keys))]
    print(f'training structures to fingerprint: {len(tr):,}  (~5 min)', flush=True)
    with MPool(4) as mp:
        res = mp.map(fp_and_mass, list(tr.normalized_smiles), chunksize=500)
    ok = [i for i, r in enumerate(res) if r is not None]
    tr_fp = np.packbits(np.stack([res[i][0] for i in ok]), axis=1)
    tr_mass = np.array([res[i][1] for i in ok])
    tr_keys = tr.inchikey14.values[ok]; tr_smi = tr.normalized_smiles.values[ok]

    fp = np.vstack([co_fp, tr_fp])
    mass = np.concatenate([co_mass, tr_mass])
    keys = np.concatenate([co_keys, tr_keys])
    smis = np.concatenate([co_smis, tr_smi])
    good = np.isfinite(mass)
    print(f'pool: {int(good.sum()):,} structures   ({time.time()-t0:.0f}s)', flush=True)
    return Pool(fp[good], mass[good], keys[good], smis[good], cm['nbits'])
