
# ===================================================================================
#  Adduct -> neutral mass.  The instrument measures the *ion*; candidates are neutral
#  molecules, so every adduct has to be undone before we can compare masses.
# ===================================================================================

MASS = dict(C=12.0,H=1.00782503207,N=14.0030740048,O=15.9949146196,P=30.97376163,
            S=31.97207100,F=18.99840322,Cl=34.96885268,Br=78.9183371,I=126.904473,
            Na=22.9897692809,K=38.96370668,Si=27.9769265325,B=11.0093054,Se=79.9165213)
E=0.00054857990; PROTON=MASS['H']-E; H2O=2*MASS['H']+MASS['O']
NH4=MASS['N']+4*MASS['H']; FORMATE=MASS['C']+2*MASS['H']+2*MASS['O']
ACETATE=2*MASS['C']+4*MASS['H']+2*MASS['O']
ADDUCTS = {
 "[M+H]+":(1,1,PROTON), "[M+NH4]+":(1,1,NH4-E), "[M+Na]+":(1,1,MASS['Na']-E),
 "[M+K]+":(1,1,MASS['K']-E), "[M-H2O+H]+":(1,1,PROTON-H2O), "[M-2H2O+H]+":(1,1,PROTON-2*H2O),
 "[M+2H]2+":(1,2,2*PROTON), "[M]+":(1,1,-E), "[M-H2O]+":(1,1,-E-H2O),
 "[M+CH3OH+H]+":(1,1,PROTON+MASS['C']+4*MASS['H']+MASS['O']),
 "[M+CH3CN+H]+":(1,1,PROTON+2*MASS['C']+3*MASS['H']+MASS['N']),
 "[M-H]-":(1,1,-PROTON), "[M-H2O-H]-":(1,1,-PROTON-H2O), "[M+CH2O2-H]-":(1,1,FORMATE-PROTON),
 "[M+C2H4O2-H]-":(1,1,ACETATE-PROTON), "[M+Cl]-":(1,1,MASS['Cl']+E), "[M]-":(1,1,E),
 "[M-2H]-":(1,2,-2*PROTON), "[M+Na-2H]-":(1,1,MASS['Na']-2*PROTON),
 "[2M+H]+":(2,1,PROTON), "[2M+Na]+":(2,1,MASS['Na']-E), "[2M+NH4]+":(2,1,NH4-E),
 "[2M+K]+":(2,1,MASS['K']-E), "[2M-H]-":(2,1,-PROTON), "[2M+CH2O2-H]-":(2,1,FORMATE-PROTON),
 "[2M+C2H4O2-H]-":(2,1,ACETATE-PROTON), "[2M+Na-2H]-":(2,1,MASS['Na']-2*PROTON),
 "[3M+H]+":(3,1,PROTON), "[3M-H]-":(3,1,-PROTON),
}
MONO = dict(C=12.0, H=1.00782503207, N=14.0030740048, O=15.9949146196, S=31.97207100,
            P=30.97376163, Cl=34.96885268, Br=78.9183371, F=18.99840322, I=126.904473,
            Si=27.9769265325, Se=79.9165213, Na=22.9897692809, K=38.96370668, B=11.0093054,
            As=74.9215965, Fe=55.9349375, D=2.0141017778)
_TOK = re.compile(r'([A-Z][a-z]?)(\d*)')
def formula_mass(f):
    """Neutral monoisotopic mass by ARITHMETIC on the molecular_formula column.

    The library index used to derive each reference spectrum's neutral mass from
    precursor_mz + adduct. That inherits two errors the formula does not have:
      * reported precursor precision -- riken is accurate to ~0.005 Da, which is a fine
        ABSOLUTE error and 17 ppm at 300 Da, so a relative window drops it at the light end
        (121,805 training spectra fall outside +-10 ppm: credit @dariushafshar);
      * wrong adduct annotations -- gnps misses cluster at discrete offsets of -1.008 Da
        (one hydrogen) and +21.982 Da (sodium vs proton).
    Formula mass is immune to both, and it makes the index TIGHTER rather than looser.
    Measured on the Class-2 holdout: Class-1 channel 0.9206 -> 0.9253 while inspecting fewer
    library spectra per query (571 -> 663 vs 914 for a widened ppm window), and the analog
    channel 0.5213 -> 0.5241.  Falls back to the precursor derivation when the formula is
    missing or contains an element not tabulated above.
    """
    if not isinstance(f, str) or not f: return np.nan
    m = 0.0
    for el, n in _TOK.findall(f):
        if not el: continue
        if el not in MONO: return np.nan      # refuse, do not guess
        m += MONO[el] * (int(n) if n else 1)
    return m

def neutral_mass(mz, adduct):
    out=np.full(len(mz), np.nan); ad=np.asarray(adduct, dtype=object)
    for a,(n,z,d) in ADDUCTS.items():
        m=(ad==a)
        if m.any(): out[m]=(mz[m]*z-d)/n
    return out


# ===================================================================================
#  Library + candidate pool
# ===================================================================================
def load_library(path):
    t0 = time.time()
    t = pq.read_table(path, columns=['inchikey14','normalized_smiles','adduct','precursor_mz',
                                     'molecular_formula','instrument_type',
                                     'ms2_mzs','ms2_normalized_intensities'])
    mzc = t.column('ms2_mzs').combine_chunks(); itc = t.column('ms2_normalized_intensities').combine_chunks()
    off = mzc.offsets.to_numpy().astype(np.int64)
    allmz = mzc.values.to_numpy(zero_copy_only=False).astype(np.float32)
    allin = itc.values.to_numpy(zero_copy_only=False).astype(np.float32)
    prec = t.column('precursor_mz').to_numpy(zero_copy_only=False).astype(np.float64)
    add = np.asarray(t.column('adduct').cast(pa.string()).to_pylist(), dtype=object)
    ik  = np.asarray(t.column('inchikey14').cast(pa.string()).to_pylist(), dtype=object)
    smi = np.asarray(t.column('normalized_smiles').cast(pa.string()).to_pylist(), dtype=object)
    # Index on the FORMULA mass (exact arithmetic), falling back to the precursor derivation
    # only where the formula is missing or exotic. See formula_mass() for why this matters.
    fo  = np.asarray(t.column('molecular_formula').cast(pa.string()).to_pylist(), dtype=object)
    _fm = {f: formula_mass(f) for f in set(x for x in fo if isinstance(x, str))}
    nmf = np.array([_fm.get(f, np.nan) for f in fo])
    nmp = neutral_mass(prec, add)
    nm  = np.where(np.isfinite(nmf), nmf, nmp) if CFG.LIB_MASS_FORMULA else nmp
    print(f'  library mass index: formula available for {np.isfinite(nmf).mean():.1%} of spectra, in use: {CFG.LIB_MASS_FORMULA}, '
          f'{np.mean(np.isfinite(nmp) & ~np.isfinite(nmf)):.1%} from precursor+adduct')
    ok = np.isfinite(nm)
    order = np.argsort(np.where(ok, nm, 1e18), kind='mergesort')
    best = {}
    for k, s in zip(ik, smi):
        if k and s and k not in best: best[k] = s
    print(f'library: {len(off)-1:,} spectra / {len(best):,} structures  ({time.time()-t0:.0f}s)', flush=True)
    ins = np.asarray(t.column('instrument_type').cast(pa.string()).to_pylist(), dtype=object)
    return dict(off=off, mz=allmz, it=allin, nm=nm, ik=ik, best=best, ins=ins,
                order=order, snm=nm[order], n_ok=int(ok.sum()))

def lib_window(L, target, tol):
    lo = np.searchsorted(L['snm'][:L['n_ok']], target-tol, 'left')
    hi = np.searchsorted(L['snm'][:L['n_ok']], target+tol, 'right')
    return L['order'][lo:hi]

def build_rep(L, prefer_instrument=None):
    """One representative spectrum per structure, sorted by neutral mass.

    Richest-spectrum-wins was the old rule, and it quietly propagated timsTOF queries from
    Orbitrap neighbours. Analog propagation compares a *shifted* neighbour spectrum peak by
    peak, so the neighbour's instrument decides which fragments are even present and at what
    relative intensity -- matching it is not cosmetic. The whole test set is one instrument
    (timsTOF), and the library holds 1.15M timsTOF spectra, so a same-instrument
    representative usually exists.

        richest spectrum (old)        c2 0.5241
        prefer any TOF/QTOF           c2 0.5311
        prefer the exact instrument   c2 0.5412      <- shipped

    Note the exact match is worth more than double the loose one: 'QTOF' is not 'timsTOF'.
    The preferred instrument is read off the test set at runtime rather than hardcoded, so
    this adapts if the hidden set is mixed. Using 3 representatives per structure is still
    WORSE (0.49 vs 0.52) -- extra spectra raise the max similarity of irrelevant structures
    too, which flattens the discrimination.
    """
    npk = np.diff(L['off']); best = {}; ik = L['ik']
    ins = L.get('ins')
    pref = None
    if prefer_instrument and ins is not None:
        pref = str(prefer_instrument).lower().replace('-', '').replace(' ', '')
        match = np.array([isinstance(x, str) and
                          x.lower().replace('-', '').replace(' ', '') == pref for x in ins])
        print(f'  analog representatives: preferring instrument {prefer_instrument!r} '
              f'({match.mean():.1%} of library spectra)')
    for i in range(len(ik)):
        k = ik[i]
        if not k: continue
        if k not in best: best[k] = i; continue
        cur = best[k]
        if pref is not None and match[i] != match[cur]:
            if match[i]: best[k] = i          # same instrument beats richer
            continue
        if npk[i] > npk[cur]: best[k] = i
    rep = np.array(sorted(best.values()))
    nm = L['nm'][rep]; ok = np.isfinite(nm)
    rep = rep[ok]; nm = nm[ok]; key = ik[rep]
    o = np.argsort(nm)
    return rep[o], key[o], nm[o]

# ===================================================================================
#  The two evidence channels
# ===================================================================================
def clean(mz, it):
    return _clean(np.asarray(mz, np.float32), np.asarray(it, np.float32),
                  CFG.INT_FLOOR, CFG.MAX_PEAKS, CFG.INT_POWER, CFG.ENT_WEIGHT)

def lib_sim(L, specs, target):
    """CLASS 1: direct match against library spectra of the same neutral mass."""
    cand = lib_window(L, target, target*CFG.PPM_WIN/1e6)
    if len(cand) == 0: return {}
    agg = {}
    for mz, it in specs:
        qm, qp = clean(mz, it)
        if len(qm) == 0: continue
        sc = search(qm, qp, cand, L['off'], L['mz'], L['it'],
                    CFG.MZ_TOL, CFG.INT_FLOOR, CFG.MAX_PEAKS, CFG.INT_POWER, CFG.ENT_WEIGHT, 1)
        for c, s in zip(cand, sc):
            k = L['ik'][c]
            if s > agg.get(k, -1.0): agg[k] = float(s)
    return agg

def analog_sim(L, specs, target, rep, rep_key, rep_nm):
    """CLASS 2: mass-SHIFTED match over a wide window. Relatives of the unknown fragment
       into the same ions offset by the mass difference, so they still match."""
    lo = np.searchsorted(rep_nm, target-CFG.ANALOG_WIN, 'left')
    hi = np.searchsorted(rep_nm, target+CFG.ANALOG_WIN, 'right')
    cand = rep[lo:hi]
    if len(cand) == 0: return []
    shift = (target - rep_nm[lo:hi]).astype(np.float32)
    ckey = rep_key[lo:hi]; agg = {}
    for mz, it in specs:
        qm, qp = clean(mz, it)
        if len(qm) == 0: continue
        sc = search_shift(qm, qp, cand, L['off'], L['mz'], L['it'],
                          CFG.MZ_TOL, CFG.INT_FLOOR, CFG.MAX_PEAKS,
                          CFG.INT_POWER, CFG.ENT_WEIGHT, 1, shift)
        for c, k, s in zip(cand, ckey, sc):
            if s > agg.get(k, -1.0): agg[k] = float(s)
    return sorted(agg.items(), key=lambda x: -x[1])[:CFG.N_ANALOG]
