
# ===================================================================================
#  CONFIG — every knob in one place. Each is annotated with the measurement behind it.
# ===================================================================================
class CFG:
    # --- candidate generation -------------------------------------------------------
    PPM_WIN      = 10.0   # neutral-mass window for candidates. Tighter IS better -- up to the
                          # point where it starts deleting answers, which is the same trap as
                          # CAND_CAP below. Check window RECALL, not just MRR:
                          #   5.0 ppm 0.972 | 7.0 0.992 | 8.5 0.992 | 10.0 1.000 | 12+ 1.000
                          # 10 ppm is the smallest window that loses nothing, and everything
                          # wider only adds decoys. Full four-channel ranker agrees:
                          #   8.5 -> predLB 0.361 | 10.0 -> 0.371 | 12.0 -> 0.363
                          # (Much wider genuinely does hurt: +-20 -> 0.509, +-30 -> 0.500 C2 MRR.)
                          # timsTOF precursor error stays under ~9 ppm (+1.4 ppm offset).
    PPM_FALLBACK = 30.0   # only used if the tight window returns nothing at all.

    # --- spectrum cleaning ----------------------------------------------------------
    INT_FLOOR    = 0.002  # drop peaks below this fraction of the base peak
    MAX_PEAKS    = 256    # keep the N most intense peaks after the floor
    MZ_TOL       = 0.01   # Da tolerance when matching two peaks
    INT_POWER    = 1.0    # intensity transform before similarity (1.0 + entropy weighting
                          # beat sqrt: Class-1 0.919 vs 0.895)
    ENT_WEIGHT   = True   # Li et al. 2021 entropy weighting of low-entropy spectra

    # --- analog propagation (the main idea) -----------------------------------------
    ANALOG_WIN   = 200.0  # +- Da mass-shift window. +-400 gave no gain (0.520 vs 0.521).
    N_ANALOG     = 200    # analogs kept per molecule. This used to be FLAT above 80 (predicted LB
                          # 0.3683/0.3709/0.3705/0.3703 at 60/80/100/140) and it is not any more.
                          # Once the representatives are instrument-matched, the extra analogs are
                          # real neighbours instead of cross-instrument noise, so keeping more
                          # pays. Analog-channel c2, re-swept on the new channel:
                          #    N=80  0.5412 | N=200 0.5510 | N=600 0.5536 | N=800 0.5595
                          # 200 takes most of the gain; the surface past it is noisy (300 and 450
                          # both dip below 200) and the tail is not worth chasing on 250 queries.
                          # A parameter's optimum moves when the channel beneath it changes -- the
                          # same lesson as the class prior after the model channel improved.
    SIM_POWER    = 3.0    # sim^p weighting. p=1 -> 0.498, p=3 -> 0.521, p=4 -> 0.525 on local
                          # validation -- but see the model-selection section: that validation set
                          # is the one the checkpoint was early-stopped on, so small local wins on
                          # it are not trustworthy. p=3 is the value that actually scored 0.335.

    # --- ranker ---------------------------------------------------------------------
    W1_PRIORS    = (0.30, 0.60)  # REVERTED. A narrow plateau (.40,.45,.50) scored 0.3789 in my
                          # own sweep and then LOST on the leaderboard (0.335 -> 0.330). The sweep
                          # was in-sample: it trained on all 819 query groups and evaluated on 250
                          # of them. See the ranker section -- hold out BY QUERY or you are just
                          # measuring capacity to memorise your own evaluation set.
    SEEDS        = (0, 1, 2, 3)  # seed alone moves the LB by ~0.006; bag several.
    W1           = 0.50   # weight on the Class-1 simulation. NOT the class share (0.16) --
                          # it is the leaderboard-calibrated value, chosen by 5-fold CV held out
                          # by query in make_ranker3.py.
    GBM = dict(max_depth=6, max_iter=500, learning_rate=0.03,
               min_samples_leaf=80, l2_regularization=1.0)
               # Depth 6. Two sweeps said otherwise and both were wrong on the leaderboard:
               #   in-sample sweep        -> preferred (.40,.45,.50) d10, scored 0.330 vs 0.335
               #   query-held-out 5-fold  -> preferred (.30,.60)  d10, scored 0.321 vs 0.335
               # The second one is a correct CV and it still did not transfer, because the 250
               # held-out queries are library natural products and the test is timsTOF molecules
               # with no reference spectra. Extra capacity fits the local distribution instead.
               # Local numbers here are a filter for broken things, not a way to choose between
               # working configurations. See the ranker section.
    USE_BIO_DB   = False  # (AP-2: dataset detached -- CC BY-NC-SA, and it lost on the LB.)
                          # add ChEBI + LIPID MAPS. Costs -0.026 Class-2 MRR in dilution but
                          # adds 7-19% coverage of in-library structures. Looked net-positive
                          # on validation but the leaderboard disagreed (0.299 -> 0.295), so it
                          # ships OFF. Flip it if your pool recall differs.
                          # (Adding all of PubChem instead costs -0.35: measured, do not.)
    CAND_CAP     = 500    # pure runtime guard on in-silico fragmentation (~14 ms/candidate).
                          # It is deliberately LARGE. A cap of 80 ranked by "library hit, then
                          # closest in mass" looks like an adaptive version of "tighter windows
                          # win" -- it is not, it is a recall bug. Class-2 answers have
                          # library_sim = 0 *by definition* (no reference spectrum exists), so
                          # they get ordered by mass alone, which inside a +-8.5 ppm window is
                          # arbitrary. Measured truth retention on the Class-2 holdout:
                          #   no cap 0.992 | cap 400 0.992 | cap 200 0.952 | cap 80 0.752
                          # i.e. a cap of 80 throws away a QUARTER of the reachable answers.
                          # Class 1 is untouched (0.992 at every cap) because lv*100 protects it,
                          # which is exactly why the bug survives casual validation.
                          # Median window is 52 candidates, max 401, so 500 essentially never
                          # fires -- and when it does it ranks by the model, not by mass.
    LIB_MASS_FORMULA = True   # index the library on molecular_formula mass (exact arithmetic)
                          # instead of precursor+adduct. Class-1 channel 0.9206 -> 0.9253 in a
                          # TIGHTER window; immune to riken's 0.005 Da precision and gnps's wrong
                          # adduct labels. See "Two fixes from the community".
    ANALOG_MATCH_INSTR = True   # pick analog representatives from the test's own instrument.
                          # Analog channel 0.5241 -> 0.5412. Set either to False to A/B it.
    PC_TOPK      = 0      # PubChem pool expansion, OFF. Admitting the model's top-50 isomers
                          # scored 0.205 against the 0.335 it replaced -- worse than my worst
                          # modelled case. The recall it buys is ~zero (the pool already holds the
                          # Class-2 answers) and the real dilution is about twice what the holdout
                          # predicted. Set it to 50 to reproduce the experiment; see the section
                          # "Deciding a bet your validation set cannot score" for the full autopsy.
    PC_WINCAP    = 10000  # isomers fingerprinted per query when PC_TOPK > 0. Presence of the true
                          # structure: 2k -> 0.788 | 5k -> 0.884 | 12k -> 0.940 | all -> 0.948.
    TOPN         = 25     # the metric allows 25 guesses; there is no penalty for using them all

    # --- AP-2: stages that act on the ranker's ORDERING, after scoring --------------
    # One switch per stage. All five False = the V32 list, byte for byte (checked locally on the
    # visible test). The section "AP-2: what happens after the ranker" explains each one.
    DEDUP_METRIC = True   # Stage 0. One slot per answer the official scorer can tell apart
                          # (heavy-atom composition + tautomer-canonical InChIKey14), refilled
                          # from the same complete ordering. Can only move a guess up. V32 wasted
                          # 3 slots this way on the 400 visible molecules.
    USE_POP_PRIOR = False # Stage 2. Re-order POOL candidates in the top POP_TOP by
                          # z(ranker probability) + POP_MU * pop, pop = log1p(PubChem substances)
                          # + log1p(PubMed). Needs the popularity table. Cannot be validated
                          # offline: it goes to the leaderboard with replicates.
    PC_TIER      = False  # Stage 3. PubChem structures that are not pool candidates enter ONLY
                          # through fixed slots, never through the ranker. Needs the PubChem
                          # mass index (and the popularity table for its ordering).
    USE_FORWARD  = False  # Stage 1 HOOK. forward_scores() returns None in this version.
    USE_SECOND_LIST = False  # Stage 5 HOOK. second_list() returns None in this version.

    LIB_GATE     = 0.90   # every stage above except 0 leaves a molecule alone when its best
                          # library match (entropy similarity) is >= this: a near-identical
                          # reference spectrum beats any prior or simulated spectrum.
    POP_MU       = 0.20   # public leaderboard evidence (other authors): 0.15 and 0.25 both helped
    POP_TOP      = 60     # the re-ordered head; also the head the Stage 1 hook would see
    PC_TIER_N1   = 5000   # pass-1 survivors (Morgan-only partial f.z) per molecule
    PC_TIER_NDOC = 200    # ... plus this many most-documented structures of the window
    PC_TIER_POP_W = 0.25  # tier order: z(f.z) + PC_TIER_POP_W * pop
    PC_TIER_K    = 25     # tier length kept (only len(PC_SLOTS) of them can reach the top 25)
    PC_TIER_WINCAP = 60000  # even-sample cap on a window, a runtime guard only (visible-test
                          # windows: median 12,282, max 39,434 structures, so it never fires there)
    PC_SLOTS     = (4, 8, 12, 16, 20)   # default positions for tier structures
    PC_SLOTS_STRONG = (2, 4, 6, 8, 10)  # used when the tier's best f.z beats the pool's best
    PC_MARGIN    = 300.0  # ... by more than this, in our model's f.z units. Set on the clean partition so
                          # that a pool-held answer triggers the strong slots for <= 5% of molecules
                          # (8% in the natural-product stratum); see Stage 3.
    PC_TIER_DEADLINE_S = 6.5 * 3600  # failsafe: no new tier after this much notebook time (the visible
                          # test never opens the gate, so a commit run cannot show the tier's cost)
    FWD_LAMBDA   = 1.0    # Stage 1 hook weight on z(forward score) inside a formula group
    RRF_K, RRF_W2, RRF_TOP = 3.0, 0.6, 40   # Stage 5 hook: 1/(K+r1) + W2/(K+r2) over the top RRF_TOP

# Frozen composition settings. Each feature can be disabled independently.
# All four off retains the matched S0 metric deduplication control.
COMPOSITION_SWITCHES = {'precursor_union': False, 'popularity': True,
                       'pubchem_tier': True, 'forward': True}
CFG.USE_PRECURSOR_UNION = COMPOSITION_SWITCHES['precursor_union']
CFG.USE_POP_PRIOR = COMPOSITION_SWITCHES['popularity']
CFG.PC_TIER = COMPOSITION_SWITCHES['pubchem_tier']
CFG.USE_FORWARD = COMPOSITION_SWITCHES['forward']
CFG.DEDUP_METRIC = True
CFG.USE_SECOND_LIST = False
CFG.PC_TOPK = 0
CFG.FWD_WEIGHTS = {'iceberg': 0.0, 'glacier': 1.0}  # F1 2026-10-08: GLACIER only ([M+H]+ in the runner); ICEBERG weight 0
CFG.FWD_BUDGET_S = 9400
assert CFG.PPM_WIN == 10.0 and CFG.PPM_FALLBACK == 30.0
assert CFG.CAND_CAP == 500 and CFG.LIB_GATE == 0.90
