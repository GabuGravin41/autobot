"""
Empirical NLP Parser & Concordance Benchmark on RSNA Knee Gold-Standard Reports.
Evaluates weak supervision extraction across Spanish, English, and Dutch radiology reports
against the 58 ground-truth annotated cases in train.csv.
"""

import re
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

TARGETS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus',
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion',
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

# Multilingual Negation prefixes across Spanish, English, Dutch
NEGATIONS = [
    r'sin\s+(?:signos|evidencia|datos)\s+de',
    r'no\s+(?:se\s+observa|se\s+aprecia|se\s+identifica|hay|presenta)',
    r'dentro\s+de\s+l[ií]mites\s+normales',
    r'de\s+caracter[ií]sticas\s+normales',
    r'conservad[oa]s?',
    r'intact[oa]s?',
    r'ausencia\s+de',
    r'no\s+fractura',
    r'no\s+derrame',
    r'no\s+rotura',
    r'without\s+(?:evidence|signs)\s+of',
    r'no\s+(?:evidence\s+of|focal|significant|surfacing)',
    r'normal\s+in\s+appearance',
    r'unremarkable',
    r'is\s+intact',
    r'are\s+intact',
    r'not\s+torn',
    r'zonder\s+(?:tekens|significant|duidelijke)',
    r'intact',
    r'geen\s+(?:aanwijzingen|ruptuur|scheur|botoedeem|hydrops)'
]

NEG_REGEX = re.compile(r'|'.join(NEGATIONS), re.IGNORECASE)

def parse_report_for_pathologies(text: str) -> dict:
    if not isinstance(text, str) or not text.strip():
        return {t: 0.0 for t in TARGETS}
    
    # Split text into sentences/clauses
    sentences = re.split(r'[\.\n\r;]+', text)
    findings = {t: 0.0 for t in TARGETS}
    
    # Check global mentions like tricompartmental OA
    full_lower = text.lower()
    if 'tricompartmental' in full_lower or 'tricompartimental' in full_lower:
        findings['Medial OA'] = 1.0
        findings['Lateral OA'] = 1.0
        findings['PF OA'] = 1.0
        
    if 'gonartrose' in full_lower or 'gonartrosis' in full_lower:
        findings['Medial OA'] = 1.0
        findings['PF OA'] = 1.0

    for sent in sentences:
        s = sent.strip().lower()
        if not s:
            continue
            
        is_negated = bool(NEG_REGEX.search(s))
        
        # 1. ACL (Spanish, English, Dutch)
        if re.search(r'\b(?:cruzado\s+anterior|lca|acl|anterior\s+cruciate|voorste\s+kruisband|vka)\b', s):
            if re.search(r'\b(?:rotura|desgarro|rupture|tear|tearing|avulsi[oó]n|lesi[oó]n|interrupci[oó]n|scheur|degeneratie|ruptuur|disrupted)\b', s):
                if not is_negated:
                    findings['ACL'] = 1.0
                    
        # 2. MCL
        if re.search(r'\b(?:colateral\s+medial|lcm|mcl|medial\s+collateral|colateral\s+interno|mediale\s+collaterale|mcl-complex)\b', s):
            if re.search(r'\b(?:rotura|desgarro|rupture|tear|lesi[oó]n|esguince|scheur|sprain|thickened)\b', s):
                if not is_negated:
                    findings['MCL'] = 1.0

        # 3. Medial Meniscus
        if re.search(r'\b(?:menisco\s+(?:medial|interno)|medial\s+meniscus|mediale\s+meniscus|cuerno\s+posterior.*menisco\s+medial)\b', s):
            if re.search(r'\b(?:rotura|desgarro|rupture|tear|tearing|amputaci[oó]n|scheur|ruptuur|extrusion|extrusie|radial|horizontal|root)\b', s):
                if not is_negated and not re.search(r'without\s+surfacing\s+tear|geen\s+scheur|sin\s+rotura', s):
                    findings['Medial Meniscus'] = 1.0

        # 4. Lateral Meniscus
        if re.search(r'\b(?:menisco\s+(?:lateral|externo)|lateral\s+meniscus|laterale\s+meniscus|cuerno.*menisco\s+lateral)\b', s):
            if re.search(r'\b(?:rotura|desgarro|rupture|tear|tearing|amputaci[oó]n|scheur|ruptuur|radial|horizontal)\b', s):
                if not is_negated and not re.search(r'without\s+surfacing\s+tear|geen\s+scheur|sin\s+rotura', s):
                    findings['Lateral Meniscus'] = 1.0

        # 5. Medial OA
        if re.search(r'\b(?:medial\s+compartment|compartimento\s+femorotibial\s+medial|mediaal\s+femorotibiaal|medial\s+femoral\s+condyle|medial\s+tibial\s+plateau|c[oó]ndilo\s+femoral\s+medial)\b', s):
            if re.search(r'\b(?:cartilage\s+loss|cartilage\s+fissuring|chondral|condropat[ií]a|artrosis|osteoarthritis|kraakbeenlijden|kraakbeenverlies|spurring|osteofito|pinzamiento|thinning)\b', s):
                if not is_negated:
                    findings['Medial OA'] = 1.0
        elif re.search(r'\b(?:artrosis|osteoartritis|osteoartrosis|condropat[ií]a|desgaste|adelgazamiento\s+condral)\b', s):
            if re.search(r'\b(?:medial|interno)\b', s) and not re.search(r'\btr[oó]clea|patel[ao]|r[oó]tula\b', s):
                if not is_negated:
                    findings['Medial OA'] = 1.0

        # 6. Lateral OA
        if re.search(r'\b(?:lateral\s+compartment|compartimento\s+femorotibial\s+lateral|lateraal\s+femorotibiaal|lateral\s+femoral\s+condyle|lateral\s+tibial\s+plateau|c[oó]ndilo\s+femoral\s+lateral)\b', s):
            if re.search(r'\b(?:cartilage\s+loss|cartilage\s+fissuring|chondral|condropat[ií]a|artrosis|osteoarthritis|kraakbeenlijden|kraakbeenverlies|spurring|thinning)\b', s):
                if not is_negated:
                    findings['Lateral OA'] = 1.0
        elif re.search(r'\b(?:artrosis|osteoartritis|osteoartrosis|condropat[ií]a|desgaste)\b', s):
            if re.search(r'\b(?:lateral|externo)\b', s) and not re.search(r'\btr[oó]clea|patel[ao]|r[oó]tula\b', s):
                if not is_negated:
                    findings['Lateral OA'] = 1.0

        # 7. PF OA
        if re.search(r'\b(?:patellofemoral|patelofemoral|femororrotuliana|femoropatelar|tr[oó]clea|trochlea|r[oó]tula|rotulian[ao]|patellar|patella)\b', s):
            if re.search(r'\b(?:artrosis|osteoartritis|condropat[ií]a|adelgazamiento|lesi[oó]n\s+condral|[uú]lcera\s+condral|cartilage\s+loss|cartilage\s+fissuring|chondromalacia|defect|osteochondral|thinning)\b', s):
                if not is_negated and not re.search(r'without\s+(?:cartilage|defect)|sin\s+alteraciones', s):
                    findings['PF OA'] = 1.0

        # 8. Effusion
        if re.search(r'\b(?:derrame|l[ií]quido\s+articular|effusion|joint\s+effusion|hydrops|vocht\s+in\s+het\s+gewricht)\b', s):
            eff_neg = re.search(r'\bno\s+(?:significant\s+|focal\s+|knee\s+|joint\s+)?effusion|no\s+evidence\s+of\s+.*effusion|without\s+.*effusion|sin\s+(?:signos\s+de\s+)?derrame|no\s+(?:hay|se\s+observa|se\s+aprecia)\s+derrame|geen\s+hydrops\b', s)
            if not is_negated and not eff_neg:
                findings['Effusion'] = 1.0

        # 9. Synovitis
        if re.search(r'\b(?:sinovitis|engrosamiento\s+sinovial|hipertrofia\s+sinovial|synovitis|proliferaci[oó]n\s+sinovial|thickened\s+synovial|synovium\s+thickening|verdikkingen.*synovium)\b', s):
            if not is_negated:
                findings['Synovitis'] = 1.0

        # 10. Baker's
        if re.search(r'\b(?:quiste\s+(?:de\s+baker|popl[ií]teo)|baker(?:\'s)?\s+cyst|popliteal\s+cyst|popliteale\s+cyste)\b', s):
            if not is_negated and not re.search(r'\bno\s+(?:hay|se\s+observan)\s+quistes|geen\s+popliteale\s+cyste|no\s+popliteal\s+cyst\b', s):
                findings['Baker\'s'] = 1.0

        # 11. Contusion
        if re.search(r'\b(?:edema\s+[oó]seo|contusi[oó]n\s+[oó]sea|bone\s+marrow\s+edema|bone\s+bruise|contusion|marrow\s+edema|botoedeem)\b', s):
            if not is_negated and not re.search(r'\bno\s+(?:bone|edema)|sin\s+edema|zonder\s+significant\s+botoedeem\b', s):
                findings['Contusion'] = 1.0

        # 12. Fracture
        if re.search(r'\b(?:fractura|fracture|hundimiento|desprendimiento\s+[oó]seo|insufficiency\s+fracture|subchondral\s+fracture|osteochondral\s+fracture)\b', s):
            if not is_negated and not re.search(r'\bno\s+fractura|sin\s+fractura|no\s+fracture\b', s):
                findings['Fracture'] = 1.0

    return findings

def evaluate():
    df = pd.read_csv('competitions/rsna_knee/train.csv')
    labeled = df.dropna(subset=TARGETS, how='any').copy()
    print(f"Loaded {len(labeled)} gold-standard labeled studies with clinical reports.")
    
    parsed_records = []
    for idx, row in labeled.iterrows():
        preds = parse_report_for_pathologies(row['Report'])
        preds['StudyInstanceUID'] = row['StudyInstanceUID']
        parsed_records.append(preds)
        
    pred_df = pd.DataFrame(parsed_records).set_index('StudyInstanceUID')
    gt_df = labeled.set_index('StudyInstanceUID')[TARGETS]
    
    print("\n" + "="*70)
    print("CONCORDANCE BENCHMARK: NLP RULE EXTRACTOR vs. GOLD-STANDARD RADIOLOGIST LABELS")
    print("="*70)
    print(f"{'Target':<18} | {'Pos GT':<6} | {'Pos Pred':<8} | {'Precision':<10} | {'Recall':<8} | {'F1':<6} | {'ROC-AUC':<8}")
    print("-"*70)
    
    f1_list, auc_list = [], []
    for target in TARGETS:
        y_true = gt_df[target].to_numpy().astype(int)
        y_pred = pred_df[target].to_numpy()
        
        pos_gt = y_true.sum()
        pos_pred = int(y_pred.sum())
        
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 1.0
        except Exception:
            auc = 0.5
            
        f1_list.append(f1)
        auc_list.append(auc)
        print(f"{target:<18} | {pos_gt:<6} | {pos_pred:<8} | {prec:<10.3f} | {rec:<8.3f} | {f1:<6.3f} | {auc:<8.3f}")
        
    print("-"*70)
    print(f"Macro F1 Score   : {np.mean(f1_list):.4f}")
    print(f"Macro ROC-AUC    : {np.mean(auc_list):.4f}")
    print("="*70)

if __name__ == '__main__':
    evaluate()
