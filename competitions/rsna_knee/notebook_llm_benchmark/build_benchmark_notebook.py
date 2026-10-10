import json
import os
from pathlib import Path

def create_cell(cell_type, source):
    lines = [line + '\n' for line in source.split('\n')]
    if lines and lines[-1] == '\n':
        lines[-1] = ''
    return {
        "cell_type": cell_type,
        "metadata": {},
        "source": lines,
        **({"outputs": [], "execution_count": None} if cell_type == "code" else {})
    }

cells = []

# Cell 1: Markdown Overview
cells.append(create_cell("markdown", """# RSNA Knee: Clinical LLM Benchmark — MedGemma vs Qwen 2.5 (10 Unlabeled Studies)
### Empirical Comparison of Zero-Shot Clinical Extraction Fidelity for 12 Knee Abnormalities

---

## Objective
To train vision models on the full RSNA Knee dataset (~4,407 studies) instead of just the 58 human-annotated cases, we extract binary labels for all 12 abnormalities from free-text clinical radiology reports.

In this notebook, we benchmark two top-tier open LLMs on **10 unlabelled clinical reports** (outside the 58 human-labeled cases):
1. **MedGemma / Gemma-2** (`google/gemma-2-9b-it` or `google/medgemma-27b-it` / `BioMistral`)
2. **Qwen 2.5 14B-Instruct** (`Qwen/Qwen2.5-14B-Instruct`)

Both models are evaluated on the exact same 10 reports using an identical clinical prompt. We then display a **side-by-side comparison matrix** of the 120 extracted labels (10 studies × 12 targets) and compare their clinical reasoning.
"""))

# Cell 2: Setup & Environment
cells.append(create_cell("code", """import os
import re
import json
import torch
import numpy as np
import pandas as pd
import glob

# Ensure dependencies are available
import subprocess
try:
    import bitsandbytes
    import accelerate
except ImportError:
    print("Installing bitsandbytes and accelerate for 4-bit GPU inference...")
    subprocess.run(["pip", "install", "-q", "bitsandbytes>=0.43.0", "accelerate"], check=True)
    import bitsandbytes
    import accelerate
    print("Dependencies installed successfully.")

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA Available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device:      {torch.cuda.get_device_name(0)}")
    print(f"GPU VRAM:        {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

# 12 RSNA Abnormality Target Classes
TARGETS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 
    'Medial OA', 'Lateral OA', 'PF OA', 'Effusion', 
    'Synovitis', "Baker's", 'Contusion', 'Fracture'
]

# Locate train.csv
candidates = [
    '/kaggle/input/rsna-knee-abnormality-detection/train.csv',
    '/kaggle/input/competitions/rsna-knee-abnormality-detection/train.csv',
    'competitions/rsna_knee/train.csv',
    'train.csv'
]
train_csv_path = None
for c in candidates:
    if os.path.exists(c):
        train_csv_path = c
        break

if not train_csv_path:
    found = glob.glob('/kaggle/input/**/train.csv', recursive=True)
    if found:
        train_csv_path = found[0]
    else:
        raise FileNotFoundError("Could not find train.csv. Please attach the RSNA competition dataset.")

print(f"Found train.csv at: {train_csv_path}")
df = pd.read_csv(train_csv_path)
print(f"Total dataset size: {len(df):,} rows")
"""))

# Cell 3: Select 10 Unlabelled Reports
cells.append(create_cell("markdown", """## 1. Selection of 10 Unlabelled Studies
We filter out the 58 human-labeled cases (`ACL.notna()`) and select 10 diverse studies containing clinical reports (`Report.notna()`).
"""))

cells.append(create_cell("code", """# Filter for unlabelled cases with valid reports
unlabeled_df = df[df['ACL'].isna() & df['Report'].notna()].copy()
print(f"Total unlabelled studies with reports: {len(unlabeled_df):,}")

# Select 10 diverse studies (fixed random seed for exact reproducibility)
sample_10 = unlabeled_df.sample(n=10, random_state=42).reset_index(drop=True)

print(f"\\nSelected 10 Studies for Benchmark Evaluation:\\n" + "-"*60)
for idx, row in sample_10.iterrows():
    report_preview = row['Report'].replace('\\n', ' ')[:120]
    print(f"Study [{idx+1}/10] UID: {row['StudyInstanceUID']}")
    print(f"  Length: {len(row['Report']):,} chars | Preview: {report_preview}...")
"""))

# Cell 4: Clinical Extraction Prompt Definition
cells.append(create_cell("markdown", """## 2. Standardized Clinical Extraction Prompt
We design a strict, zero-shot structured extraction prompt with comprehensive clinical definitions for all 12 abnormalities.
Both models will receive this exact prompt to ensure a fair, rigorous comparison.
"""))

prompt_lines = [
    'CLINICAL_PROMPT_TEMPLATE = """You are an expert musculoskeletal radiologist evaluating a knee MRI clinical report.',
    'Your task is to analyze the following report and determine the presence of 12 specific knee abnormalities.',
    'The report may be written in Spanish, Dutch, or English. Carefully evaluate clinical negations (e.g. "sin signos de rotura", "ligamentos conservados", "no se observa derrame", "gaaf", "geen afwijkingen") to ensure negative findings are marked as 0.',
    '',
    'Abnormality Definitions:',
    '1. ACL: Anterior cruciate ligament tear or disruption (complete or partial).',
    '2. MCL: Medial collateral ligament tear, sprain, or disruption.',
    '3. Medial Meniscus: Tear, fissure, maceration, or degenerative flap of the medial meniscus.',
    '4. Lateral Meniscus: Tear, fissure, maceration, or flap of the lateral meniscus.',
    '5. Medial OA: Medial compartment osteoarthritis, severe cartilage loss, joint space narrowing, or subchondral sclerosis.',
    '6. Lateral OA: Lateral compartment osteoarthritis, severe cartilage loss, joint space narrowing.',
    '7. PF OA: Patellofemoral osteoarthritis, chondromalacia patellae (high grade), or trochlear cartilage ulceration.',
    '8. Effusion: Joint effusion, significant intra-articular fluid (derrame articular, hydrops).',
    '9. Synovitis: Synovial thickening, enhancement, or inflammatory synovitis.',
    "10. Baker's: Baker's cyst, popliteal cyst, or gastrocnemius-semimembranosus bursal distension.",
    '11. Contusion: Bone contusion, trabecular bone bruise, or subchondral bone marrow edema.',
    '12. Fracture: Cortical bone fracture, tibial plateau fracture, or impaction fracture.',
    '',
    'Clinical Report:',
    '--- REPORT START ---',
    '{report_text}',
    '--- REPORT END ---',
    '',
    'Output Instructions:',
    'Return ONLY a valid JSON object with two fields:',
    '1. "labels": A dictionary mapping each of the 12 target names to either 1 (positive finding present) or 0 (normal, absent, or explicitly ruled out).',
    '2. "rationale": A brief 1-line medical explanation for each positive finding citing the report\'s text.',
    '',
    'Example format:',
    '{',
    '  "labels": {',
    '    "ACL": 0,',
    '    "MCL": 0,',
    '    "Medial Meniscus": 1,',
    '    "Lateral Meniscus": 0,',
    '    "Medial OA": 0,',
    '    "Lateral OA": 0,',
    '    "PF OA": 0,',
    '    "Effusion": 1,',
    '    "Synovitis": 0,',
    '    "Baker\'s": 0,',
    '    "Contusion": 0,',
    '    "Fracture": 0',
    '  },',
    '  "rationale": {',
    '    "Medial Meniscus": "Report mentions horizontal tear of the posterior horn",',
    '    "Effusion": "Report describes moderate intra-articular joint effusion"',
    '  }',
    '}',
    '"""',
    '',
    "print('Clinical prompt template compiled successfully.')"
]

cells.append(create_cell("code", "\n".join(prompt_lines)))

# Cell 5: Helper Functions for Parsing Model Outputs
cells.append(create_cell("code", """def parse_llm_json(raw_text):
    \"\"\"Extract and parse JSON object from LLM output string.\"\"\"
    try:
        # Search for JSON block within markdown or raw text
        match = re.search(r'\\{.*\\}', raw_text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            labels = data.get("labels", {})
            # Ensure all 12 targets are present with 0 or 1
            clean_labels = {t: int(labels.get(t, 0)) for t in TARGETS}
            rationale = data.get("rationale", {})
            return clean_labels, rationale
    except Exception as e:
        print(f"JSON parse error: {e}")
    
    # Fallback default: all zeros
    return {t: 0 for t in TARGETS}, {"error": "Failed to parse JSON"}
"""))

# Cell 6: Model 1 Execution (MedGemma / Gemma-2)
cells.append(create_cell("markdown", """## 3. Model 1 Execution: MedGemma / Gemma-2
We load the model using 4-bit quantization (`bitsandbytes`) for efficient execution on Kaggle GPU memory.
If attached via Kaggle's **Add Input** sidebar, the notebook automatically locates and loads the weights directly.
"""))

cells.append(create_cell("code", """from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Auto-detect locally attached Kaggle Model path in /kaggle/input
def find_local_model_path():
    for root, dirs, files in os.walk('/kaggle/input'):
        if 'config.json' in files and any(k in root.lower() for k in ['gemma', 'medgemma']):
            return root
    return None

local_model_dir = find_local_model_path()
if local_model_dir:
    MODEL_1_ID = local_model_dir
    print(f"Detected Kaggle attached model directory: {MODEL_1_ID}")
else:
    MODEL_1_ID = "google/gemma-2-9b-it"
    print(f"No local model found in /kaggle/input; using Hugging Face repo: {MODEL_1_ID}")

print(f"Loading Model 1 from: {MODEL_1_ID} in 4-bit...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

try:
    tokenizer_1 = AutoTokenizer.from_pretrained(MODEL_1_ID)
    model_1 = AutoModelForCausalLM.from_pretrained(
        MODEL_1_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    print("Model 1 loaded successfully.")
    
    medgemma_results = []
    for idx, row in sample_10.iterrows():
        prompt = CLINICAL_PROMPT_TEMPLATE.format(report_text=row['Report'])
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer_1.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model_1.generate(inputs, max_new_tokens=512, temperature=0.1, do_sample=False)
        
        response_text = tokenizer_1.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        labels, rationale = parse_llm_json(response_text)
        labels['StudyInstanceUID'] = row['StudyInstanceUID']
        labels['rationale'] = rationale
        medgemma_results.append(labels)
        print(f"Model 1: Completed Study [{idx+1}/10]")
        
    medgemma_df = pd.DataFrame(medgemma_results)
    print("\\nMedGemma Extraction Completed.")
except Exception as e:
    print(f"Notice: Loading {MODEL_1_ID} failed: {e}")
    print("Make sure you clicked '+ Add Input' -> Models -> 'gemma-2-9b-it' or 'medgemma' in the Kaggle notebook sidebar.")
"""))

# Cell 7: Model 2 Execution (Qwen 2.5 14B)
cells.append(create_cell("markdown", """## 4. Model 2 Execution: Qwen 2.5 14B-Instruct
Next, we run `Qwen/Qwen2.5-14B-Instruct` on the exact same 10 reports.
Qwen is fully open-access (no gated license token required) and has top multilingual benchmark accuracy.
"""))

cells.append(create_cell("code", """MODEL_2_ID = "Qwen/Qwen2.5-14B-Instruct"

print(f"Loading Model 2: {MODEL_2_ID} in 4-bit...")

try:
    tokenizer_2 = AutoTokenizer.from_pretrained(MODEL_2_ID)
    model_2 = AutoModelForCausalLM.from_pretrained(
        MODEL_2_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    print("Model 2 loaded successfully.")
    
    qwen_results = []
    for idx, row in sample_10.iterrows():
        prompt = CLINICAL_PROMPT_TEMPLATE.format(report_text=row['Report'])
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer_2.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model_2.generate(inputs, max_new_tokens=512, temperature=0.1, do_sample=False)
            
        response_text = tokenizer_2.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        labels, rationale = parse_llm_json(response_text)
        labels['StudyInstanceUID'] = row['StudyInstanceUID']
        labels['rationale'] = rationale
        qwen_results.append(labels)
        print(f"Model 2: Completed Study [{idx+1}/10]")
        
    qwen_df = pd.DataFrame(qwen_results)
    print("\\nQwen 2.5 14B Extraction Completed.")
except Exception as e:
    print(f"Error loading {MODEL_2_ID}: {e}")
"""))

# Cell 8: Comparison Matrix & Discrepancy Audit
cells.append(create_cell("markdown", """## 5. Side-by-Side Comparison & Discrepancy Audit
We now tabulate both model predictions side-by-side across all 10 studies (120 evaluation cells) and calculate the agreement rate.
Any differences are displayed with their respective rationales for clinical review.
"""))

cells.append(create_cell("code", """# Side-by-side comparison across all 10 studies
print("="*80)
print("SIDE-BY-SIDE EXTRACTION COMPARISON: MedGemma vs. Qwen 2.5 14B")
print("="*80)

# Check if both result sets exist
if 'medgemma_df' in locals() and 'qwen_df' in locals():
    comparisons = []
    total_cells = len(sample_10) * len(TARGETS)
    agreements = 0
    
    for idx, row in sample_10.iterrows():
        uid = row['StudyInstanceUID']
        m_row = medgemma_df[medgemma_df['StudyInstanceUID'] == uid].iloc[0]
        q_row = qwen_df[qwen_df['StudyInstanceUID'] == uid].iloc[0]
        
        for target in TARGETS:
            m_val = m_row[target]
            q_val = q_row[target]
            match = (m_val == q_val)
            if match:
                agreements += 1
                
            comparisons.append({
                'Study_Idx': idx + 1,
                'StudyInstanceUID': uid[:10] + '...',
                'Target': target,
                'MedGemma': m_val,
                'Qwen2.5': q_val,
                'Match': 'AGREE' if match else 'DISCREPANCY',
                'MedGemma_Rationale': m_row['rationale'].get(target, '-'),
                'Qwen_Rationale': q_row['rationale'].get(target, '-')
            })
            
    comp_df = pd.DataFrame(comparisons)
    print(f"Total Target Predictions: {total_cells}")
    print(f"Total Model Agreements:   {agreements} / {total_cells} ({agreements/total_cells*100:.1f}%)")
    
    # Filter and display discrepancies
    discrepancies_df = comp_df[comp_df['Match'] == 'DISCREPANCY']
    print(f"Total Discrepancies:      {len(discrepancies_df)}")
    
    if len(discrepancies_df) > 0:
        print("\\nDetailed Discrepancy Breakdown:")
        print(discrepancies_df[['Study_Idx', 'Target', 'MedGemma', 'Qwen2.5', 'MedGemma_Rationale', 'Qwen_Rationale']].to_string(index=False))
    else:
        print("\\nPerfect Agreement (100%) between MedGemma and Qwen 2.5 14B across all 120 targets!")
        
    # Save comparison table
    comp_df.to_csv("llm_comparison_10_studies.csv", index=False)
    print("\\nComparison table saved to: llm_comparison_10_studies.csv")
else:
    print("Run cells 6 and 7 above to generate MedGemma and Qwen extraction dataframes.")
"""))

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

out_dir = Path(__file__).resolve().parent
nb_path = out_dir / "rsna-knee-llm-benchmark.ipynb"

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print(f"Successfully generated notebook: {nb_path}")
