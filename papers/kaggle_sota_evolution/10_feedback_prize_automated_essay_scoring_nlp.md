# From Sub-Word Encoders to Autoregressive Foundation LLMs: The Evolution of Automated Essay Scoring and Discourse Analysis in the Feedback Prize Challenges

**Authors**: Autonomous Machine Learning Systems Group & Autobot AI Research  
**Date**: September 2026  
**Subject**: Natural Language Processing, Automated Essay Scoring, Discourse Analysis, Large Language Models

---

## Abstract
Automated evaluation of student argumentative writing is a longstanding grand challenge in artificial intelligence, bridging psychometrics, educational measurement, and natural language processing. The Kaggle Feedback Prize competitions—spanning discourse element token classification, text readability, student writing effectiveness, and holistic essay scoring—benchmarked NLP architectures on hundreds of thousands of student essays across varying demographic backgrounds and grade levels. This paper tracks the technical progression of NLP algorithms across these competitions. We examine: (1) the dominance of disentangled attention transformer encoders (DeBERTa-v3) over standard BERT/RoBERTa; (2) token-level BIO sequence tagging vs. paragraph-level hierarchical pooling; (3) ordinal regression, Mean Squared Error, and Quadratic Weighted Kappa (QWK) loss dynamics; (4) prompt-engineered chain-of-thought fine-tuning and representation extraction from modern instruction-tuned LLMs (e.g., LLaMA, Mistral, Gemma); and (5) feature engineering coupling semantic embeddings with traditional stylistic, syntactic, and discourse markers.

---

## 1. Domain Context & The Feedback Prize Benchmark Series
Automated writing evaluation systems must assess essays across multiple pedagogical dimensions:
1. **Discourse Role Identification**: Segmenting argumentative discourse units: *Lead*, *Position*, *Claim*, *Counterclaim*, *Rebuttal*, *Evidence*, and *Concluding Statement*.
2. **Writing Effectiveness**: Evaluating the rhetorical strength (Ineffective, Adequate, Effective) of each identified discourse component.
3. **Automated Holistic Essay Scoring**: Assigning a global standardized score ($1$ to $6$) adhering to strict scoring rubrics.

```
                      Student Argumentative Essay
                                  │
                                  ▼
                    [Text Preprocessing & Tokenization]
                    DeBERTa-v3 Byte-Pair / SentencePiece
                                  │
                                  ▼
               [Stage 1: Contextual Transformer Encoder]
               DeBERTa-v3-Large / LLaMA-3-8B Feature Extraction
                                  │
                                  ▼
               [Stage 2: Multi-Granularity Representation]
               - Token-level Hidden States
               - Mean / Max / Attention Pooling
               - Linguistic Surface Features (Length, Lexical Diversity)
                                  │
                                  ▼
               [Stage 3: Multi-Task Classification / Regression]
               - Discourse BIO Sequence Tagging Head
               - Ordinal Logit Regression Head
                                  │
                                  ▼
               [Stage 4: Post-Processing & Threshold Optimization]
               Nelder-Mead / Powell Optimization on Quadratic Weighted Kappa
```

### Key Technical Challenges
1. **Long-Context Dependencies**: Student essays regularly exceed 800 to 1,500 words, exceeding the standard 512-token context window of classic transformer encoders.
2. **Informal and Non-Standard Language**: Student writing features frequent misspellings, colloquial syntax, run-on sentences, and unconventional punctuation that disrupt standard sub-word tokenizers.
3. **Severe Evaluator Variance & Subjectivity**: Human teacher ratings exhibit substantial inter-annotator noise; training on uncalibrated subjective grades leads to model instability.

---

## 2. Evaluation Metrics & Optimization Dynamics

### 2.1 Quadratic Weighted Kappa (QWK)
Holistic essay scoring is scored via Quadratic Weighted Kappa over discrete rubric levels ($k \in \{1, 2, 3, 4, 5, 6\}$):

$$\kappa = 1 - \frac{\sum_{i, j} w_{i, j} O_{i, j}}{\sum_{i, j} w_{i, j} E_{i, j}}, \quad w_{i, j} = \frac{(i - j)^2}{(C - 1)^2}$$

Because human raters grade essays along a continuous underlying quality continuum, rounding discrete categorical predictions introduces quantization noise. Winning models trained with **continuous Mean Squared Error (MSE)** or **Smooth L1 Loss**, followed by Nelder-Mead threshold search to optimize the decision boundaries:

$$\hat{y}_{\text{discrete}} = \begin{cases} 
1 & \text{if } \hat{y} < \tau_1 \\ 
2 & \text{if } \tau_1 \le \hat{y} < \tau_2 \\ 
\vdots \\ 
6 & \text{if } \hat{y} \ge \tau_5 
\end{cases}$$

### 2.2 Word-Level Overlap F1 Score
In discourse element segmentation, predictions were evaluated by word-level overlap: a predicted discourse span was deemed a true positive if and only if:

$$\frac{|\text{Pred} \cap \text{GroundTruth}|}{|\text{Pred} \cup \text{GroundTruth}|} \ge 0.5$$

---

## 3. The Algorithmic Evolution: From BERT to DeBERTa to Modern LLMs

### 3.1 The Absolute Reign of DeBERTa-v3
Across every single edition of the Feedback Prize, **DeBERTa-v3** (Decoding-enhanced BERT with Disentangled Attention) decisively outperformed all competing transformer architectures (RoBERTa, ELECTRA, Longformer, BigBird).
- **Disentangled Attention**: DeBERTa represents each word using two distinct vectors: content $\mathbf{c}_i$ and relative position $\mathbf{p}_{i|j}$. Attention weights are decomposed into four components:
  $$A_{i, j} = \mathbf{c}_i \mathbf{c}_j^T + \mathbf{c}_i \mathbf{p}_{j|i}^T + \mathbf{p}_{i|j} \mathbf{c}_j^T + \mathbf{p}_{i|j} \mathbf{p}_{j|i}^T$$
  Because rhetorical discourse markers (e.g., *"In conclusion"*, *"However"*) depend critically on both lexical identity and position within the essay (beginning vs. end), disentangled relative position modeling provided a profound inductive advantage.
- **Enhanced Masked Language Modeling (RTD)**: Replaced MLM with ELECTRA-style replaced token detection, boosting parameter efficiency.

### 3.2 Handling Long Context: Stride Windows & Hierarchical Pooling
To process essays extending beyond 512 tokens:
- **Sliding Window Chunking**: DeBERTa-v3 evaluated documents using 512-token windows with 128-token strides. Overlapping token embeddings were averaged, eliminating boundary disconnects.
- **Attention Pooling**: Instead of relying exclusively on the `[CLS]` token (which acts as a bottleneck), winning teams pooled all token embeddings using learnable multi-head attention pooling:
  $$\mathbf{h}_{\text{pooled}} = \sum_{t=1}^T \text{softmax}\left(\mathbf{w}^T \tanh(\mathbf{W}_h \mathbf{h}_t)\right) \mathbf{h}_t$$

### 3.3 Hybridization: LLM Foundation Features + GBDT Meta-Learners
In the most recent holistic essay scoring benchmarks, the winning paradigm converged on a hybrid neuro-symbolic framework:
1. **Transformer Encoders**: Extracted continuous essay quality representations from 5-fold cross-validated DeBERTa-v3-Large models.
2. **Instruction-Tuned LLMs (Mistral-7B / LLaMA-3)**: Prompted to generate rubric-aligned critique features and intermediate perplexity scores.
3. **Explicit Stylistic & Linguistic Features**:
   - Essay word count, paragraph count, average sentence length.
   - Lexical diversity (Type-Token Ratio, Yule's Characteristic $K$).
   - Syntactic dependency tree depth (via spaCy).
   - Spell-check error frequencies and spelling divergence indices.
4. **Second-Level Stacking**: LightGBM, CatBoost, and Ridge regressors blended deep neural embeddings with linguistic surface features, preventing over-reliance on complex vocabulary while capturing structural cohesion.

---

## 4. Leaderboard Synthesis

| Competition | Winning Paradigm | Primary Backbones | Hybrid GBDT Features | Metric Score |
| :--- | :--- | :--- | :--- | :--- |
| **Feedback Prize 1 (Discourse)** | BIO Token Classifier + CRF | DeBERTa-v3-Large + Longformer | Rule-based post-filtering | **0.732 Overlap F1** |
| **Feedback Prize 2 (Effectiveness)** | Multi-Task Classification | Dual DeBERTa-v3-Large + RoBERTa-Large | Discourse role concatenation | **0.551 Log-Loss** |
| **Feedback Prize 3 (Readability)** | Continuous MSE Regression | DeBERTa-v3 + RoBERTa-Large | Text readability formulas | **0.446 RMSE** |
| **Feedback Prize 2024 (AES)** | Neuro-Symbolic Stacking | DeBERTa-v3-Large + Mistral-7B + GBDT | 150+ Lexical/Syntactic features | **0.841 QWK** |

---

## 5. Conclusions & Pedagogical Future
The Feedback Prize series confirmed that automated essay scoring has reached human-level reliability when assessed against consensus grading rubrics. The key breakthroughs—disentangled relative positional encoding in DeBERTa-v3, continuous QWK threshold optimization, and the integration of deep semantic representations with structural linguistic priors—represent the blueprint for next-generation automated pedagogical feedback platforms.

---

## References
1. Kaggle & The Learning Agency Lab. *Feedback Prize - Automated Essay Scoring Benchmark Series*, 2021–2024.
2. He, P., Gao, J., & Chen, W. "DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding." *ICLR*, 2023.
3. Ke, G., et al. "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." *NeurIPS*, 2017.
4. Page, E. B. "The Imminence of Grading Essays by Computer." *Phi Delta Kappan*, 47(5), 238-243, 1966.
