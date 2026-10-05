# Gemma 4 Developer Agent: BM25 Multi-View RRF & AST Graph Hybrid Synthesis (Exp 10 Design)

## 1. Problem Formulation: Localization Speed & Call Overhead
In SWE-bench / Gemma 4 issue resolution across 129 hidden benchmark tasks:
- Naive agents waste 8–15 tool turns wandering directories using generic `ls` and `grep` commands.
- While AST graph tools (`search_similar_code`, `get_code_neighbors`) are highly accurate once a symbol is identified, identifying the initial candidate symbols from a lengthy issue description (often containing hundreds of lines of traceback and conversational noise) requires turn overhead.

---

## 2. Lucifer19 / Black Cat SWE-Agent (0.12 LB) Audit Insights
From our decompilation and audit of `lucifer19/black-cat-swe-agent-pack-instinct`:
1. **Four-View BM25 Reciprocal Rank Fusion (RRF)**:
   - Evaluates 4 distinct lexical views:
     - View 1: Extracted identifier names (CamelCase and snake_case tokens).
     - View 2: Quoted traceback frames and exception types (`TypeError`, `ValueError`).
     - View 3: Behavior divergence clauses ("expected ... but got ...").
     - View 4: Full issue text (bounded to 6,000 characters).
   - Combines results via Reciprocal Rank Fusion ($RRF(s) = \sum_{v=1}^4 \frac{1}{60 + rank_v(s)}$) in $\le 1.2$ seconds.
2. **Strict Output Contract**:
   ```
   LOCATION: <path>:<start>-<end> (<function or class>)
   ROOT CAUSE: <one or two sentences>
   FIX PLAN: <concrete change>
   RELATED: <other call sites or files needing same change, or "none">
   TESTS: <existing test files that exercise this code>
   CONFIDENCE: high | medium | low
   ```
3. **Execution Guardrails**:
   - `timeout_seconds: 300`
   - `max_tool_calls: 100`
   - `max_turns: 250`

---

## 3. Autobot Exp 10 Synthesis: BM25 Scout + AST Graph Hybrid
When Exp 9 (`56838874`) finishes scoring, our Exp 10 architecture combines:
1. **Turn 0 Pre-computation**: Instantaneous Python BM25 RRF indexing over the target repository to extract top-5 candidate functions.
2. **Turn 1 AST Neighbor Verification**: `get_code_neighbors()` and `get_code_subgraph()` directly seeded by the BM25 top candidates.
3. **Turn 2 Ephemeral Reproduction**: `/tmp/repro.py` isolated bug validation.
4. **Turn 3 Resilient Anchored Patch**: 6-line context edit with AST syntax validation before git commit.
