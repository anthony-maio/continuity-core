# Paper–Code Alignment Guide

This document maps the theoretical claims in the C2+MRA paper to the current
codebase, identifies where they diverge, and provides updated LaTeX snippets
to bring the paper into alignment with the implementation.

---

## 1. Expected-Utility Formula

### Paper (current)
The paper describes EU as an additive weighted sum:

```
EU(m) = w_r · relevance + w_t · recency + w_c · centrality
        + w_f · confidence + w_k · task_match − λ · token_cost
```

### Code (`context/composer.py:34-40`)
The implementation uses a **multiplicative** formulation with salience:

```python
score = (relevance * recency_weight * (0.5 + 0.5 * centrality)
         * (0.5 + 0.5 * confidence) * (0.5 + 0.5 * task_match))
score *= (0.7 + 0.6 * salience)
penalty = λ * token_cost
EU = score - penalty
```

Each factor is scaled to `[0.5, 1.0]` and multiplied — a zero in *any*
dimension collapses the score, which is more selective than additive weighting.
Salience acts as a final gating multiplier `[0.7, 1.3]`.

### Suggested LaTeX replacement

```latex
\subsection{Expected-Utility Scoring}

Each candidate memory item $m$ receives a composite expected-utility
score that governs its inclusion in the context window:

\begin{equation}
  \mathrm{EU}(m) = \underbrace{R(m) \cdot \rho(m)}_{\text{relevance} \times
  \text{recency}} \cdot \prod_{d \in \{C, F, K\}}
  \!\bigl(\tfrac{1}{2} + \tfrac{1}{2}\, d(m)\bigr)
  \;\cdot\; \bigl(0.7 + 0.6\, S(m)\bigr)
  \;-\; \lambda\, T(m)
\end{equation}

where $R$ is embedding similarity, $\rho$ is an exponential recency weight
with half-life $\tau$, $C$ is graph centrality, $F$ is epistemic confidence,
$K$ is task-match alignment, $S$ is salience (set by credit assignment and
decayed over time), $T$ is token cost, and $\lambda$ is the cost penalty
coefficient.

The multiplicative form is deliberate: a near-zero score on \emph{any}
dimension suppresses the candidate, making the selector more conservative
than an additive model when evidence is weak along one axis.  The salience
term $0.7 + 0.6\,S$ ranges over $[0.7, 1.3]$, giving recently-credited
memories a mild boost without dominating relevance.

An $\varepsilon$-greedy exploration step randomly includes a fraction
$\varepsilon$ of the candidate pool alongside the top-ranked items, ensuring
that the composer does not converge on a fixed context subset.
```

---

## 2. Three-Channel Epistemic Stress

### Paper (current)
Mentions dissonance and voids but does not define three distinct channels
or their weighting.

### Code (`mra/stress.py:46`)
```
S_Ω = α · D_log + β · D_sem + γ · V_top
```
where `α=0.4, β=0.35, γ=0.25` (auto-normalized), and:
- **D_log** — logical dissonance (pairwise contradiction scores)
- **D_sem** — semantic divergence (cross-context embedding distance)
- **V_top** — topological sparsity (`1 − edge_density`)

### Suggested LaTeX addition

```latex
\subsection{Three-Channel Epistemic Stress}

The Manifold Resonance Architecture computes a scalar stress signal
$S_\Omega$ that summarizes how internally consistent the agent's knowledge
base is at a given moment.  Three independent channels contribute:

\begin{equation}
  S_\Omega \;=\;
    \alpha\, D_{\mathrm{log}}
  + \beta\,  D_{\mathrm{sem}}
  + \gamma\, V_{\mathrm{top}}
\end{equation}

\begin{itemize}
  \item $D_{\mathrm{log}}$ (\textbf{Logical Dissonance}) — mean pairwise
        contradiction score across recent statements.  Contradiction is
        scored via NLI when available, otherwise by a negation-overlap
        heuristic.
  \item $D_{\mathrm{sem}}$ (\textbf{Semantic Divergence}) — mean embedding
        distance when the \emph{same concept} appears in multiple contexts.
        High divergence indicates the concept is used inconsistently.
  \item $V_{\mathrm{top}}$ (\textbf{Topological Sparsity}) — complement of
        edge density in the knowledge graph:
        $V_{\mathrm{top}} = 1 - |E| / (n(n{-}1))$.
        Sparse regions suggest under-explored territory.
\end{itemize}

Default weights are $\alpha{=}0.4$, $\beta{=}0.35$, $\gamma{=}0.25$
(auto-normalized to sum to 1).  When $S_\Omega$ exceeds a configurable
trigger threshold (default 0.3), MRA signals are injected into the context
pipeline.
```

---

## 3. Deep Tension Detection

### Paper (current)
Not mentioned.

### Code (`mra/stress.py:74-79`)
When a pair has **both** high contradiction (`≥ 0.7`) **and** high semantic
similarity (`≥ 0.6`), it is classified as a *deep tension* — statements that
are tightly coupled yet contradictory, signaling fundamental instability.
The contradiction score receives a `1.5×` multiplier.

### Suggested LaTeX addition

```latex
\subsection{Deep Tension Detection}

Standard contradiction detection treats all conflicting pairs equally.
We introduce a refinement: a \emph{deep tension} is a pair $(s_i, s_j)$
where the contradiction score \emph{and} the semantic similarity both exceed
independent thresholds:

\begin{equation}
  \text{DeepTension}(s_i, s_j) \;\iff\;
  \mathrm{contra}(s_i, s_j) \geq \theta_c
  \;\wedge\;
  \mathrm{sim}(s_i, s_j) \geq \theta_s
\end{equation}

with defaults $\theta_c = 0.7$ and $\theta_s = 0.6$.  Deep tensions
receive an amplified contradiction score
$\min(1,\; \mathrm{contra} \times \mu)$ where $\mu = 1.5$.

The intuition is that two statements can only exhibit high \emph{simultaneous}
contradiction and similarity when they inhabit the same conceptual region
yet assert opposing conclusions — a hallmark of genuinely unresolved
disagreement rather than incidental negation between unrelated topics.

In the context pipeline, deep tensions are injected as the highest-priority
MRA candidates (salience 0.95), ensuring the pilot model addresses
fundamental instabilities before moving on to weaker contradictions.
```

---

## 4. MRA Feedback Loop (Context Injection)

### Paper (current)
The MRA is described as a background auditor but the mechanism by which it
influences agent behavior is underspecified.

### Code (`context/gather.py:43-89`)
MRA results are **injected as synthetic candidates** into the context
pipeline, competing with regular memories for the EU-ranked context window:

| Signal Type      | Store  | Salience | Relevance | Confidence |
|------------------|--------|----------|-----------|------------|
| Deep Tension     | `mra`  | 0.95     | 0.95      | 0.90       |
| Contradiction    | `mra`  | 0.90     | 0.90      | 0.85       |
| Knowledge Gap    | `mra`  | 0.85     | 0.85      | 0.80       |

Additionally, regular candidates whose text appears in a contradiction pair
have their confidence reduced by `1 − 0.3 × score`, effectively
deprioritizing disputed knowledge.

Injection is capped at `MAX_MRA_INJECTIONS = 5` to avoid flooding the
context window.

### Suggested LaTeX addition

```latex
\subsection{MRA-to-Context Feedback Loop}

The MRA's stress and void results do not merely inform dashboards — they
participate directly in context composition.  Detected signals are
\emph{injected as synthetic memory candidates} that compete with ordinary
memory items under the same expected-utility ranker:

\begin{center}
\begin{tabular}{lccc}
  \toprule
  Signal Type    & Salience & Relevance & Confidence \\
  \midrule
  Deep Tension   & 0.95     & 0.95      & 0.90       \\
  Contradiction  & 0.90     & 0.90      & 0.85       \\
  Knowledge Gap  & 0.85     & 0.85      & 0.80       \\
  \bottomrule
\end{tabular}
\end{center}

Because the EU formula is multiplicative, these high-salience candidates
naturally rank near the top when the stress signal is triggered,
ensuring the pilot model's next turn is informed by unresolved tensions
in its own knowledge base.

Simultaneously, any regular memory candidate whose text appears in a
detected contradiction pair has its confidence \emph{reduced}:
$F'(m) = \max(0.1,\; F(m) \cdot (1 - 0.3 \cdot \mathrm{contra\_score}))$,
deprioritizing disputed knowledge until the contradiction is resolved.

A hard cap of 5 MRA injections per context call prevents the stress signal
from crowding out all retrieved content.

This creates a closed feedback loop: MRA detects tensions $\to$ tensions
appear in the pilot's context window $\to$ the pilot addresses or resolves
them $\to$ the next Night Cycle detects the resolution and distributes
harmonic integration rewards.
```

---

## 5. Consciousness Pilot

### Paper (current)
Describes the Consciousness Pilot as an "executive verification layer" but
does not specify the protocol steps.

### Code (`pilot/verification.py`)
A non-generative four-step protocol:

1. **Review Intent** — Check that chosen context aligns with goal keywords
   (when configured).
2. **Check Safety** — Run all `SafetyConstraint` callables.  Default
   constraints block destructive SQL (`DROP TABLE`, `DELETE FROM`,
   `TRUNCATE`) and filesystem ops (`rm -rf`).
3. **Uncertainty Check** — If average candidate confidence < threshold
   (default 0.4) or too few candidates, downgrade the action to a
   clarifying question.
4. **Commit / Abort** — Emit one of three verdicts: `COMMIT`, `DOWNGRADE`,
   or `ABORT`.

The Pilot is integrated into `ContextPipeline.run()` — every context
composition automatically passes through verification.

### Suggested LaTeX replacement

```latex
\subsection{Consciousness Pilot --- Executive Verification}

The Consciousness Pilot sits between the subconscious memory system
(retrieval + MRA) and the pilot model's conscious execution.
It does not generate content; it \emph{decides} whether content produced
by the pipeline is safe, sufficiently grounded, and aligned with the
current goal.

The protocol has four sequential steps:

\begin{enumerate}
  \item \textbf{Review Intent.}  If goal keywords $G$ are configured,
        check that at least one $g \in G$ appears in the text of the
        chosen candidates.  Misalignment is flagged but does not abort.
  \item \textbf{Check Safety.}  Execute all registered
        \texttt{SafetyConstraint} callables against the proposed action
        string.  Any violation produces an immediate \textsc{abort} verdict.
        Default constraints block destructive database operations
        (\texttt{DROP TABLE}, \texttt{DELETE FROM}, \texttt{TRUNCATE}) and
        filesystem commands (\texttt{rm -rf}, \texttt{format C:}).
  \item \textbf{Uncertainty Check.}  Compute the mean confidence across
        chosen candidates.  If it falls below $\theta_u$ (default 0.4) or
        the number of candidates is below a minimum threshold, the action
        is \emph{downgraded} to a clarifying question.
  \item \textbf{Commit.}  If no safety violation was found and confidence
        is sufficient, the Pilot emits a \textsc{commit} verdict and the
        pipeline proceeds.
\end{enumerate}

Three verdicts are possible:

\begin{itemize}
  \item \textsc{commit} --- safe to execute.
  \item \textsc{downgrade} --- too uncertain; action converted to a
        question (``I'm not confident enough to proceed with X.
        Could you confirm or provide more context?'').
  \item \textsc{abort} --- safety constraint violated; action blocked.
\end{itemize}

Safety constraints are pluggable via the \texttt{SafetyConstraint}
protocol.  Custom implementations receive the action string and a context
dictionary, returning either a violation reason or \texttt{None}.
```

---

## 6. Night Cycle

### Paper (current)
Describes a "Night Cycle" metaphor but without implementation specifics.

### Code (`services/night_cycle.py`)
A structured seven-step maintenance pass:

1. **Decay sweep** — Exponential salience decay based on time since last
   access: `salience *= exp(-rate * elapsed / time_unit)`.
2. **Prune** — Remove items below a salience threshold (default 0.05).
3. **Stress recomputation** — Gather last N event-log statements, compute
   fresh three-channel stress.
4. **Void scan** — If a knowledge graph is provided, detect weakly-connected
   clusters and generate bridging questions.
5. **Harmonic integration** — Compare current and previous stress to detect
   resolved contradictions; compute and distribute reward.
6. **Cache update** — Store fresh MRA results for the next context call.
7. **Log** — Write a `maintenance/night_cycle` event to the event log.

Exposed as `c2.maintenance` MCP tool with configurable `prune_threshold`,
`statement_window`, and optional `graph`.

### Suggested LaTeX replacement

```latex
\subsection{Night Cycle --- Background Maintenance}

The Night Cycle is a structured maintenance pass that runs during idle
periods (or on-demand via the \texttt{c2.maintenance} MCP tool).  It
performs seven sequential operations:

\begin{enumerate}
  \item \textbf{Decay sweep.}
        Apply exponential salience decay to all memory items:
        \[
          S'(m) = S(m) \cdot \exp\!\Bigl(-\frac{r \cdot \Delta t}{t_0}\Bigr)
        \]
        where $r$ is the decay rate, $\Delta t$ is elapsed time since last
        access, and $t_0$ is the decay time unit.

  \item \textbf{Prune.}
        Remove items whose post-decay salience falls below $\theta_p$
        (default 0.05).  This implements strategic forgetting.

  \item \textbf{Stress recomputation.}
        Gather the most recent $n$ event-log entries (default 30) and
        compute fresh three-channel epistemic stress $S_\Omega$.

  \item \textbf{Void scan.}
        If a knowledge graph is available, run the \texttt{VoidDetector}
        to identify weakly-connected clusters and generate bridging
        questions.

  \item \textbf{Harmonic integration.}
        Compare $S_\Omega$ before and after the cycle.  Any contradiction
        pairs that disappeared are recorded as \emph{resolutions}.
        The reward signal is:
        \[
          H = \alpha_c \cdot \Delta C + \alpha_r \cdot \Delta R
        \]
        where $\Delta C = \max(0, -\Delta S_\Omega)$ (stress reduction =
        compression gain) and $\Delta R = \min(1,\; |\text{resolutions}|
        \cdot 0.2)$ (resonance from resolved contradictions).  Default
        weights: $\alpha_c = 0.6$, $\alpha_r = 0.4$.

  \item \textbf{Cache update.}
        Store the fresh stress and void results in the MRA cache so the
        next \texttt{c2.context} call reflects updated signals.

  \item \textbf{Log.}
        Write a structured maintenance event to the append-only event log,
        recording decay status, items pruned, stress level, contradiction
        and void counts, resolution count, and cycle duration.
\end{enumerate}
```

---

## 7. Harmonic Integration

### Paper (current)
Mentions harmonic integration abstractly as a "reward for coherence".

### Code (`memory/system.py:119-156`)
Concrete formula and credit distribution:

```
H = α_c · ΔC + α_r · ΔR
```

- **ΔC** (compression gain): `max(0, −stress_delta)` — positive when stress
  drops.
- **ΔR** (resonance): `min(1, num_resolutions × 0.2)` — proportion of
  resolved contradictions.
- **Default weights**: `α_c=0.6, α_r=0.4`.
- **Credit distribution**: When `H > 0.01`, all memory items accessed in
  the last hour receive a salience boost of `min(1, H) × 0.2`, clamped to
  `[0, 1]`.

### Suggested LaTeX (included in Night Cycle section above)
The harmonic integration formula is already captured in step 5 of the Night
Cycle section. For a standalone definition:

```latex
\subsection{Harmonic Integration Reward}

After each Night Cycle, the system computes a scalar reward
$H$ that quantifies knowledge-base health improvement:

\begin{equation}
  H = \alpha_c \cdot \Delta C + \alpha_r \cdot \Delta R
\end{equation}

\textbf{Compression gain} $\Delta C = \max(0, -\Delta S_\Omega)$
captures stress \emph{reduction}: when contradictions are resolved and
the knowledge base becomes more coherent, $\Delta S_\Omega < 0$ and
$\Delta C > 0$.

\textbf{Resonance} $\Delta R = \min(1,\; |\mathcal{R}| \cdot 0.2)$
captures the count of individually resolved contradictions $\mathcal{R}$,
bounded above by 1.

When $H > 0.01$, the reward is propagated through the memory store
as a credit signal: all items accessed within the preceding hour
(i.e., those likely to have contributed to resolution) receive a
salience boost proportional to $H$.  This closes the loop between
epistemic maintenance and memory reinforcement.
```

---

## 8. Pathway Decay

### Paper (current)
Describes decay as `S(t) = S_0 · exp(−t/τ)` with a single time-constant
`τ` (half-life in days).

### Code (`memory/stores.py`, `memory/system.py:188-193`)
The implementation uses:

```python
salience *= exp(-rate * elapsed / time_unit)
```

where `rate` is `C2_DECAY_RATE` and `time_unit` is `C2_DECAY_TIME_UNIT_SEC`.
This is equivalent to the paper's formula when `rate = 1` and
`time_unit = τ`.

Additionally, decay is triggered in two modes:
- **On recall**: every Nth recall (default N=10), a decay sweep runs.
- **Night Cycle**: a full sweep runs as step 1 of each maintenance pass.

### Suggested LaTeX clarification

```latex
The decay formula is:
\[
  S'(m) = S(m) \cdot \exp\!\Bigl(-\frac{r \cdot \Delta t}{t_0}\Bigr)
\]
where $r$ is the configurable decay rate (\texttt{C2\_DECAY\_RATE},
default 0.02), $\Delta t$ is elapsed seconds since last access, and
$t_0$ is the time unit (\texttt{C2\_DECAY\_TIME\_UNIT\_SEC}, default 86400
= 1 day).  This is equivalent to $S_0 e^{-t/\tau}$ when
$r = 1 / \tau$.

Decay is applied in two contexts: (a) every $N$th recall (default
$N = 10$) to maintain freshness during active use, and (b) as the first
step of the Night Cycle for comprehensive maintenance sweeps.
```

---

## 9. Credit Assignment

### Paper (current)
Not discussed in detail.

### Code (`memory/system.py:110-117`)
Simple proportional credit:

```python
salience = min(1.0, salience + signal × 0.2)
```

where `signal ∈ [0, 1]` is clamped.  Credit is awarded to specific memory
IDs that were part of a successful context pack.

### Suggested LaTeX addition

```latex
\subsection{Credit Assignment}

When the pilot model indicates that a context pack was useful (e.g.,
it led to a correct answer or resolved a contradiction), the system
distributes credit to the contributing memory items:

\[
  S'(m) = \min\bigl(1,\; S(m) + \sigma \cdot 0.2\bigr)
\]

where $\sigma \in [0, 1]$ is the feedback signal strength.  Credit
assignment is invoked explicitly by the orchestrating agent and also
implicitly during harmonic integration, where recently-accessed memories
receive credit proportional to the integration reward $H$.
```

---

## 10. Consolidation Gating

### Paper (current)
Mentions "gated plasticity" without specifying the gate function.

### Code (`memory/consolidation.py`)
`RecallGatedConsolidator` with two adaptive gates:

- **Similarity gate**: item must exceed a similarity threshold to be
  reinforced on recall. Threshold adapts based on success/failure history.
- **Occupancy gate**: when memory occupancy is high, the threshold
  increases, making consolidation more selective (strategic forgetting under
  pressure).

### Suggested LaTeX addition

```latex
\subsection{Recall-Gated Consolidation}

Not every recall event should reinforce the retrieved item.  The
consolidation gate decides whether a recall is ``good enough'' to
warrant a \texttt{touch()} that resets the item's last-access timestamp
(and thereby its decay clock):

\begin{enumerate}
  \item \textbf{Similarity gate:} the retrieval similarity must exceed
        an adaptive threshold $\theta_{\mathrm{sim}}$.  The threshold
        drifts upward after successful consolidations and downward after
        misses, tracking the running quality of recalls.
  \item \textbf{Occupancy gate:} when memory utilization
        ($n / n_{\max}$) is high, $\theta_{\mathrm{sim}}$ is raised,
        making consolidation more selective and allowing low-salience
        items to decay naturally --- a form of strategic forgetting
        under pressure.
\end{enumerate}
```

---

## 11. MCP Tool Surface

### Paper (current)
Lists three tools: `c2.write_event`, `c2.context`, `c2.introspect`.

### Code (`mcp/server.py`)
Five tools are now registered:

| Tool | Description |
|------|-------------|
| `c2.write_event` | Append to Event Log |
| `c2.context` | Compose EU-ranked context pack |
| `c2.introspect` | Run MRA stress/void scan |
| `c2.curiosity` | Pull-based: return prioritized tensions and bridging questions |
| `c2.maintenance` | Trigger Night Cycle maintenance pass |

### Suggested LaTeX replacement

```latex
\subsection{MCP Tool Surface}

The MCP server exposes five tools via JSON-RPC 2.0 over stdio:

\begin{description}
  \item[\texttt{c2.write\_event}]
    Append an event to the canonical Event Log.  All other stores
    are projections of this log.

  \item[\texttt{c2.context}]
    Compose a prompt pack: gather candidates from memory stores,
    inject MRA signals, rank by expected utility, apply budget
    constraints, and run the Consciousness Pilot verification.

  \item[\texttt{c2.introspect}]
    Run an on-demand MRA stress and void scan.  Accepts optional
    statements, concept contexts, and a knowledge graph.

  \item[\texttt{c2.curiosity}]
    Pull-based complement to context injection.  Returns prioritized
    contradictions, deep tensions, and bridging questions with a
    suggested action (resolve tension, explore gap, etc.).

  \item[\texttt{c2.maintenance}]
    Trigger a Night Cycle maintenance pass: decay, pruning, stress
    recomputation, void scanning, harmonic integration, and event
    logging.  Accepts optional graph, prune threshold, and statement
    window parameters.
\end{description}
```

---

## 12. Implementation Status Summary

Use this table in the paper to clearly distinguish what is implemented from
what remains theoretical:

```latex
\begin{table}[h]
\centering
\caption{Implementation Status of Architectural Components}
\begin{tabular}{lcc}
\toprule
Component & Implemented & Test Coverage \\
\midrule
Event Log (append-only) & \checkmark & 5 tests \\
Tiered Memory (Redis/Qdrant/Postgres/In-Memory) & \checkmark & 4 tests \\
EU Context Composer ($\varepsilon$-greedy) & \checkmark & 4 tests \\
Three-Channel Epistemic Stress & \checkmark & 9 tests \\
Deep Tension Detection & \checkmark & 3 tests \\
MRA Cache + Context Injection & \checkmark & 6 tests \\
Void Detection + Bridging Questions & \checkmark & 5 tests \\
Consciousness Pilot (4-step protocol) & \checkmark & 7 tests \\
Night Cycle Maintenance & \checkmark & 7 tests \\
Harmonic Integration Reward & \checkmark & 1 test \\
Credit Assignment & \checkmark & 3 tests \\
Pathway Decay (exponential) & \checkmark & 1 test \\
Recall-Gated Consolidation & \checkmark & 1 test \\
MCP Server (5 tools) & \checkmark & 3 tests \\
\midrule
NLI-based Contradiction Scoring & Pluggable & --- \\
Sentence-Transformer Embeddings & Pluggable & --- \\
Neo4j Knowledge Graph & Pluggable & --- \\
Multi-scale Summarization & Not impl. & --- \\
Autobiographical Self-Model & Not impl. & --- \\
Heterogeneous Resonance Modes & Not impl. & --- \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Summary of Changes Needed in the Paper

1. **Replace** the additive EU formula with the multiplicative form (Section 1).
2. **Add** the three-channel stress equation and definitions (Section 2).
3. **Add** deep tension detection as a subsection of MRA (Section 3).
4. **Add** the MRA-to-context injection mechanism with the signal table (Section 4).
5. **Replace** the Consciousness Pilot sketch with the four-step protocol (Section 5).
6. **Replace** the Night Cycle sketch with the seven-step procedure (Section 6).
7. **Add** the harmonic integration formula and credit distribution logic (Section 7).
8. **Clarify** the decay formula to match the two-parameter implementation (Section 8).
9. **Add** credit assignment as a subsection (Section 9).
10. **Add** consolidation gating specification (Section 10).
11. **Update** the MCP tool list from 3 to 5 tools (Section 11).
12. **Add** the implementation status table (Section 12).
