# Theoretical Background

This page provides the theoretical foundation for the Piston Leak dynamical systems model.

## Narrative Networks as Dynamical Systems

We conceptualize institutional narratives as emergent properties of coupled networks with directed graphs whose vertices represent narrative nodes (individuals, facts, claims) and edges represent narrative influence weights.

The key insight is that credibility is not a property of individual nodes but rather an emergent property of the entire network. When semantic tensions exceed critical thresholds, the network undergoes phase transitions analogous to those seen in physical systems.

## The Coupled ODE System

The core of our model is a coupled system of ordinary differential equations tracking three macro-variables:

\begin{aligned}
\dot{T} &= -\alpha_1\,N + \beta_1\,R - \gamma_1\,P\\
\dot{N} &= \alpha_2\,P + f_\text{noise}(t) - \beta_2\,T\\
\dot{P} &= k_\text{policy}\,g(N,T) - \delta P
\end{aligned}

Where:
- $T$ represents Trust
- $N$ represents Narrative Entropy
- $P$ represents Suppression Pressure
- $R$ represents Transparency (which can be time-dependent)
- $f_\text{noise}$ represents exogenous leaks
- $g(N,T)$ is the institutional policy response function

## Attractor Fracture Phenomena

The most significant contribution of our model is the identification of five key attractor-fracture modes that precede narrative collapse:

### Fractured Character Continuity (FCC)

When a single narrative node is forced to carry contradictory epistemic roles, it creates a topological fork in belief space. For example, when Dr. Fauci is simultaneously presented as an "apolitical scientist" and a "narrative architect," the roles create irreconcilable semantic tension.

FCC is quantified as:

$$\text{FCC}(t) = \gamma_{FCC} \frac{N(t) \cdot P(t)}{1 + T(t)}$$

### Recursive Suppression Disclosure (RSD)

When attempts to suppress information themselves become evidence for the suppressed claim, creating a feedback loop that amplifies entropy. The stronger the suppression effort, the more it signals the importance of the suppressed information.

RSD is quantified as:

$$\text{RSD}(t) = \gamma_{RSD} \frac{P(t)^2 \cdot N(t)}{1 + T(t)^2}$$

### Collapse Criterion

A Phase-I Collapse event is triggered when:

$$\frac{dT}{dt} < -\lambda_1 \quad \text{and} \quad N > N_c \quad \text{and} \quad |\text{FCC}(t) \cap \text{RSD}(t)| > \varepsilon$$

Where:
- $\lambda_1$ is the critical trust slope threshold
- $N_c$ is the critical entropy threshold
- $\varepsilon$ is the overlap significance threshold

## Cars Universe as Metaphoric Simulator

To make these abstract dynamics more intuitive, we map the institutional narrative system onto Pixar's *Cars* universe:

* **Cruz Ramirez** ≙ Fauci (institutional hub)
* **Doc Hudson** ≙ Legacy gain-of-function research
* **Lightning McQueen** ≙ Public consciousness

This metaphoric mapping enables the expression of otherwise inexpressible contradictions while maintaining intuitive accessibility.

## Entropy-First Governance

The model suggests that traditional "truth-first" governance approaches may be inadequate for maintaining narrative coherence under high stress. Instead, "entropy-first" governance focuses on maintaining semantic stability even when perfect information transmission is impossible.

The critical surface separating recovery from collapse basins occurs at:

$$(R/P)_c \approx 1.7$$

This suggests that maintaining a transparency-to-suppression ratio above this threshold is key to preventing narrative collapse.
