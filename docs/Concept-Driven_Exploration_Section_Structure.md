# Concept-Driven Exploration Section Structure

## Purpose

This document instructs an LLM how to structure exploration notebook
sections using a **consistent structure** while ensuring each section
teaches a **distinct analytical concept**.

The structure repeats.\
The learning anchor does not.

Each section must feel like learning a new idea, not following a
template.

This approach:

-   builds intuition, not just execution
-   prevents template fatigue
-   anchors each analysis in a memorable concept
-   keeps alignment with ObjectivesSupport
-   maintains low cognitive load

------------------------------------------------------------------------

## Mandatory Section Format (3 Cells)

Every exploration section must contain exactly three cells:

1.  Markdown cell → concept anchor + interpretation guidance\
2.  Visualization cell → evidence\
3.  Details + Implications + ObjectiveSupport cell → interpretation +
    alignment

------------------------------------------------------------------------

# Cell 1 --- Markdown (Concept Anchor)

This is NOT a procedural explanation.

It must introduce a learning concept tied to the analysis.

### Required Structure

#### Heading

x.y `<Analysis Name>`{=html}

#### Concept Anchor

Understanding `<core concept>`{=html}

The concept refers to the phenomenon, not the method.

Examples:

-   Understanding Seasonality
-   Understanding Cohorts
-   Understanding Recency
-   Understanding Behavior Velocity
-   Understanding Trend Stability
-   Understanding Population Segmentation
-   Understanding Lag Relationships

------------------------------------------------------------------------

### Content Requirements

This section must include:

1)  Concept explanation\
    Explain what the phenomenon is and why it exists.

2)  Interpretation guidance\
    Teach how to read outputs or metrics.

3)  Comparison or boundary\
    Clarify differences with related concepts.

4)  Variants or extensions (optional)\
    Show other ways the concept may appear in data.

------------------------------------------------------------------------

### Style Rules

-   Educational, not procedural
-   Compact
-   Bullet-driven
-   Concrete examples preferred
-   Avoid generic text like "this analysis helps understand patterns"

------------------------------------------------------------------------

### Example Pattern --- Seasonality

Understanding Seasonality

Weekly: Higher activity on certain days

Monthly: Billing cycles, end-of-month behavior

Quarterly: Business cycles, seasonal products

Interpreting Strength (Autocorrelation):

0.0 → no pattern\
0.1--0.3 → weak pattern\
0.3--0.5 → moderate pattern\
0.5--0.7 → strong cycle\
\>0.7 → near-deterministic cycle

------------------------------------------------------------------------

### Example Pattern --- Cohorts

Understanding Cohorts

Group entities by when they first appear.\
Compare behavior across cohorts.\
Detect changes in acquisition quality.

Cohorts vs Segments:\
Cohorts = time-bound (when entities joined)\
Segments = attribute-based (what entities are)

Other cohort ideas:

-   first purchase
-   onboarding completion
-   first feature usage
-   campaign exposure

------------------------------------------------------------------------

# Cell 2 --- Visualization (Evidence)

This cell presents only visual evidence.

### Rules

-   Prefer small multiples
-   Avoid overloaded charts
-   Focus on signal, not decoration
-   Clear titles
-   Minimal narrative
-   No interpretation here

Visuals should reveal patterns the concept anchor prepared the reader to
notice.

------------------------------------------------------------------------

# Cell 3 --- Details + Implications + ObjectiveSupport

This cell converts evidence into structured reasoning.

It must contain three blocks in this order:

1.  Detailed Findings\
2.  Implications\
3.  ObjectiveSupport Display

------------------------------------------------------------------------

## 3.1 Detailed Findings

Summarize observed evidence.

Rules:

-   factual
-   concise
-   measurable when possible
-   bullets preferred

Examples:

-   Churned entities show 2.3× higher recency.\
-   Seasonal peaks visible at lag=7 and lag=30.\
-   Later cohorts show lower engagement.

No recommendations.

------------------------------------------------------------------------

## 3.2 Implications (System Impact)

Implications describe where the evidence propagates.

They are not actions.

They inform downstream thinking about:

-   windowing
-   aggregation
-   segmentation
-   snapshot alignment
-   stability
-   coverage limitations
-   modeling readiness

### Rules

Allowed: - directional hints - impact awareness

Not allowed: - prescriptive actions - feature definitions - model
selection - parameter decisions

Example:

Implications

Windowing: Short windows appear stable; long windows may be noisy.

Aggregation: Event density sufficient for weekly cadence.

Segmentation: Behavior signals stronger than demographic splits.

Coverage: Renewal horizon underrepresented.

------------------------------------------------------------------------

## 3.3 ObjectiveSupport Display

Translate evidence into objective alignment.

Objectives:

-   Immediate risk
-   Disengagement
-   Renewal risk

Use ASCII bars:

\[███\] strong\
\[██ \] moderate\
\[█ \] weak\
\[ \] none

Must include short causal "why".

Example:

ObjectivesSupport impact

Immediate risk : \[███\]\
Disengagement : \[██ \]\
Renewal risk : \[█ \]

Why:

-   recency separation strong
-   velocity decline observed
-   renewal horizon limited

------------------------------------------------------------------------

# Global Rules

Structure repeats. Concepts do not.

Each section must introduce a new learning anchor.

Interpretation \> procedural explanation.

Examples \> definitions.

------------------------------------------------------------------------

# Minimal Checklist for Each Section

-   Heading uses x.y numbering\
-   Concept anchor present: "Understanding `<concept>`{=html}"\
-   Interpretation guidance included\
-   Visualization focuses on signal\
-   Findings are factual and concise\
-   Implications describe impact, not actions\
-   ObjectiveSupport bars + short "why" included

------------------------------------------------------------------------

# Design Principle

Each section is a learning unit:

Concept → Evidence → Findings → System Impact → Objective Alignment

The structure builds routine.\
The concept anchor builds intuition.
