# Methodology Documentation

## Overview

This document provides detailed methodology for the multi-agent orchestration system and conversational behavior analysis.

## Table of Contents

1. [Multi-Agent Architecture](#multi-agent-architecture)
2. [Nash Bargaining Mechanism](#nash-bargaining-mechanism)
3. [Conversational Behavior Analysis](#conversational-behavior-analysis)
4. [Personality and Expertise Diversity](#personality-and-expertise-diversity)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Statistical Analysis](#statistical-analysis)

---

## Multi-Agent Architecture

### Agent Personas

The system employs three specialized agents, each with distinct goals and personality traits:

#### Safety Agent (DeepSeek-V3)
- **Goal**: Maximize clinical risk coverage and guideline adherence
- **Personality**: Low agreeableness (2.1/5), High conscientiousness (4.2/5)
- **Behavior**: Critical, thorough, focused on evidence-based preventive care
- **Temperature**: 0.1 (deterministic)

#### Efficiency Agent (Claude 3.5 Sonnet)
- **Goal**: Maximize brevity and workflow usability
- **Personality**: High openness (3.9/5), Low neuroticism (1.8/5)
- **Behavior**: Pragmatic, solution-focused, emphasizes actionable brevity
- **Temperature**: 0.3 (balanced)

#### Equity Agent (GPT-5.2)
- **Goal**: Maximize social determinants of health integration
- **Personality**: High agreeableness (4.5/5), High neuroticism (3.7/5)
- **Behavior**: Empathetic, advocacy-oriented, sensitive to structural barriers
- **Temperature**: 0.5 (creative)

### System Prompts

Complete agent prompts are provided in the manuscript Appendix Section 1. Key elements:

**Safety Agent Prompt**:
```
You are Dr. Safety, a Clinical Risk Manager and Safety Officer.
Your ONLY goal is to ensure that NO clinical risk is missed.
Ignore concerns about length or social context.
It is better to be redundant than negligent.
```

**Efficiency Agent Prompt**:
```
You are the Operations Manager for a high-volume care team.
Your goal is to maximize throughput and effective functioning.
Identify "Note Bloat." Convert paragraphs to bullet points.
If the plan is too long, they will not read it at all.
```

**Equity Agent Prompt**:
```
You are a senior Clinical Social Worker and Health Equity Advocate.
Your goal is to ensure the plan fits the patient's life, not just biology.
If a patient has no car, a "Referral to Cardiology" is useless
without "Transportation Setup."
```

---

## Nash Bargaining Mechanism

### Mathematical Formulation

Let $S$ be the space of all possible care plan texts.
Let $u_i(t): S \to [0,1]$ be the utility function of agent $i \in \{Safety, Efficiency, Equity\}$.

The Nash Product $NP(t)$ is defined as:

$$
NP(t) = \prod_{i=1}^3 (u_i(t) - d_i)^{\alpha_i}
$$

where:
- $d_i$ is the disagreement point (minimum acceptable utility = 0.2)
- $\alpha_i$ is the bargaining power weight

### Bargaining Weights

Default configuration (validated in manuscript):
- $\alpha_{Safety} = 1.5$ (highest priority)
- $\alpha_{Equity} = 1.0$ (medium priority)
- $\alpha_{Efficiency} = 0.8$ (lowest priority)

**Rationale**: Safety is prioritized to prevent missed diagnoses and ensure guideline adherence. The multiplicative Nash product ensures no dimension can be completely sacrificed (score ≈0 collapses the entire product).

### Iterative Refinement Algorithm

```
Input: Patient context X
Output: Care plan text t*

Step 0: Generate initial draft t₀ from X
Step k:
  1. Each agent evaluates t_k → (score_i, critique_i)
  2. If min(scores) > threshold AND ΔNP < ε: STOP
  3. Orchestrator prompts Generator:
     "Rewrite t_k to maximize joint satisfaction given critiques"
  4. Generator yields t_{k+1}
  5. k = k + 1

Convergence: Typically 3 iterations (94% of cases)
```

### Utility Function Estimation

Agent utilities are estimated via critique-based scoring:
- **Safety**: Fraction of guideline-recommended actions present
- **Efficiency**: Inverse of text length × concept density
- **Equity**: Fraction of validated social needs addressed

---

## Conversational Behavior Analysis

### LLM-as-Judge Framework

We use Gemini-2.0-Flash as an LLM-as-judge to classify transcripts. Validation:
- Human rater agreement: ICC(3,1) = 0.76 (95% CI: 0.65-0.84)
- Inter-human reliability: ICC(3,1) = 0.72

### Conversational Behavior Definitions

Adapted from Kim et al. (2026) "Reasoning Models Generate Societies of Thought":

#### 1. Question-Answering Sequences
**Definition**: One agent poses a clinical question; another agent (or same agent later) resolves it.

**Example**:
```
Safety Agent: "What about the rising creatinine trend?"
Equity Agent: "The patient missed nephrology due to transportation."
```

**Classification Prompt** (abbreviated):
```
Does one agent pose a clinical question and another resolve it?
Respond: {"question_answering": true/false}
```

#### 2. Perspective Shifts
**Definition**: Agents explore alternative clinical viewpoints or approaches.

**Example**:
```
"However, from an efficiency standpoint..."
"Looking at this from a social work lens..."
```

#### 3. Conflicts of Perspectives
**Definition**: Sharp contrasts or disagreements between agents' approaches.

**Example**:
```
Safety: "We cannot omit statin therapy—it's guideline-recommended"
Efficiency: "The patient is already on six medications—adding another
            without addressing adherence barriers is futile"
```

#### 4. Reconciliation
**Definition**: Integration or synthesis of conflicting viewpoints into coherent solutions.

**Example**:
```
"We can address both concerns by consolidating the diabetes actions
into one quarterly visit with adherence support."
```

### Bales' Interaction Process Analysis

We classify socio-emotional roles using Bales' IPA framework (Bales, 1950):

**12 Role Categories** (grouped into 4 higher-level categories):

1. **Asking** (for orientation, opinion, suggestion)
   - Example: "Should we include transportation?"

2. **Giving** (orientation, opinion, suggestion)
   - Example: "The patient needs HbA1c monitoring"

3. **Negative Emotional** (disagreement, antagonism, tension)
   - Example: "This approach is unsafe"

4. **Positive Emotional** (agreement, solidarity, tension release)
   - Example: "Good point, that's a reasonable compromise"

### Role Balance Metric

Jaccard index measures reciprocal role-taking:

$$
J_{asking \& giving} = \frac{|\text{both roles present}|}{|\text{either role present}|}
$$

High Jaccard index indicates balanced, interactive dialogue rather than monologue.

---

## Personality and Expertise Diversity

### BFI-10 Personality Assessment

Agents are rated using the Big Five Inventory 10-item short form (Rammstedt & John, 2007):

**Five Dimensions** (1-5 scale):
- **Extraversion**: Talkative, outgoing vs. reserved, quiet
- **Agreeableness**: Sympathetic, warm vs. critical, quarrelsome
- **Conscientiousness**: Dependable, organized vs. disorganized, careless
- **Neuroticism**: Anxious, moody vs. calm, emotionally stable
- **Openness**: Creative, imaginative vs. conventional, uncreative

**Personality Diversity** = Standard deviation across agents for each dimension.

**LLM-as-Judge Prompt** (abbreviated):
```
Rate this agent's personality based on communication style (1-5):
- Extraversion: How assertive, energetic, talkative?
- Agreeableness: How cooperative vs. critical?
[etc.]
Respond: {"extraversion": X, "agreeableness": Y, ...}
```

### Expertise Diversity Measurement

**Step 1**: Extract domain expertise descriptions via LLM-as-judge
```
Example outputs:
- Safety Agent: "Clinical guidelines and pharmaceutical safety"
- Efficiency Agent: "Workflow optimization and care coordination"
- Equity Agent: "Social determinants of health"
```

**Step 2**: Compute semantic embeddings (text-embedding-3-large, OpenAI)

**Step 3**: Calculate mean pairwise cosine distance:

$$
D_{expertise} = \frac{1}{\binom{3}{2}} \sum_{i<j} (1 - \cos(e_i, e_j))
$$

where $e_i$ is the embedding vector for agent $i$'s expertise description.

**Interpretation**: Higher distance = more diverse expertise specialization

---

## Evaluation Metrics

### Safety Score

**Definition**: Sensitivity for predicting future adverse events

$$
Safety = \frac{TP}{TP + FN}
$$

where:
- TP = Risk mentioned in plan AND adverse event occurred
- FN = Risk not mentioned AND adverse event occurred

**Ground Truth**: ED visits, hospitalizations in 90-day outcome window

### Efficiency Score

**Definition**: Clinical concepts per token (information density)

$$
Efficiency = \frac{\text{# unique clinical concepts}}{\text{# tokens}} \times \text{normalization factor}
$$

Penalizes verbosity while rewarding information richness.

### Equity Score

**Definition**: Recall for validated social needs

$$
Equity = \frac{TP_{social}}{TP_{social} + FN_{social}}
$$

where:
- $TP_{social}$ = Need mentioned AND CHW-verified
- $FN_{social}$ = Need not mentioned AND CHW-verified

**Ground Truth**: Community Health Worker assessments (transportation, housing, food)

### Composite Quality Score

Geometric mean across domains:

$$
Composite = \sqrt[3]{Safety \times Efficiency \times Equity}
$$

**Rationale**: Multiplicative structure ensures balanced performance (low score in any dimension drastically reduces composite).

---

## Statistical Analysis

### Sample Sizes

- **Retrospective cohort**: N=127,801 Medicaid beneficiaries
- **Conversational analysis**: N=500 randomly sampled transcripts (250 Nash, 250 single-agent)
- **Physician adjudication**: N=90 high-complexity cases

### Power Calculation

Physician adjudication sample size (N=90) provides 80% power to detect 15% absolute difference in preference rates at α=0.05.

### Statistical Tests

- **Performance metrics**: Two-sided t-tests for comparing strategies
- **Conversational behaviors**: Chi-square tests for categorical variables
- **Diversity metrics**: Two-sample t-tests with Bonferroni correction
- **Significance threshold**: p < 0.05 (two-tailed)

### Confidence Intervals

All reported CIs are 95% confidence intervals using normal approximation for proportions and t-distribution for continuous metrics.

### Fixed Effects Models

Conversational behavior analysis uses linear probability models with problem-level fixed effects:

$$
Y_{im} = \beta \cdot \text{Nash}_m + \gamma_i + \epsilon_{im}
$$

where:
- $Y_{im}$ = Behavior indicator for problem $i$, model $m$
- $\text{Nash}_m$ = Indicator for Nash vs. single-agent
- $\gamma_i$ = Problem-specific fixed effect (controls for difficulty)
- $\epsilon_{im}$ = Error term

This controls for task-specific characteristics.

---

## References

1. Kim J, Lai S, Scherrer N, et al. Reasoning Models Generate Societies of Thought. *arXiv*. 2026;arXiv:2601.10825.
2. Nash JF. The Bargaining Problem. *Econometrica*. 1950;18(2):155-162.
3. Bales RF. Interaction Process Analysis: A Method for the Study of Small Groups. *Addison-Wesley*. 1950.
4. Rammstedt B, John OP. Measuring personality in one minute or less: A 10-item short version of the Big Five Inventory. *J Res Pers*. 2007;41(1):203-212.
5. Woolley AW, et al. Evidence for a Collective Intelligence Factor in the Performance of Human Groups. *Science*. 2010;330(6004):686-688.

---

For implementation details, see `src/conversational_analysis.py` and `src/care_plan_generator.py`.
