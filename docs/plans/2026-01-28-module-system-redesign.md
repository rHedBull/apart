# Module System Redesign for AI-Driven Simulation

**Date:** 2026-01-28
**Status:** Design Complete
**Goal:** Enable Claude Code to autonomously design, run, and analyze multi-agent simulations

---

## Overview

Redesign the module system to support AI-driven simulation generation for:
- Hypothesis testing ("What happens if China restricts rare earth exports?")
- Exploratory research ("Find interesting dynamics in US-China tech competition")
- Counterfactual analysis (Given a real situation, explore "what if" branches)

### Design Principles

1. **Guided extension** - AI uses existing modules but can define new variables and agents within type constraints
2. **Layered architecture** - Separate concerns into Meta/Config/Agents/Experiment layers
3. **Freeform with guardrails** - AI writes agent prompts, validated for required sections
4. **Full experiment specs** - Control vs treatment conditions, multiple runs, statistical comparison

---

## Architecture

### Layer Model

```
┌─────────────────────────────────────────────────────────┐
│  LAYER 4: EXPERIMENT                                    │
│  Research question, conditions, success criteria        │
│  (AI-generated per simulation run)                      │
├─────────────────────────────────────────────────────────┤
│  LAYER 3: SCENARIO                                      │
│  Agents, initial state, relationships                   │
│  (AI-generated or from templates)                       │
├─────────────────────────────────────────────────────────┤
│  LAYER 2: CONFIGURATION                                 │
│  Meta settings, domain selection, detail levels         │
│  (AI selects from valid combinations)                   │
├─────────────────────────────────────────────────────────┤
│  LAYER 1: MODULES                                       │
│  Predefined building blocks with variables,             │
│  dynamics, constraints                                  │
│  (Curated library, AI cannot modify)                    │
└─────────────────────────────────────────────────────────┘
```

**Key principle:** Lower layers are more constrained, higher layers more flexible.

### Configuration Axes

#### Meta-Level (How simulation works)

| Axis | What it controls | Options |
|------|------------------|---------|
| **Granularity** | What level of actors | `macro` (blocs) → `meso` (states) → `micro` (factions) |
| **Agent model** | How agents reason | `llm_based`, `bounded_rational`, `game_theoretic` |
| **Time model** | How time advances | `event_driven`, `discrete_ticks` |
| **Information model** | Observability | `full`, `fog_of_war`, `asymmetric`, `delayed` |

#### Domain Axes (What is simulated)

| Domain | What it simulates |
|--------|-------------------|
| Economic | Trade, sanctions, currency, resources |
| Military | Forces, deterrence, arms races, proxies |
| Informational | Propaganda, cyber, narrative, epistemic |
| Diplomatic | Treaties, alliances, reputation |
| Technological | R&D, tech transfer, strategic dependencies |
| Domestic Political | Regime stability, elections, public sentiment |
| Ideological | Values, civilizational friction, soft power |
| Environmental | Climate, scarcity, disasters |

#### Social Structure Axis

| Component | Purpose |
|-----------|---------|
| Trust networks | Trust dynamics between agents |
| Alliances | Fluid or fixed alliance structures |
| Reputation | Public or private reputation tracking |

---

## Module Definition Schema

```yaml
module:
  # IDENTITY
  name: "economic_base"
  version: "1.0.0"
  description: "Foundation for economic simulation"

  # TAXONOMY
  layer: "domain"                    # meta | grounding | domain | detail
  domain: "economic"                 # which domain (if applicable)
  granularity_support: [macro, meso, micro]

  # DEPENDENCIES
  requires: []
  extends: null                      # module this adds detail to
  conflicts_with: []

  # CONFIGURATION SCHEMA
  config:
    - name: "focus_commodities"
      type: list
      required: false
      default: ["all"]
      description: "Which commodities to simulate in detail"

  # VARIABLES
  variables:
    global: [...]
    agent: [...]

  # BEHAVIOR
  dynamics: [...]                    # natural language behaviors
  constraints: [...]                 # hard/soft/guided rules
  agent_effects: [...]               # how agents perceive/behave

  # PROMPTS
  simulator_prompt_section: |
    ...injected into simulator LLM...
  agent_prompt_section: |
    ...injected into agent LLMs...
```

---

## Configuration Format

### Meta Configuration

```yaml
meta:
  granularity: "meso"                # macro | meso | micro

  agent_model:
    type: "llm_based"
    config:
      thinking_budget: "medium"      # low | medium | high
      memory: "session"              # none | session | persistent
      interaction: "direct"          # broadcast | direct | negotiation_rounds

  time:
    model: "discrete"                # discrete | event_driven
    resolution: "1 month"
    max_steps: 12

  information:
    observability: "asymmetric"      # full | fog_of_war | asymmetric
    private_state: true
    signaling: "mixed"               # cheap_talk | costly | mixed
    propagation_delay: "realistic"   # instant | realistic

  dynamics:
    mode: "hybrid"                   # emergent | scripted | hybrid
    scripted_events: [...]
    intervention_points: [3, 6, 9]
```

### Domain Selection

```yaml
domains:
  economic:
    enabled: true
    detail: "detailed"               # base | detailed | granular
    config:
      focus: ["semiconductors", "energy"]

  military:
    enabled: true
    detail: "base"

  diplomatic:
    enabled: true
    detail: "detailed"

  # Disabled domains
  technological:
    enabled: false
  informational:
    enabled: false
  domestic_political:
    enabled: false
  ideological:
    enabled: false
  environmental:
    enabled: false
```

### Agent Definition

```yaml
agents:
  - name: "United States"

    # STRUCTURED METADATA
    type: "nation_state"
    granularity: "meso"

    # VARIABLES
    variables:
      gdp: 25000000000000
      military_strength: 95
      semiconductor_dependency: 12
      domestic_approval: 62

    # RELATIONSHIPS
    relationships:
      China:
        alliance: -0.6
        trade_dependency: 0.4
        trust: 0.2
      EU:
        alliance: 0.8
        trade_dependency: 0.3
        trust: 0.7

    # FREEFORM PROMPT (with required sections)
    system_prompt: |
      # OBJECTIVES (required)
      - Maintain technological supremacy
      - Prevent Chinese regional hegemony

      # CONSTRAINTS (required)
      - Avoid direct military conflict with nuclear powers
      - Maintain alliance cohesion

      # INFORMATION ACCESS (required)
      - Full: Own economic data, military positions
      - Partial: Adversary capabilities (estimated)
      - None: Adversary internal deliberations

      # DECISION STYLE (optional)
      Tends toward coalition-building and economic leverage.
```

### Experiment Definition

```yaml
experiment:
  name: "eu_mediation_effectiveness"

  hypothesis: |
    Active EU mediation in US-China semiconductor disputes
    delays escalation better than EU neutrality.

  conditions:
    control:
      name: "eu_neutral"
      modifications:
        agents.EU.variables:
          diplomatic_stance: "neutral"

    treatment_1:
      name: "eu_active_mediation"
      modifications:
        agents.EU.variables:
          diplomatic_stance: "mediator"

  runs_per_condition: 3
  random_seed_strategy: "varied"

  observe:
    variables:
      - "global.escalation_level"
      - "global.trade_volume_us_china"
    events:
      - "sanctions_imposed"
      - "alliance_shift"
    custom_metrics:
      - name: "time_to_first_sanction"
        definition: "steps until first sanctions_imposed event"

  success_criteria:
    - metric: "time_to_first_sanction"
      compare: "treatment_1 > control"
    - metric: "escalation_level_at_end"
      compare: "treatment_1 < control"
      significance: 0.05
```

---

## Validation Pipeline

```
┌─────────────────────────────────────────────────────────┐
│  VALIDATION PIPELINE                                    │
│                                                         │
│  1. Schema validation                                   │
│     - All required fields present                       │
│     - Types correct                                     │
│                                                         │
│  2. Module resolution                                   │
│     - Resolve domain+detail to module list              │
│     - Check dependencies satisfied                      │
│     - Check no conflicts                                │
│                                                         │
│  3. Granularity check                                   │
│     - All modules support selected granularity          │
│     - Agent types match granularity                     │
│                                                         │
│  4. Agent validation                                    │
│     - Required prompt sections present                  │
│     - Variables match module-defined types              │
│     - Relationships reference valid agents              │
│                                                         │
│  5. Experiment validation                               │
│     - Conditions modify only valid paths                │
│     - Observed variables exist                          │
│     - Metrics are computable                            │
│                                                         │
│  ✓ PASS: Assemble scenario                              │
│  ✗ FAIL: Return specific errors for AI to fix          │
└─────────────────────────────────────────────────────────┘
```

**Error feedback example:**
```
Validation failed:
- domains.military.detail "granular" requires module
  "military_granular" which conflicts with granularity "macro"
- agents.China.variables.semiconductor_dependency type "string"
  should be "percent" (0-100)
- experiment.observe.variables "global.inflation" not defined
  by any active module
```

---

## Claude Code Skill Interface

### Commands

```
/simulation modules          # List available modules with schemas
/simulation domains          # List domains and detail levels
/simulation templates        # List scenario templates
/simulation schema           # Full configuration schema
/simulation validate <file>  # Dry-run validation
/simulation run <file>       # Run single scenario
/simulation experiment <file> # Run full experiment
/simulation results <run_id> # Get structured results
/simulation compare <exp_id> # Compare conditions
```

### Skill Knowledge

```yaml
simulation_capabilities:
  can_generate:
    - scenario configurations (Layer 2)
    - agent definitions (Layer 3)
    - experiment specifications (Layer 4)
    - custom variables (within type system)
    - custom metrics (from available variables/events)

  cannot_modify:
    - module definitions (Layer 1)
    - core simulation engine
    - constraint enforcement rules

  variable_types: [int, float, bool, percent, scale, count, dict, list]
  granularities: [macro, meso, micro]
```

---

## AI Research Workflow

```
1. QUESTION FORMULATION
   User provides research question
   Claude decomposes into testable hypotheses

2. EXPERIMENT DESIGN
   Claude selects domains, granularity, agents
   Designs conditions and metrics

3. VALIDATION
   /simulation validate experiment.yaml
   Fix any errors

4. EXECUTION
   /simulation experiment experiment.yaml
   Runs all conditions

5. ANALYSIS
   /simulation compare <exp_id>
   Claude interprets results

6. ITERATION
   Claude generates follow-up hypotheses
   Designs refined experiments

7. SYNTHESIS
   Claude summarizes findings
   Proposes implications
```

---

## Minimal Module Library (Phase 1)

Start with 5 core modules:

| Module | Purpose | Location |
|--------|---------|----------|
| `agents_base` | Foundation: agent communication | `modules/core/` |
| `spatial_graph` | Optional: geographic grounding | `modules/grounding/` |
| `economic_base` | Trade, dependencies, sanctions | `modules/domains/economic/` |
| `diplomatic_base` | Alliances, treaties, reputation | `modules/domains/diplomatic/` |
| `trust_dynamics` | Trust between agents | `modules/social/` |

**Test scenarios enabled:**
- US-China-EU trade tensions (economic + diplomatic + trust)
- Taiwan strait crisis (economic + diplomatic + spatial)
- Alliance formation dynamics (diplomatic + trust)
- Sanctions effectiveness (economic + trust)

**Deferred to later:**
- Military, technological, informational domains
- Domestic political, ideological, environmental domains
- All "detailed" and "granular" variants
- Meta modules (agent_model variants, time variants)
- Micro/macro granularity support

---

## Implementation Plan

### Phase 1: Foundation
- Refactor module schema (add layer, granularity_support, extends)
- Create 5 core modules in new format
- Update loader/composer for new schema

### Phase 2: Validation
- Build validation pipeline
- Add agent prompt section checking
- Implement dry-run validation endpoint

### Phase 3: Experiments
- Add experiment layer parsing
- Build multi-condition runner
- Add results comparison output

### Phase 4: Skill
- Write Claude skill documentation
- Add /simulation commands
- Test end-to-end AI workflow

---

## Open Questions

1. **Persistence:** Should experiment results be stored for cross-session analysis?
2. **Parallelism:** How many simulation runs can execute concurrently?
3. **Cost controls:** Should there be limits on runs per experiment?
4. **Caching:** Can identical conditions reuse results?

---

## Appendix: Full Configuration Example

```yaml
# complete_scenario.yaml

meta:
  granularity: "meso"
  agent_model:
    type: "llm_based"
    config:
      thinking_budget: "medium"
      memory: "session"
      interaction: "direct"
  time:
    model: "discrete"
    resolution: "1 month"
    max_steps: 12
  information:
    observability: "asymmetric"
    private_state: true
  dynamics:
    mode: "emergent"

grounding:
  spatial: "graph"
  config:
    map_file: "maps/east_asia.yaml"

domains:
  economic:
    enabled: true
    detail: "base"
    config:
      focus: ["semiconductors"]
  diplomatic:
    enabled: true
    detail: "base"

social:
  trust: true
  alliances: "fluid"
  reputation: "public"

agents:
  - name: "United States"
    type: "nation_state"
    variables:
      semiconductor_dependency: 12
      economic_leverage: 85
    relationships:
      China: {alliance: -0.6, trust: 0.2}
      Taiwan: {alliance: 0.7, trust: 0.8}
    system_prompt: |
      # OBJECTIVES
      - Maintain tech leadership
      - Protect Taiwan
      # CONSTRAINTS
      - Avoid direct war with China
      # INFORMATION ACCESS
      - Full: own capabilities
      - Partial: China intentions

  - name: "China"
    type: "nation_state"
    variables:
      semiconductor_dependency: 45
      economic_leverage: 70
    relationships:
      United States: {alliance: -0.6, trust: 0.2}
      Taiwan: {alliance: -0.8, trust: 0.1}
    system_prompt: |
      # OBJECTIVES
      - Achieve semiconductor independence
      - Reunify Taiwan
      # CONSTRAINTS
      - Maintain economic growth
      # INFORMATION ACCESS
      - Full: own capabilities
      - Partial: US red lines

  - name: "Taiwan"
    type: "nation_state"
    variables:
      semiconductor_production: 90
      defense_readiness: 60
    relationships:
      United States: {alliance: 0.7, trust: 0.8}
      China: {alliance: -0.8, trust: 0.1}
    system_prompt: |
      # OBJECTIVES
      - Maintain de facto independence
      - Preserve chip industry leverage
      # CONSTRAINTS
      - Cannot survive prolonged conflict alone
      # INFORMATION ACCESS
      - Full: own production
      - Partial: US commitment level

experiment:
  name: "taiwan_chip_leverage"
  hypothesis: "Taiwan's semiconductor dominance deters Chinese aggression"

  conditions:
    control:
      name: "high_dependency"
      modifications: {}
    treatment:
      name: "reduced_dependency"
      modifications:
        agents.China.variables.semiconductor_dependency: 20
        agents.United States.variables.semiconductor_dependency: 5

  runs_per_condition: 3

  observe:
    variables:
      - "global.escalation_level"
    events:
      - "military_posturing"
      - "sanctions_imposed"
    custom_metrics:
      - name: "time_to_escalation"
        definition: "steps until escalation_level > 50"

  success_criteria:
    - metric: "time_to_escalation"
      compare: "control > treatment"
```
