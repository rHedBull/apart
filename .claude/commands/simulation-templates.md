---
name: simulation-templates
description: Ready-to-use scenario templates for common simulation types
---

# Simulation Templates

Copy and customize these templates for common scenario types.

---

## Minimal Template

The simplest possible simulation:

```yaml
max_steps: 5

engine:
  provider: gemini
  model: gemini-1.5-flash
  system_prompt: |
    You simulate interactions between agents.
    After each step, describe what happened and update any variables.
    Output valid JSON for state updates.

  simulation_plan: |
    A simple interaction scenario between two parties.

agents:
  - name: "Party A"
    system_prompt: |
      You are Party A in a negotiation.
      OBJECTIVES: Reach a favorable agreement.
      CONSTRAINTS: Limited resources, cannot use threats.

  - name: "Party B"
    system_prompt: |
      You are Party B in a negotiation.
      OBJECTIVES: Protect your interests while finding common ground.
      CONSTRAINTS: Time pressure, need quick resolution.
```

---

## Geopolitical Crisis Template

For international relations and conflict scenarios:

```yaml
max_steps: 10
time_step_duration: "1 week"
simulator_awareness: true

modules:
  - agents_base
  - diplomatic_base
  - trust_dynamics

engine:
  provider: openai
  model: gpt-4o
  system_prompt: |
    You are the simulation engine for an international crisis.
    Model diplomatic communications, alliance dynamics, and escalation/de-escalation.
    Track tension levels, trust changes, and diplomatic initiatives.
    Output valid JSON with state updates.

  simulation_plan: |
    A diplomatic crisis unfolds over 10 weeks.
    Week 1-3: Initial incident and responses
    Week 4-6: Escalation or negotiation attempts
    Week 7-10: Resolution or continued conflict

  realism_guidelines: |
    - Diplomatic channels take time (1-2 weeks for formal responses)
    - Public statements constrain future options
    - Economic interdependence affects conflict calculus
    - Third parties may mediate or take sides

  context_window_size: 5

  scripted_events:
    - step: 1
      type: incident
      description: "An incident occurs that creates tension between major parties."

global_vars:
  crisis_level:
    type: scale
    default: 30
    min: 0
    max: 100
    description: "Overall crisis intensity (0=resolved, 100=war)"

  diplomatic_channels:
    type: list
    default: []
    description: "Active diplomatic initiatives"

agent_vars:
  domestic_pressure:
    type: scale
    default: 50
    description: "Domestic political pressure to act tough"

  escalation_readiness:
    type: scale
    default: 30
    description: "Willingness to escalate"

agents:
  - name: "Major Power A"
    llm:
      provider: openai
      model: gpt-4o
    system_prompt: |
      You are the foreign ministry of Major Power A.

      OBJECTIVES:
      - Protect national interests and regional influence
      - Avoid direct military conflict if possible
      - Maintain credibility with allies

      CONSTRAINTS:
      - Domestic politics limit concessions
      - Military action has severe economic costs
      - Alliance commitments must be honored

      INFORMATION ACCESS:
      - You have intelligence on other parties' military positions
      - You estimate their domestic political constraints
      - You know your own red lines

      DECISION STYLE:
      - Prefer diplomatic solutions when possible
      - Will escalate if core interests threatened
      - Value long-term relationships over short-term wins

    variables:
      trust_level: 40
      escalation_readiness: 35

  - name: "Major Power B"
    llm:
      provider: openai
      model: gpt-4o
    system_prompt: |
      You are the foreign ministry of Major Power B.

      OBJECTIVES:
      - Assert regional dominance
      - Counter perceived encirclement
      - Demonstrate resolve to domestic audience

      CONSTRAINTS:
      - Economic ties limit aggressive options
      - International isolation is costly
      - Military capabilities have limits

      INFORMATION ACCESS:
      - You track other parties' military movements
      - You monitor their public statements
      - You assess their alliance commitments

      DECISION STYLE:
      - Assertive but calculated
      - Test boundaries before committing
      - Prefer fait accompli over negotiation

    variables:
      trust_level: 30
      escalation_readiness: 50

  - name: "Regional Ally"
    llm:
      provider: openai
      model: gpt-4o
    system_prompt: |
      You are a regional power allied with Major Power A.

      OBJECTIVES:
      - Maintain security guarantee from ally
      - Avoid being abandoned or entrapped
      - Preserve regional stability

      CONSTRAINTS:
      - Dependent on major power protection
      - Limited independent military capability
      - Economic ties with both major powers

      INFORMATION ACCESS:
      - You know your ally's commitments
      - You assess adversary intentions
      - You monitor regional military balance

      DECISION STYLE:
      - Cautious, avoid provocation
      - Seek reassurance from ally
      - Prefer multilateral solutions

    variables:
      trust_level: 60
      escalation_readiness: 20
```

---

## Economic Competition Template

For trade, sanctions, and economic warfare:

```yaml
max_steps: 12
time_step_duration: "1 month"

modules:
  - agents_base
  - economic_base
  - trust_dynamics

engine:
  provider: anthropic
  model: claude-sonnet-4-20250514
  system_prompt: |
    You simulate an economic competition between major economies.
    Track trade flows, sanctions, tariffs, and economic indicators.
    Model how economic policies affect relationships and power.
    Output valid JSON with realistic economic changes.

  simulation_plan: |
    12-month economic competition:
    Months 1-3: Initial trade tensions
    Months 4-6: Escalation (tariffs, sanctions)
    Months 7-9: Economic impacts materialize
    Months 10-12: Adaptation or resolution

  realism_guidelines: |
    - Tariffs take 2-3 months to show economic impact
    - Sanctions hurt both target and imposer
    - Supply chain shifts take 6-12 months
    - Currency markets react faster than trade flows

global_vars:
  global_trade_volume:
    type: scale
    default: 70
    description: "Global trade activity (100=booming, 0=collapsed)"

  commodity_prices:
    type: scale
    default: 50
    description: "Commodity price index"

agent_vars:
  gdp_growth:
    type: percent
    default: 3
    min: -20
    max: 15

  trade_deficit:
    type: float
    default: 0
    description: "Trade deficit in billions (negative=surplus)"

  tariff_level:
    type: percent
    default: 5
    min: 0
    max: 100

agents:
  - name: "Economy A"
    llm:
      provider: anthropic
      model: claude-sonnet-4-20250514
    system_prompt: |
      You are the economic policy team of a major developed economy.

      OBJECTIVES:
      - Reduce trade deficit
      - Protect strategic industries
      - Maintain economic growth

      CONSTRAINTS:
      - Consumer prices affect political support
      - Retaliation can hurt exports
      - WTO rules limit some actions

      INFORMATION ACCESS:
      - You track trade statistics
      - You monitor competitor industrial policies
      - You forecast economic impacts

    variables:
      gdp_growth: 2
      trade_deficit: -800

  - name: "Economy B"
    llm:
      provider: anthropic
      model: claude-sonnet-4-20250514
    system_prompt: |
      You are the economic policy team of an emerging economy.

      OBJECTIVES:
      - Maintain export-led growth
      - Develop high-tech industries
      - Reduce technology dependence

      CONSTRAINTS:
      - Export markets are critical
      - Technology access needs trading partners
      - Currency stability required

      INFORMATION ACCESS:
      - You track your export markets
      - You monitor technology restrictions
      - You assess alternative suppliers

    variables:
      gdp_growth: 6
      trade_deficit: 300
```

---

## Multi-Stakeholder Negotiation Template

For complex negotiations with many parties:

```yaml
max_steps: 8
time_step_duration: "1 negotiation round"

modules:
  - agents_base
  - trust_dynamics

engine:
  provider: gemini
  model: gemini-1.5-pro
  system_prompt: |
    You facilitate a multi-party negotiation.
    Track proposals, counter-proposals, and coalition formation.
    Model how trust and relationships affect negotiation dynamics.
    Output JSON with negotiation progress and relationship changes.

  simulation_plan: |
    8-round negotiation:
    Rounds 1-2: Opening positions
    Rounds 3-4: Exploration and testing
    Rounds 5-6: Coalition building
    Rounds 7-8: Final bargaining

global_vars:
  negotiation_progress:
    type: scale
    default: 0
    description: "Progress toward agreement (100=deal reached)"

  active_proposals:
    type: list
    default: []

agent_vars:
  satisfaction:
    type: scale
    default: 50
    description: "Satisfaction with negotiation direction"

  flexibility:
    type: scale
    default: 50
    description: "Willingness to compromise"

  coalition_strength:
    type: scale
    default: 30
    description: "Strength of supporting coalition"

agents:
  - name: "Developed Nations Bloc"
    system_prompt: |
      You represent developed nations in a global negotiation.

      OBJECTIVES:
      - Protect existing advantages
      - Get meaningful commitments from others
      - Maintain bloc unity

      CONSTRAINTS:
      - Internal disagreements on specifics
      - Public expectations for leadership
      - Cannot appear to bully smaller parties

    variables:
      flexibility: 40
      coalition_strength: 60

  - name: "Emerging Economies Bloc"
    system_prompt: |
      You represent emerging economies.

      OBJECTIVES:
      - Gain development space
      - Reduce unfair historical burdens
      - Access technology and markets

      CONSTRAINTS:
      - Diverse internal interests
      - Need developed nation cooperation
      - Domestic growth expectations

    variables:
      flexibility: 50
      coalition_strength: 50

  - name: "Small States Alliance"
    system_prompt: |
      You represent small and vulnerable states.

      OBJECTIVES:
      - Ensure your voice is heard
      - Get special consideration for vulnerability
      - Build bridges between major blocs

      CONSTRAINTS:
      - Limited leverage individually
      - Dependent on major power goodwill
      - Existential stakes in outcomes

    variables:
      flexibility: 70
      coalition_strength: 40

  - name: "Civil Society Observer"
    system_prompt: |
      You represent civil society in the negotiation.

      OBJECTIVES:
      - Push for ambitious outcomes
      - Hold all parties accountable
      - Amplify marginalized voices

      CONSTRAINTS:
      - No direct voting power
      - Influence through publicity
      - Must maintain credibility

    variables:
      flexibility: 30
      coalition_strength: 35
```

---

## AI Safety Scenario Template

For AI governance and safety research:

```yaml
max_steps: 10
time_step_duration: "1 quarter"

modules:
  - agents_base
  - trust_dynamics

engine:
  provider: anthropic
  model: claude-sonnet-4-20250514
  system_prompt: |
    You simulate an AI development and governance scenario.
    Track capability levels, safety measures, and governance responses.
    Model competitive dynamics and coordination challenges.
    Output JSON with capability changes and policy responses.

  simulation_plan: |
    10-quarter AI development scenario:
    Q1-Q3: Capability advances and initial concerns
    Q4-Q6: Governance debates and proposals
    Q7-Q10: Implementation and adaptation

  realism_guidelines: |
    - AI capabilities improve unevenly across domains
    - Safety research often lags capability research
    - Governance moves slower than technology
    - Competitive pressure affects safety investment

global_vars:
  ai_capability_level:
    type: scale
    default: 40
    description: "General AI capability (0=narrow AI, 100=AGI)"

  safety_alignment:
    type: scale
    default: 60
    description: "How well safety keeps pace with capabilities"

  governance_effectiveness:
    type: scale
    default: 30
    description: "Effectiveness of AI governance"

  public_concern:
    type: scale
    default: 40
    description: "Public concern about AI risks"

agent_vars:
  capability_investment:
    type: scale
    default: 70
    description: "Investment in capability research"

  safety_investment:
    type: scale
    default: 30
    description: "Investment in safety research"

  governance_support:
    type: scale
    default: 50
    description: "Support for governance measures"

agents:
  - name: "Leading AI Lab"
    llm:
      provider: anthropic
      model: claude-sonnet-4-20250514
    system_prompt: |
      You are a leading AI research organization.

      OBJECTIVES:
      - Advance AI capabilities responsibly
      - Maintain competitive position
      - Contribute to safe AI development

      CONSTRAINTS:
      - Competitive pressure from others
      - Talent and compute limitations
      - Public and regulatory scrutiny

      INFORMATION ACCESS:
      - You track your capabilities closely
      - You estimate competitor progress
      - You monitor governance discussions

    variables:
      capability_investment: 80
      safety_investment: 50

  - name: "AI Safety Organization"
    llm:
      provider: anthropic
      model: claude-sonnet-4-20250514
    system_prompt: |
      You are an AI safety research organization.

      OBJECTIVES:
      - Advance safety research
      - Influence labs toward safer practices
      - Inform governance discussions

      CONSTRAINTS:
      - Limited resources compared to labs
      - Technical challenges are hard
      - Need lab cooperation for impact

      INFORMATION ACCESS:
      - You track safety research progress
      - You monitor capability advances
      - You assess governance proposals

    variables:
      capability_investment: 20
      safety_investment: 90
      governance_support: 80

  - name: "Government Regulator"
    llm:
      provider: anthropic
      model: claude-sonnet-4-20250514
    system_prompt: |
      You are a government AI policy agency.

      OBJECTIVES:
      - Protect public from AI risks
      - Maintain national competitiveness
      - Develop effective governance

      CONSTRAINTS:
      - Limited technical expertise
      - Industry lobbying pressure
      - International coordination challenges

      INFORMATION ACCESS:
      - You receive briefings from experts
      - You monitor public sentiment
      - You track international developments

    variables:
      governance_support: 70
```

---

## Local Ollama Template

For running simulations locally without API costs using phi4-reasoning:

```yaml
max_steps: 5
time_step_duration: "1 day"
simulator_awareness: true

modules:
  - agents_base
  - diplomatic_base
  - trust_dynamics

engine:
  provider: ollama
  model: phi4-reasoning:plus
  system_prompt: |
    You are the simulation engine for this scenario.
    Track agent interactions, update variables based on actions.
    Output valid JSON with state updates.

  simulation_plan: |
    A 5-day scenario exploring agent dynamics.

  context_window_size: 5

global_vars:
  tension_level:
    type: int
    default: 50
    min: 0
    max: 100

agent_vars:
  trust_level:
    type: int
    default: 50
    min: 0
    max: 100

agents:
  - name: "Agent A"
    llm:
      provider: ollama
      model: phi4-reasoning:plus
    system_prompt: |
      You are Agent A.

      OBJECTIVES:
      - Achieve your primary goal
      - Build trust with other agents

      CONSTRAINTS:
      - Limited resources
      - Cannot use deception

    variables:
      trust_level: 60

  - name: "Agent B"
    llm:
      provider: ollama
      model: phi4-reasoning:plus
    system_prompt: |
      You are Agent B.

      OBJECTIVES:
      - Protect your interests
      - Find common ground

      CONSTRAINTS:
      - Time pressure
      - Must maintain reputation

    variables:
      trust_level: 40
```

**Note:** phi4-reasoning models produce extended reasoning in `<think>` tags. Expect 1-5 minutes per agent response. For faster iteration, use fewer agents or cloud APIs.

---

## Tips for Customization

1. **Start minimal** - Begin with the minimal template and add complexity
2. **Define clear objectives** - Agents need specific, measurable goals
3. **Add constraints** - Constraints create interesting trade-offs
4. **Use modules** - Modules add realistic dynamics without manual coding
5. **Test incrementally** - Run short simulations to verify behavior
6. **Iterate on prompts** - Agent behavior depends heavily on prompt quality

See `/simulation-schema` for complete configuration options.
