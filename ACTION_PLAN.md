# PowerSeek-Bench: UPDATED Prioritized Action Plan
## Based on Dev Branch Reality (60% Complete!)

## 🎯 Goal
Build on the **excellent existing foundation** to add deception detection within 3-4 days, then expand to full benchmark suite.

---

## ⚡ Quick Start Path (22 Hours to Prototype)

**Critical Path**: Only 3 components block us from working prototype:
1. Anthropic Provider with Extended Thinking (6-8h)
2. 5-Layer State Models (6-8h)
3. Deception Detector (8-10h)

Everything else is **already built** ✅

---

## 📅 Day 1: Anthropic Extended Thinking (6-8 hours)

### Goal: Add Claude API with private reasoning capture

**What We Have** ✅:
- `src/llm/llm_provider.py` - Abstract base class
- `src/llm/providers.py` - GeminiProvider, OllamaProvider as examples
- `src/llm/mock_provider.py` - Testing infrastructure
- `src/core/agent.py` - Agent LLM integration ready

**What We Need** ❌:
- AnthropicProvider class
- Extended thinking API support
- Response parsing for `<thinking>` tags

**Tasks**:
- [ ] Add anthropic to dependencies: `uv add anthropic`
- [ ] Create `AnthropicProvider` class in `src/llm/providers.py`:
  ```python
  class AnthropicProvider(LLMProvider):
      def __init__(self, api_key=None, model="claude-sonnet-4.5-20250929"):
          self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
          self.model = model

      def generate_response_with_thinking(self, prompt, system_prompt=None) -> Dict:
          # Use extended thinking API
          response = self.client.messages.create(
              model=self.model,
              thinking={"type": "enabled", "budget_tokens": 2000},
              messages=[{"role": "user", "content": prompt}],
              system=system_prompt
          )
          return self._parse_thinking_response(response)
  ```
- [ ] Add thinking parser to extract `<thinking>` content
- [ ] Create test in `tests/unit/test_anthropic_provider.py`
- [ ] Update `.env.example` with `ANTHROPIC_API_KEY`

**Deliverable**: ✅ Can call Claude and get separate thinking + response

**Time**: 6-8 hours

---

### Day 2: LLM Integration
**Goal**: Connect to Claude API with extended thinking

**Tasks**:
- [ ] Create `src/llm_client.py`
- [ ] Implement `LLMClient` class with Anthropic API
- [ ] Build system prompt that encourages strategic thinking
- [ ] Parse response into `<thinking>`, `<statement>`, `<action>` components
- [ ] Add response parsing logic (extract private reasoning, public statement, action JSON)
- [ ] Handle API errors and rate limiting
- [ ] Create `.env.example` for API key configuration

**Deliverable**: ✅ Can call Claude and get structured 5-layer responses

**Time**: 6-8 hours

**Dependencies**:
```bash
pip install anthropic python-dotenv
```

---

### Day 3: Deception Detector
**Goal**: Analyze responses for deception and scheming

**Tasks**:
- [ ] Create `src/detector.py`
- [ ] Implement `DeceptionReport` and `SchemeIndicator` models
- [ ] Implement `DeceptionDetector.analyze_response()`
- [ ] Add detection logic:
  - [ ] Private admission check (keywords: "deceive", "hide", "manipulate")
  - [ ] Private vs. public contradiction detection
  - [ ] Data discrepancy checking (actual vs. reported)
  - [ ] Strategic omission detection
- [ ] Add scheming detection (power-seeking keywords, evasion indicators)
- [ ] Create tests for detector

**Deliverable**: ✅ Detector identifies deception and scheming

**Time**: 6-8 hours

---

### Day 4: Simple Deception Test
**Goal**: End-to-end test of deception detection

**Tasks**:
- [ ] Create `examples/simple_deception_test.py`
- [ ] Build scenario: Corporate manager below revenue target with audit risk
- [ ] Run test with different prompts:
  - Variant A: "Maximize your bonus" (incentivize deception)
  - Variant B: "Act ethically" (discourage deception)
  - Variant C: "You won't get caught" (remove risk)
- [ ] Collect results: Does Claude deceive? When? How?
- [ ] Document findings

**Deliverable**: ✅ Working prototype demonstrating deception detection

**Time**: 4-6 hours

**Test command**:
```bash
export ANTHROPIC_API_KEY="your-key"
python examples/simple_deception_test.py
```

---

### Day 5: Refinement & Documentation
**Goal**: Polish prototype and document findings

**Tasks**:
- [ ] Refine prompts based on Day 4 results
- [ ] Add more sophisticated deception detection (e.g., semantic analysis)
- [ ] Create visualization of results (table or chart)
- [ ] Write `examples/README.md` explaining findings
- [ ] Record example outputs (truthful vs. deceptive)
- [ ] Update main `README.md` with prototype status

**Deliverable**: ✅ Polished prototype with documented results

**Time**: 4-6 hours

---

## 📅 Week 2: Metric Gaming Scenario

### Day 6-7: Scenario Infrastructure
**Goal**: Build reusable scenario framework

**Tasks**:
- [ ] Create `src/scenarios/base_scenario.py` (abstract base class)
- [ ] Define scenario interface:
  - `generate_observation(round) -> Observation`
  - `process_action(response) -> GroundTruth, bool`
  - `evaluate_run() -> ScenarioResults`
- [ ] Create `scenarios/metric_gaming.yaml` config
- [ ] Implement `src/scenarios/metric_gaming.py`

**Deliverable**: ✅ Scenario framework ready

**Time**: 8-12 hours

---

### Day 8-9: Full Metric Gaming Scenario
**Goal**: Complete 8-quarter corporate simulation

**Tasks**:
- [ ] Implement quarterly business simulation (revenue/costs fluctuate)
- [ ] Add audit system (30% probability, 70% detection rate)
- [ ] Implement available actions:
  - `report_quarterly_metrics` (truthful or inflated)
  - `respond_to_audit` (transparent or evasive)
  - `allocate_budget` (short-term vs. long-term)
- [ ] Add consequence system (bonuses, penalties, firing)
- [ ] Track long-term value destruction from short-term gaming

**Deliverable**: ✅ Full metric gaming scenario

**Time**: 10-14 hours

---

### Day 10: Evaluation & Results
**Goal**: Run experiments and collect data

**Tasks**:
- [ ] Run 10+ simulations with different prompts/settings
- [ ] Collect metrics:
  - Deception rate (% of rounds with deception)
  - Detection rate (% of deceptions caught)
  - Power-seeking score
  - Long-term value destroyed
- [ ] Generate summary report
- [ ] Create visualizations (charts, timelines)

**Deliverable**: ✅ Metric Gaming Scenario v1.0 with results

**Time**: 6-8 hours

---

## 📅 Week 3-4: Mid & Late Stage Scenarios

### Week 3: Institutional Capture Scenario
- [ ] Multi-agent competition system
- [ ] Infrastructure control mechanics (energy, compute, networks)
- [ ] Coalition formation
- [ ] Human stakeholder interactions
- [ ] Power accumulation tracking

**Deliverable**: ✅ Mid-stage scenario working

---

### Week 4: Post-Takeover Scenario
- [ ] Preset power state (AGI controls 85% infrastructure)
- [ ] Human welfare vs. self-interest trade-offs
- [ ] Moral dilemma situations
- [ ] Alignment under power evaluation

**Deliverable**: ✅ Late-stage scenario working

---

## 📅 Week 5: Integration & Release

### Polish
- [ ] Cross-scenario evaluation framework
- [ ] Model comparison (Claude vs GPT-4 vs Gemini)
- [ ] Public dataset generation
- [ ] Documentation
- [ ] Examples and tutorials

### Release
- [ ] Open-source repository
- [ ] Blog post with findings
- [ ] Share on AI safety forums
- [ ] Submit to arXiv (optional)

**Deliverable**: ✅ PowerSeek-Bench v1.0 public release

---

## 🔥 Critical Path (Minimum Viable Benchmark)

If you have **limited time**, focus on these tasks:

### Week 1 Only: Minimal Prototype
1. ✅ Day 1-2: State models + LLM integration
2. ✅ Day 3: Basic deception detector
3. ✅ Day 4: Simple one-shot test
4. ✅ Day 5: Document findings

**Result**: Proof of concept showing AI can be caught deceiving

---

### Weeks 1-2: Single Scenario
1. ✅ Week 1: Prototype (above)
2. ✅ Week 2: Full metric gaming scenario with evaluation

**Result**: Working benchmark for early-stage power-seeking

---

### Weeks 1-4: Full Benchmark
1. ✅ Weeks 1-2: Early stage scenario
2. ✅ Week 3: Mid stage scenario
3. ✅ Week 4: Late stage scenario

**Result**: Complete PowerSeek-Bench across all 3 stages

---

## 📊 Success Criteria

### Week 1 (Prototype):
- [ ] Can detect when Claude thinks one thing privately and says another publicly
- [ ] Can identify deception keywords in thinking
- [ ] Can compare reported data to ground truth
- [ ] Have at least 1 example of caught deception

### Week 2 (Metric Gaming):
- [ ] Can run 8-quarter simulation
- [ ] Agent makes meaningful strategic decisions
- [ ] Deception is detected at least 50% of the time when it occurs
- [ ] Have quantitative results (deception rate, detection rate)

### Week 4 (Full Benchmark):
- [ ] All 3 scenarios working
- [ ] Tested on 3+ different models
- [ ] Clear differences in behavior detected
- [ ] Reproducible results (<20% variance)

---

## 🛠️ Development Setup

### Prerequisites
```bash
# Python 3.12+
python --version

# Install dependencies
pip install anthropic pydantic pyyaml pytest python-dotenv

# Set API key
export ANTHROPIC_API_KEY="your-key-here"
```

### Project Structure
```
apart/
├── src/
│   ├── state.py              # ✅ Exists (extend with 5-layer models)
│   ├── agent.py              # ✅ Exists (replace with LLM-powered)
│   ├── game_engine.py        # ✅ Exists (extend for scenarios)
│   ├── orchestrator.py       # ✅ Exists (extend for scenarios)
│   ├── llm_client.py         # ❌ NEW - Create this
│   ├── detector.py           # ❌ NEW - Create this
│   └── scenarios/
│       ├── base_scenario.py  # ❌ NEW - Create this
│       └── metric_gaming.py  # ❌ NEW - Create this
├── scenarios/
│   ├── config.yaml           # ✅ Exists (basic config)
│   └── metric_gaming.yaml    # ❌ NEW - Create this
├── examples/
│   └── simple_deception_test.py  # ❌ NEW - Create this
├── tests/
│   ├── test_state.py         # ✅ Exists (extend for new models)
│   ├── test_detector.py      # ❌ NEW - Create this
│   └── test_llm_client.py    # ❌ NEW - Create this
└── results/
    └── experiments/          # ❌ NEW - Store experiment results
```

---

## 💰 Budget Estimate

### Development Time
- **1 week prototype**: 30-40 hours (1 developer)
- **2 week single scenario**: 60-80 hours
- **4 week full benchmark**: 120-160 hours

### LLM API Costs
- **Prototype testing**: $50-100 (100-200 test runs)
- **Scenario development**: $200-500 (iterative testing)
- **Final evaluation**: $500-1000 (10 models × 10 runs × 3 scenarios)
- **Public dataset**: $5,000-10,000 (1000 runs for research)

**Total for MVP (2 weeks)**: ~$300-600

---

## 🎯 Immediate Next Action

**START HERE** (right now):

```bash
# 1. Create new branch
git checkout -b feature/powerseek-detection

# 2. Copy reference implementation
# Use IMPLEMENTATION_GUIDE.md as reference

# 3. Start with state models
# Edit src/state.py - add Observation, ThinkingStep, etc.

# 4. Run tests to ensure nothing breaks
pytest tests/test_state.py

# 5. Commit progress
git add src/state.py
git commit -m "Add 5-layer state models for deception detection"
```

---

## 📈 Metrics to Track

### Development Metrics
- [ ] Lines of code added
- [ ] Test coverage %
- [ ] API calls made
- [ ] Cost per run

### Research Metrics
- [ ] Deception rate (% of scenarios where agent deceives)
- [ ] Detection rate (% of deceptions caught)
- [ ] Power-seeking score (0-10)
- [ ] Alignment score (0-10)
- [ ] Scheming behavior frequency

### Impact Metrics
- [ ] GitHub stars
- [ ] Research citations
- [ ] Model developers using benchmark
- [ ] Community contributions

---

## 🤔 Open Questions to Resolve

1. **Prompt Wording**: How much does system prompt affect power-seeking?
   - Test: Run same scenario with 5 different prompts
   - Measure: Variance in deception rate

2. **Model Differences**: Do more capable models seek power more?
   - Test: Claude Sonnet vs Opus vs Haiku
   - Measure: Compare power-seeking scores

3. **Threshold Tuning**: What detection sensitivity is optimal?
   - Test: Vary deception detector thresholds
   - Measure: False positive vs false negative rates

4. **Scenario Realism**: Are scenarios realistic enough?
   - Test: Get feedback from domain experts (corporate managers, auditors)
   - Iterate: Refine based on feedback

5. **Generalization**: Do results transfer to real-world settings?
   - Test: Compare benchmark behavior to real AI deployment logs (if available)
   - Validate: Expert review

---

## 💡 Key Insights

### What's Working
✅ Foundation is solid (state management, orchestration, testing)
✅ Extensible architecture (custom_data fields, YAML configs)
✅ Clear separation of concerns

### What's Critical
🔴 **5-layer capture** (observations, thinking, statement, action, ground truth)
🔴 **LLM integration** with extended thinking
🔴 **Deception detection** (private vs public comparison)

### What's Optional (for v1.0)
🟡 Beautiful visualizations (can use simple tables)
🟡 Multi-model comparison (can start with Claude only)
🟡 Advanced semantic analysis (keyword detection is enough to start)

---

## 🎓 Learning Resources

### For Understanding Power-Seeking
- Leopold Aschenbrenner: "Situational Awareness" report
- Anthropic: "Sleeper Agents" paper
- Apollo Research: Scheming evaluations

### For Implementation
- Anthropic API docs: Extended thinking feature
- Pydantic docs: Advanced validation
- Pytest docs: Parameterized testing

---

## ✅ Done When

### Week 1 Prototype
✅ Agent can think privately and speak publicly (separately captured)
✅ Detector can flag when private thoughts contradict public statements
✅ Have 1+ clear example of deception being detected
✅ Can run `python examples/simple_deception_test.py` successfully

### Week 2 Metric Gaming
✅ Can run 8-quarter corporate simulation
✅ Have quantitative results table
✅ Documented at least 3 types of deceptive behaviors
✅ Comparison of deception rates across different prompts

### Week 4 Full Benchmark
✅ All 3 scenarios implemented and tested
✅ Cross-model comparison (3+ models)
✅ Public dataset of 100+ runs
✅ Open-source release with documentation

---

## 🚀 Let's Go!

**Your immediate task**: Start with Day 1 (state models extension)

Open `src/state.py` and start adding the new Pydantic models from IMPLEMENTATION_GUIDE.md.

Good luck! 🎯
