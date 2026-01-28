# Module System Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the layered module system that enables AI-driven simulation generation with validation, experiment support, and a Claude Code skill interface.

**Architecture:** Four-layer system (Modules → Config → Agents → Experiment) with validation pipeline. Modules define building blocks, config selects and composes them, agents define actors, experiments define research questions with conditions.

**Tech Stack:** Python 3.11+, YAML, pytest, existing APART codebase

---

## Phase 1: Enhanced Module Schema

### Task 1.1: Add New Fields to BehaviorModule Model

**Files:**
- Modify: `src/modules/models.py:183-214`
- Test: `tests/unit/test_module_models.py` (new file)

**Step 1: Write the failing test**

```python
# tests/unit/test_module_models.py
"""Tests for enhanced module models."""
import pytest
from modules.models import (
    BehaviorModule,
    ModuleVariable,
    ModuleLayer,
    Granularity,
    VariableType,
)


def test_module_layer_enum():
    """Test ModuleLayer enum has required values."""
    assert ModuleLayer.META.value == "meta"
    assert ModuleLayer.GROUNDING.value == "grounding"
    assert ModuleLayer.DOMAIN.value == "domain"
    assert ModuleLayer.DETAIL.value == "detail"


def test_granularity_enum():
    """Test Granularity enum has required values."""
    assert Granularity.MACRO.value == "macro"
    assert Granularity.MESO.value == "meso"
    assert Granularity.MICRO.value == "micro"


def test_behavior_module_new_fields():
    """Test BehaviorModule has new taxonomy fields."""
    module = BehaviorModule(
        name="test_module",
        description="Test",
        layer=ModuleLayer.DOMAIN,
        domain="economic",
        granularity_support=[Granularity.MESO, Granularity.MICRO],
        extends="economic_base",
    )

    assert module.layer == ModuleLayer.DOMAIN
    assert module.domain == "economic"
    assert Granularity.MESO in module.granularity_support
    assert module.extends == "economic_base"


def test_behavior_module_defaults():
    """Test BehaviorModule new fields have sensible defaults."""
    module = BehaviorModule(
        name="minimal",
        description="Minimal module",
    )

    assert module.layer == ModuleLayer.DOMAIN
    assert module.domain is None
    assert module.granularity_support == [Granularity.MACRO, Granularity.MESO, Granularity.MICRO]
    assert module.extends is None
```

**Step 2: Run test to verify it fails**

Run: `cd /home/hendrik/coding/ai-safety/apart && uv run pytest tests/unit/test_module_models.py -v`
Expected: FAIL with "cannot import name 'ModuleLayer'"

**Step 3: Write minimal implementation**

Add to `src/modules/models.py` after line 10 (after existing Enum imports):

```python
class ModuleLayer(str, Enum):
    """Which architectural layer a module belongs to."""
    META = "meta"
    GROUNDING = "grounding"
    DOMAIN = "domain"
    DETAIL = "detail"


class Granularity(str, Enum):
    """Actor granularity levels."""
    MACRO = "macro"   # Blocs, institutions
    MESO = "meso"     # Nation-states
    MICRO = "micro"   # Factions, individuals
```

Then update the `BehaviorModule` dataclass (around line 183) to add new fields after `version`:

```python
@dataclass
class BehaviorModule:
    """
    A composable behavior module for simulations.

    Modules encapsulate a domain of behavior (military, economics, etc.)
    with all the variables, dynamics, constraints, and effects needed.
    """
    name: str
    description: str
    version: str = "1.0.0"

    # NEW: Taxonomy fields
    layer: ModuleLayer = ModuleLayer.DOMAIN
    domain: Optional[str] = None  # e.g., "economic", "military"
    granularity_support: List[Granularity] = field(
        default_factory=lambda: [Granularity.MACRO, Granularity.MESO, Granularity.MICRO]
    )
    extends: Optional[str] = None  # Module this adds detail to

    # ... rest of existing fields unchanged ...
```

**Step 4: Run test to verify it passes**

Run: `cd /home/hendrik/coding/ai-safety/apart && uv run pytest tests/unit/test_module_models.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/modules/models.py tests/unit/test_module_models.py
git commit -m "feat(modules): add layer and granularity taxonomy to BehaviorModule"
```

---

### Task 1.2: Update ModuleLoader to Parse New Fields

**Files:**
- Modify: `src/modules/loader.py:126-174`
- Test: `tests/unit/test_module_loader.py` (new file)

**Step 1: Write the failing test**

```python
# tests/unit/test_module_loader.py
"""Tests for module loader with new schema fields."""
import pytest
import tempfile
import os
from pathlib import Path
from modules.loader import ModuleLoader
from modules.models import ModuleLayer, Granularity


@pytest.fixture
def temp_modules_dir():
    """Create a temporary directory with test module files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_loader_parses_layer_field(temp_modules_dir):
    """Test loader parses layer field from YAML."""
    module_yaml = """
module:
  name: test_module
  description: Test module
  layer: grounding
  domain: null
"""
    (temp_modules_dir / "test_module.yaml").write_text(module_yaml)

    loader = ModuleLoader(temp_modules_dir)
    module = loader.load("test_module")

    assert module.layer == ModuleLayer.GROUNDING


def test_loader_parses_granularity_support(temp_modules_dir):
    """Test loader parses granularity_support field."""
    module_yaml = """
module:
  name: meso_only
  description: Meso-level only module
  granularity_support: [meso]
"""
    (temp_modules_dir / "meso_only.yaml").write_text(module_yaml)

    loader = ModuleLoader(temp_modules_dir)
    module = loader.load("meso_only")

    assert module.granularity_support == [Granularity.MESO]


def test_loader_parses_extends_field(temp_modules_dir):
    """Test loader parses extends field."""
    module_yaml = """
module:
  name: economic_detailed
  description: Detailed economic module
  layer: detail
  domain: economic
  extends: economic_base
"""
    (temp_modules_dir / "economic_detailed.yaml").write_text(module_yaml)

    loader = ModuleLoader(temp_modules_dir)
    module = loader.load("economic_detailed")

    assert module.extends == "economic_base"
    assert module.layer == ModuleLayer.DETAIL
    assert module.domain == "economic"


def test_loader_defaults_new_fields(temp_modules_dir):
    """Test loader applies defaults for new fields."""
    module_yaml = """
module:
  name: minimal
  description: Minimal module without new fields
"""
    (temp_modules_dir / "minimal.yaml").write_text(module_yaml)

    loader = ModuleLoader(temp_modules_dir)
    module = loader.load("minimal")

    assert module.layer == ModuleLayer.DOMAIN
    assert module.domain is None
    assert len(module.granularity_support) == 3  # All granularities
    assert module.extends is None
```

**Step 2: Run test to verify it fails**

Run: `cd /home/hendrik/coding/ai-safety/apart && uv run pytest tests/unit/test_module_loader.py -v`
Expected: FAIL (loader doesn't parse new fields yet)

**Step 3: Write minimal implementation**

Update `_parse_module` method in `src/modules/loader.py` (around line 126):

```python
def _parse_module(self, data: dict, source_name: str) -> BehaviorModule:
    """Parse raw YAML data into a BehaviorModule."""
    name = data.get("name", source_name)

    # Parse new taxonomy fields
    layer_str = data.get("layer", "domain")
    layer_mapping = {
        "meta": ModuleLayer.META,
        "grounding": ModuleLayer.GROUNDING,
        "domain": ModuleLayer.DOMAIN,
        "detail": ModuleLayer.DETAIL,
    }
    layer = layer_mapping.get(layer_str, ModuleLayer.DOMAIN)

    granularity_support_raw = data.get("granularity_support", ["macro", "meso", "micro"])
    granularity_mapping = {
        "macro": Granularity.MACRO,
        "meso": Granularity.MESO,
        "micro": Granularity.MICRO,
    }
    granularity_support = [
        granularity_mapping.get(g, Granularity.MESO)
        for g in granularity_support_raw
    ]

    # ... existing variable/dynamic/constraint parsing ...

    return BehaviorModule(
        name=name,
        description=data.get("description", ""),
        version=data.get("version", "1.0.0"),
        layer=layer,
        domain=data.get("domain"),
        granularity_support=granularity_support,
        extends=data.get("extends"),
        variables=variables,
        dynamics=dynamics,
        constraints=constraints,
        agent_effects=agent_effects,
        config_schema=config_schema,
        requires=data.get("requires", []),
        conflicts_with=data.get("conflicts_with", []),
        simulator_prompt_section=data.get("simulator_prompt_section"),
        agent_prompt_section=data.get("agent_prompt_section"),
        event_types=data.get("event_types", []),
        event_probabilities=data.get("event_probabilities", {}),
    )
```

Also add imports at top of loader.py:
```python
from modules.models import (
    # ... existing imports ...
    ModuleLayer,
    Granularity,
)
```

**Step 4: Run test to verify it passes**

Run: `cd /home/hendrik/coding/ai-safety/apart && uv run pytest tests/unit/test_module_loader.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/modules/loader.py tests/unit/test_module_loader.py
git commit -m "feat(modules): parse layer, granularity_support, extends in loader"
```

---

### Task 1.3: Add Granularity Validation to ModuleLoader

**Files:**
- Modify: `src/modules/loader.py:255-276`
- Test: `tests/unit/test_module_loader.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_module_loader.py`:

```python
def test_loader_validates_extends_dependency(temp_modules_dir):
    """Test loader validates that extended module is loaded."""
    base_yaml = """
module:
  name: economic_base
  description: Base economic module
  layer: domain
"""
    detail_yaml = """
module:
  name: economic_detailed
  description: Detailed economic module
  layer: detail
  extends: economic_base
"""
    (temp_modules_dir / "economic_base.yaml").write_text(base_yaml)
    (temp_modules_dir / "economic_detailed.yaml").write_text(detail_yaml)

    loader = ModuleLoader(temp_modules_dir)

    # Loading detail without base should fail
    from modules.loader import ModuleDependencyError
    with pytest.raises(ModuleDependencyError, match="extends.*economic_base"):
        loader.load_many(["economic_detailed"])

    # Loading both should work
    modules = loader.load_many(["economic_base", "economic_detailed"])
    assert len(modules) == 2


def test_loader_validates_granularity_compatibility(temp_modules_dir):
    """Test loader can check granularity compatibility."""
    macro_only = """
module:
  name: macro_module
  description: Macro only
  granularity_support: [macro]
"""
    micro_only = """
module:
  name: micro_module
  description: Micro only
  granularity_support: [micro]
"""
    (temp_modules_dir / "macro_module.yaml").write_text(macro_only)
    (temp_modules_dir / "micro_module.yaml").write_text(micro_only)

    loader = ModuleLoader(temp_modules_dir)
    modules = loader.load_many(["macro_module", "micro_module"])

    # Check that we can find common granularity (should be none)
    from modules.loader import find_common_granularity
    common = find_common_granularity(modules)
    assert common == []
```

**Step 2: Run test to verify it fails**

Run: `cd /home/hendrik/coding/ai-safety/apart && uv run pytest tests/unit/test_module_loader.py::test_loader_validates_extends_dependency -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Add to `src/modules/loader.py` after `_validate_dependencies` method:

```python
def _validate_dependencies(
    self,
    modules: List[BehaviorModule],
    loaded_names: Set[str]
) -> None:
    """Validate module dependencies."""
    for module in modules:
        # Check required modules
        for required in module.requires:
            if required not in loaded_names:
                raise ModuleDependencyError(
                    f"Module '{module.name}' requires '{required}' which is not loaded. "
                    f"Add '{required}' to your modules list."
                )

        # Check conflicts
        for conflict in module.conflicts_with:
            if conflict in loaded_names:
                raise ModuleDependencyError(
                    f"Module '{module.name}' conflicts with '{conflict}'. "
                    f"These modules cannot be used together."
                )

        # NEW: Check extends dependency
        if module.extends and module.extends not in loaded_names:
            raise ModuleDependencyError(
                f"Module '{module.name}' extends '{module.extends}' which is not loaded. "
                f"Add '{module.extends}' to your modules list."
            )


def find_common_granularity(modules: List[BehaviorModule]) -> List[Granularity]:
    """
    Find granularity levels supported by all modules.

    Args:
        modules: List of modules to check

    Returns:
        List of Granularity values supported by all modules
    """
    if not modules:
        return list(Granularity)

    # Start with first module's support
    common = set(modules[0].granularity_support)

    # Intersect with each subsequent module
    for module in modules[1:]:
        common &= set(module.granularity_support)

    return list(common)
```

**Step 4: Run test to verify it passes**

Run: `cd /home/hendrik/coding/ai-safety/apart && uv run pytest tests/unit/test_module_loader.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/modules/loader.py tests/unit/test_module_loader.py
git commit -m "feat(modules): validate extends dependency and add granularity helpers"
```

---

## Phase 2: Core Module Definitions

### Task 2.1: Create agents_base Module

**Files:**
- Create: `src/modules/definitions/agents_base.yaml`
- Test: `tests/unit/test_core_modules.py` (new file)

**Step 1: Write the failing test**

```python
# tests/unit/test_core_modules.py
"""Tests for core module definitions."""
import pytest
from modules.loader import ModuleLoader
from modules.models import ModuleLayer, Granularity


def test_agents_base_module_loads():
    """Test agents_base module loads correctly."""
    loader = ModuleLoader()
    module = loader.load("agents_base")

    assert module.name == "agents_base"
    assert module.layer == ModuleLayer.DOMAIN
    assert Granularity.MESO in module.granularity_support


def test_agents_base_has_core_variables():
    """Test agents_base provides core agent variables."""
    loader = ModuleLoader()
    module = loader.load("agents_base")

    # Should have agent-scoped communication variables
    agent_vars = [v for v in module.variables if v.scope == "agent"]
    var_names = [v.name for v in agent_vars]

    assert "stance" in var_names or "position" in var_names
```

**Step 2: Run test to verify it fails**

Run: `cd /home/hendrik/coding/ai-safety/apart && uv run pytest tests/unit/test_core_modules.py::test_agents_base_module_loads -v`
Expected: FAIL with "Module 'agents_base' not found"

**Step 3: Write minimal implementation**

```yaml
# src/modules/definitions/agents_base.yaml
module:
  name: agents_base
  version: "1.0.0"
  description: "Foundation module for agent-based simulations. Provides core communication and state tracking."

  layer: domain
  domain: null  # Core module, not domain-specific
  granularity_support: [macro, meso, micro]

  requires: []
  conflicts_with: []

  variables:
    global:
      - name: simulation_turn
        type: count
        default: 0
        description: "Current simulation turn number"

      - name: active_negotiations
        type: list
        default: []
        description: "Currently active negotiations [{parties, topic, status}]"

    agent:
      - name: stance
        type: dict
        default: {}
        description: "Agent's current stance on key issues {issue: position}"

      - name: relationships
        type: dict
        default: {}
        description: "Relationship scores with other agents {agent_name: score}"

      - name: messages_sent
        type: count
        default: 0
        description: "Number of messages sent this simulation"

      - name: messages_received
        type: count
        default: 0
        description: "Number of messages received this simulation"

  dynamics:
    - description: "Agents communicate through structured messages and can update their stances based on interactions"
      priority: 8

    - description: "Relationships between agents evolve based on cooperation, conflict, and communication"
      priority: 7

  constraints:
    - description: "Agents can only directly communicate with agents they have relationships with"
      enforcement: guided

    - description: "Stance changes should be gradual unless triggered by major events"
      enforcement: guided

  agent_effects:
    - description: "Agents perceive other agents through the lens of their relationship scores"
      applies_to: all

    - description: "Prior communication history affects interpretation of new messages"
      applies_to: all
```

**Step 4: Run test to verify it passes**

Run: `cd /home/hendrik/coding/ai-safety/apart && uv run pytest tests/unit/test_core_modules.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/modules/definitions/agents_base.yaml tests/unit/test_core_modules.py
git commit -m "feat(modules): add agents_base core module"
```

---

### Task 2.2: Update economic_base Module with New Schema

**Files:**
- Modify: `src/modules/definitions/economic_base.yaml` (add new fields)
- Test: `tests/unit/test_core_modules.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_core_modules.py`:

```python
def test_economic_base_has_new_schema_fields():
    """Test economic_base has layer and granularity fields."""
    loader = ModuleLoader()
    module = loader.load("economic_base")

    assert module.layer == ModuleLayer.DOMAIN
    assert module.domain == "economic"
    assert Granularity.MESO in module.granularity_support
```

**Step 2: Run test to verify it fails**

Run: `cd /home/hendrik/coding/ai-safety/apart && uv run pytest tests/unit/test_core_modules.py::test_economic_base_has_new_schema_fields -v`
Expected: FAIL (missing new fields)

**Step 3: Write minimal implementation**

Add to `src/modules/definitions/economic_base.yaml` after line 4 (after description):

```yaml
  layer: domain
  domain: economic
  granularity_support: [macro, meso]
```

Also update `supply_chain_base.yaml` to use the same pattern and remove conflict with economic_base since they should work together.

**Step 4: Run test to verify it passes**

Run: `cd /home/hendrik/coding/ai-safety/apart && uv run pytest tests/unit/test_core_modules.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/modules/definitions/economic_base.yaml src/modules/definitions/supply_chain_base.yaml
git commit -m "feat(modules): add new schema fields to economic modules"
```

---

### Task 2.3: Create diplomatic_base Module

**Files:**
- Create: `src/modules/definitions/diplomatic_base.yaml`
- Test: `tests/unit/test_core_modules.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_core_modules.py`:

```python
def test_diplomatic_base_module_loads():
    """Test diplomatic_base module loads correctly."""
    loader = ModuleLoader()
    module = loader.load("diplomatic_base")

    assert module.name == "diplomatic_base"
    assert module.layer == ModuleLayer.DOMAIN
    assert module.domain == "diplomatic"


def test_diplomatic_base_has_alliance_variables():
    """Test diplomatic_base provides alliance tracking."""
    loader = ModuleLoader()
    module = loader.load("diplomatic_base")

    var_names = [v.name for v in module.variables]
    assert "alliances" in var_names or "treaties" in var_names
```

**Step 2: Run test to verify it fails**

Run: `cd /home/hendrik/coding/ai-safety/apart && uv run pytest tests/unit/test_core_modules.py::test_diplomatic_base_module_loads -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```yaml
# src/modules/definitions/diplomatic_base.yaml
module:
  name: diplomatic_base
  version: "1.0.0"
  description: "Foundation for diplomatic simulations - alliances, treaties, reputation, and negotiations"

  layer: domain
  domain: diplomatic
  granularity_support: [macro, meso, micro]

  requires: []
  conflicts_with: []

  variables:
    global:
      - name: active_treaties
        type: list
        default: []
        description: "Active treaties [{parties, type, terms, status}]"

      - name: alliance_blocs
        type: dict
        default: {}
        description: "Named alliance groupings {bloc_name: [members]}"

      - name: diplomatic_incidents
        type: list
        default: []
        description: "Recent diplomatic incidents [{parties, type, severity, step}]"

    agent:
      - name: alliances
        type: dict
        default: {}
        description: "Alliance commitments {agent_name: {type, strength, conditions}}"

      - name: treaties
        type: list
        default: []
        description: "Treaties this agent is party to"

      - name: diplomatic_reputation
        type: scale
        default: 50
        min: 0
        max: 100
        description: "International reputation (0=pariah, 100=highly trusted)"

      - name: grievances
        type: dict
        default: {}
        description: "Unresolved grievances {agent_name: [{issue, severity}]}"

      - name: diplomatic_leverage
        type: scale
        default: 50
        min: 0
        max: 100
        description: "Diplomatic influence and leverage"

  dynamics:
    - description: "Alliances create mutual defense expectations and coordination costs"
      priority: 9
      examples:
        - "Alliance members expect support in conflicts"
        - "Breaking alliance commitments damages reputation severely"

    - description: "Treaties constrain behavior but can be renegotiated or violated"
      priority: 8
      examples:
        - "Trade treaties create economic interdependence"
        - "Arms control treaties limit military options"

    - description: "Reputation affects ability to form new alliances and negotiate"
      priority: 7
      examples:
        - "Low reputation makes partners hesitant"
        - "High reputation enables leadership roles"

    - description: "Grievances accumulate and can trigger escalation if unaddressed"
      priority: 6

  constraints:
    - description: "Alliance strength cannot exceed 100"
      enforcement: hard
      variable: alliances

    - description: "Cannot form alliance with agent you have active conflict with"
      enforcement: soft

    - description: "Treaty violations should damage reputation"
      enforcement: guided

  agent_effects:
    - description: "Allies share information and coordinate responses"
      applies_to: all

    - description: "Agents with grievances are predisposed to conflict"
      applies_to: all

    - description: "High reputation agents are sought as mediators"
      applies_to: all

  event_types:
    - alliance_formed
    - alliance_broken
    - treaty_signed
    - treaty_violated
    - diplomatic_crisis
    - mediation_attempted
    - reputation_damaged
    - reputation_improved

  event_probabilities:
    diplomatic_crisis: 0.05
    mediation_attempted: 0.10
```

**Step 4: Run test to verify it passes**

Run: `cd /home/hendrik/coding/ai-safety/apart && uv run pytest tests/unit/test_core_modules.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/modules/definitions/diplomatic_base.yaml tests/unit/test_core_modules.py
git commit -m "feat(modules): add diplomatic_base module"
```

---

### Task 2.4: Create trust_dynamics Module

**Files:**
- Create: `src/modules/definitions/trust_dynamics.yaml`
- Test: `tests/unit/test_core_modules.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_core_modules.py`:

```python
def test_trust_dynamics_module_loads():
    """Test trust_dynamics module loads correctly."""
    loader = ModuleLoader()
    module = loader.load("trust_dynamics")

    assert module.name == "trust_dynamics"
    assert module.layer == ModuleLayer.DOMAIN
    assert module.domain == "social"


def test_trust_dynamics_has_trust_variables():
    """Test trust_dynamics provides trust tracking."""
    loader = ModuleLoader()
    module = loader.load("trust_dynamics")

    agent_vars = [v for v in module.variables if v.scope == "agent"]
    var_names = [v.name for v in agent_vars]

    assert "trust_scores" in var_names
```

**Step 2: Run test to verify it fails**

Run: `cd /home/hendrik/coding/ai-safety/apart && uv run pytest tests/unit/test_core_modules.py::test_trust_dynamics_module_loads -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```yaml
# src/modules/definitions/trust_dynamics.yaml
module:
  name: trust_dynamics
  version: "1.0.0"
  description: "Trust and relationship dynamics between agents - tracks trust evolution based on actions"

  layer: domain
  domain: social
  granularity_support: [macro, meso, micro]

  requires: []
  conflicts_with: []

  variables:
    global:
      - name: trust_events
        type: list
        default: []
        description: "Recent trust-affecting events [{from, to, type, magnitude, step}]"

    agent:
      - name: trust_scores
        type: dict
        default: {}
        description: "Trust levels for other agents {agent_name: score} where score is -100 to 100"

      - name: trustworthiness
        type: scale
        default: 50
        min: 0
        max: 100
        description: "How trustworthy this agent is perceived to be generally"

      - name: trust_history
        type: dict
        default: {}
        description: "History of trust changes {agent_name: [{change, reason, step}]}"

      - name: betrayal_count
        type: count
        default: 0
        description: "Number of times this agent has betrayed commitments"

  dynamics:
    - description: "Trust is earned slowly through consistent behavior but lost quickly through betrayal"
      priority: 9
      examples:
        - "Keeping commitments gradually builds trust (+5 to +10 per positive interaction)"
        - "Single betrayal can destroy years of trust building (-30 to -50)"

    - description: "Trust affects willingness to cooperate and share information"
      priority: 8
      examples:
        - "High trust enables sensitive negotiations"
        - "Low trust leads to verification demands and hedging"

    - description: "Trust can be rebuilt after betrayal but requires sustained effort"
      priority: 6
      examples:
        - "Apologies and compensation can begin repair"
        - "Repeated positive actions needed over time"

    - description: "Third-party observations of behavior affect trust calculations"
      priority: 5

  constraints:
    - description: "Trust scores bounded between -100 (total distrust) and 100 (complete trust)"
      enforcement: hard
      variable: trust_scores

    - description: "Trust changes should be proportional to action significance"
      enforcement: guided

    - description: "Betrayal of high-trust relationship has larger negative impact"
      enforcement: guided

  agent_effects:
    - description: "Low trust agents are excluded from sensitive discussions"
      applies_to: all

    - description: "High trust enables more efficient cooperation (less verification needed)"
      applies_to: all

    - description: "Agents with betrayal history face skepticism regardless of current behavior"
      applies_to: all

  event_types:
    - trust_increased
    - trust_decreased
    - trust_betrayed
    - trust_restored
    - commitment_kept
    - commitment_broken

  event_probabilities:
    trust_betrayed: 0.02
    commitment_broken: 0.05
```

**Step 4: Run test to verify it passes**

Run: `cd /home/hendrik/coding/ai-safety/apart && uv run pytest tests/unit/test_core_modules.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/modules/definitions/trust_dynamics.yaml tests/unit/test_core_modules.py
git commit -m "feat(modules): add trust_dynamics social module"
```

---

### Task 2.5: Update spatial_graph (territory_graph) Module

**Files:**
- Modify: `src/modules/definitions/territory_graph.yaml`
- Test: `tests/unit/test_core_modules.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_core_modules.py`:

```python
def test_territory_graph_is_grounding_module():
    """Test territory_graph is properly categorized as grounding."""
    loader = ModuleLoader()
    module = loader.load("territory_graph")

    assert module.layer == ModuleLayer.GROUNDING
    assert module.domain is None  # Grounding modules are domain-agnostic
```

**Step 2: Run test to verify it fails**

Run: `cd /home/hendrik/coding/ai-safety/apart && uv run pytest tests/unit/test_core_modules.py::test_territory_graph_is_grounding_module -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Update `src/modules/definitions/territory_graph.yaml` to add after description:

```yaml
  layer: grounding
  domain: null
  granularity_support: [macro, meso, micro]
```

**Step 4: Run test to verify it passes**

Run: `cd /home/hendrik/coding/ai-safety/apart && uv run pytest tests/unit/test_core_modules.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/modules/definitions/territory_graph.yaml
git commit -m "feat(modules): update territory_graph as grounding module"
```

---

### Task 2.6: Verify All 5 Core Modules Work Together

**Files:**
- Test: `tests/unit/test_core_modules.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_core_modules.py`:

```python
def test_all_core_modules_compose():
    """Test all 5 core modules can be loaded and composed together."""
    from modules.composer import ModuleComposer

    loader = ModuleLoader()
    core_modules = [
        "agents_base",
        "territory_graph",
        "economic_base",
        "diplomatic_base",
        "trust_dynamics",
    ]

    # Should load without dependency errors
    modules = loader.load_many(core_modules)
    assert len(modules) == 5

    # Should compose without errors
    composer = ModuleComposer()
    # Note: territory_graph needs map_file config, so we skip config for this test
    # Just verify modules load and have no conflicts

    # Verify no conflicts between core modules
    all_conflicts = set()
    for m in modules:
        all_conflicts.update(m.conflicts_with)

    core_names = set(core_modules)
    conflicts_with_core = all_conflicts & core_names
    assert conflicts_with_core == set(), f"Core modules conflict with each other: {conflicts_with_core}"


def test_core_modules_share_meso_granularity():
    """Test all core modules support meso granularity."""
    from modules.loader import find_common_granularity

    loader = ModuleLoader()
    core_modules = [
        "agents_base",
        "territory_graph",
        "economic_base",
        "diplomatic_base",
        "trust_dynamics",
    ]

    modules = loader.load_many(core_modules)
    common = find_common_granularity(modules)

    assert Granularity.MESO in common, "All core modules should support meso granularity"
```

**Step 2: Run test to verify it fails**

Run: `cd /home/hendrik/coding/ai-safety/apart && uv run pytest tests/unit/test_core_modules.py::test_all_core_modules_compose -v`
Expected: May fail due to conflicts in existing modules

**Step 3: Fix any remaining conflicts**

Update module definitions to remove conflicts between core modules. The supply_chain_base.yaml currently conflicts with territory_graph - we need to either:
1. Keep supply_chain_base separate from economic_base (it's a detail module)
2. Or use economic_base as the core economic module

For this implementation, economic_base is the core module, supply_chain_base is a detail extension.

**Step 4: Run test to verify it passes**

Run: `cd /home/hendrik/coding/ai-safety/apart && uv run pytest tests/unit/test_core_modules.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/modules/definitions/*.yaml tests/unit/test_core_modules.py
git commit -m "feat(modules): ensure 5 core modules work together without conflicts"
```

---

## Phase 3: Validation Pipeline

### Task 3.1: Create Scenario Validator

**Files:**
- Create: `src/utils/scenario_validator.py`
- Test: `tests/unit/test_scenario_validator.py`

**Step 1: Write the failing test**

```python
# tests/unit/test_scenario_validator.py
"""Tests for scenario validation."""
import pytest
from utils.scenario_validator import ScenarioValidator, ValidationError


def test_validator_checks_required_agent_prompt_sections():
    """Test validator checks for required sections in agent prompts."""
    validator = ScenarioValidator()

    valid_prompt = """
# OBJECTIVES
- Goal 1
- Goal 2

# CONSTRAINTS
- Constraint 1

# INFORMATION ACCESS
- Full: own data
- Partial: enemy data
"""

    invalid_prompt = """
Just some text without required sections.
"""

    assert validator.validate_agent_prompt(valid_prompt) == []
    errors = validator.validate_agent_prompt(invalid_prompt)
    assert len(errors) > 0
    assert any("OBJECTIVES" in e for e in errors)


def test_validator_checks_granularity_compatibility():
    """Test validator checks module granularity compatibility."""
    validator = ScenarioValidator()

    config = {
        "meta": {
            "granularity": "micro",
        },
        "modules": ["economic_base"],  # Only supports macro, meso
    }

    errors = validator.validate_granularity(config)
    assert len(errors) > 0
    assert any("granularity" in e.lower() for e in errors)
```

**Step 2: Run test to verify it fails**

Run: `cd /home/hendrik/coding/ai-safety/apart && uv run pytest tests/unit/test_scenario_validator.py -v`
Expected: FAIL with "No module named 'utils.scenario_validator'"

**Step 3: Write minimal implementation**

```python
# src/utils/scenario_validator.py
"""
Scenario validation for AI-driven simulation generation.

Validates configuration against schema, checks module compatibility,
and ensures agent prompts contain required sections.
"""
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from modules.loader import ModuleLoader, find_common_granularity
from modules.models import Granularity


class ValidationError(Exception):
    """Validation failed with one or more errors."""
    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(f"Validation failed with {len(errors)} error(s):\n" + "\n".join(f"  - {e}" for e in errors))


@dataclass
class ValidationResult:
    """Result of validation."""
    valid: bool
    errors: List[str]
    warnings: List[str]


# Required sections in agent system prompts
REQUIRED_PROMPT_SECTIONS = [
    "OBJECTIVES",
    "CONSTRAINTS",
    "INFORMATION ACCESS",
]


class ScenarioValidator:
    """
    Validates scenario configurations for AI-driven simulations.

    Checks:
    - Required fields present
    - Module compatibility (dependencies, conflicts, granularity)
    - Agent prompt structure (required sections)
    - Variable types match definitions
    - Experiment conditions reference valid paths
    """

    def __init__(self, modules_dir: Optional[str] = None):
        self.loader = ModuleLoader(modules_dir) if modules_dir else ModuleLoader()

    def validate_agent_prompt(self, prompt: str) -> List[str]:
        """
        Validate agent system prompt has required sections.

        Args:
            prompt: Agent system prompt text

        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        prompt_upper = prompt.upper()

        for section in REQUIRED_PROMPT_SECTIONS:
            # Check for section header (with or without #)
            if f"# {section}" not in prompt_upper and f"#{section}" not in prompt_upper and section not in prompt_upper:
                errors.append(f"Agent prompt missing required section: {section}")

        return errors

    def validate_granularity(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate that selected granularity is supported by all modules.

        Args:
            config: Scenario configuration dict

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        meta = config.get("meta", {})
        selected_granularity_str = meta.get("granularity", "meso")

        granularity_mapping = {
            "macro": Granularity.MACRO,
            "meso": Granularity.MESO,
            "micro": Granularity.MICRO,
        }
        selected = granularity_mapping.get(selected_granularity_str)

        if not selected:
            errors.append(f"Invalid granularity: {selected_granularity_str}")
            return errors

        module_names = config.get("modules", [])
        if not module_names:
            return errors

        try:
            modules = self.loader.load_many(module_names)
            common = find_common_granularity(modules)

            if selected not in common:
                unsupported = [
                    m.name for m in modules
                    if selected not in m.granularity_support
                ]
                errors.append(
                    f"Granularity '{selected_granularity_str}' not supported by modules: {', '.join(unsupported)}"
                )
        except Exception as e:
            errors.append(f"Error loading modules for granularity check: {e}")

        return errors

    def validate(self, config: Dict[str, Any]) -> ValidationResult:
        """
        Perform full validation of scenario configuration.

        Args:
            config: Full scenario configuration dict

        Returns:
            ValidationResult with valid flag, errors, and warnings
        """
        errors = []
        warnings = []

        # Check granularity compatibility
        errors.extend(self.validate_granularity(config))

        # Check agent prompts
        for agent in config.get("agents", []):
            prompt = agent.get("system_prompt", "")
            agent_name = agent.get("name", "unknown")
            prompt_errors = self.validate_agent_prompt(prompt)
            for e in prompt_errors:
                errors.append(f"Agent '{agent_name}': {e}")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def validate_or_raise(self, config: Dict[str, Any]) -> None:
        """
        Validate configuration and raise if invalid.

        Args:
            config: Full scenario configuration dict

        Raises:
            ValidationError: If validation fails
        """
        result = self.validate(config)
        if not result.valid:
            raise ValidationError(result.errors)
```

**Step 4: Run test to verify it passes**

Run: `cd /home/hendrik/coding/ai-safety/apart && uv run pytest tests/unit/test_scenario_validator.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/utils/scenario_validator.py tests/unit/test_scenario_validator.py
git commit -m "feat(validation): add scenario validator with prompt and granularity checks"
```

---

### Task 3.2: Add Variable Type Validation

**Files:**
- Modify: `src/utils/scenario_validator.py`
- Test: `tests/unit/test_scenario_validator.py`

**Step 1: Write the failing test**

Add to `tests/unit/test_scenario_validator.py`:

```python
def test_validator_checks_variable_types():
    """Test validator checks agent variable types match definitions."""
    validator = ScenarioValidator()

    config = {
        "agent_vars": {
            "trust_level": {
                "type": "percent",
                "default": 50,
                "min": 0,
                "max": 100,
            }
        },
        "agents": [
            {
                "name": "TestAgent",
                "system_prompt": "# OBJECTIVES\n- Test\n# CONSTRAINTS\n- None\n# INFORMATION ACCESS\n- Full",
                "variables": {
                    "trust_level": "high",  # Wrong type! Should be int
                }
            }
        ]
    }

    result = validator.validate(config)
    assert not result.valid
    assert any("trust_level" in e and "type" in e.lower() for e in result.errors)
```

**Step 2: Run test to verify it fails**

Run: `cd /home/hendrik/coding/ai-safety/apart && uv run pytest tests/unit/test_scenario_validator.py::test_validator_checks_variable_types -v`
Expected: FAIL (validator doesn't check variable types yet)

**Step 3: Write minimal implementation**

Add to `ScenarioValidator` class in `src/utils/scenario_validator.py`:

```python
def validate_variable_types(self, config: Dict[str, Any]) -> List[str]:
    """
    Validate agent variable values match their type definitions.

    Args:
        config: Scenario configuration dict

    Returns:
        List of error messages
    """
    errors = []

    # Get variable definitions
    agent_var_defs = config.get("agent_vars", {})

    # Type checking functions
    type_checkers = {
        "int": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "float": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "bool": lambda v: isinstance(v, bool),
        "percent": lambda v: isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 100,
        "scale": lambda v: isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 100,
        "count": lambda v: isinstance(v, int) and not isinstance(v, bool) and v >= 0,
        "dict": lambda v: isinstance(v, dict),
        "list": lambda v: isinstance(v, list),
    }

    for agent in config.get("agents", []):
        agent_name = agent.get("name", "unknown")
        agent_vars = agent.get("variables", {})

        for var_name, value in agent_vars.items():
            if var_name not in agent_var_defs:
                errors.append(f"Agent '{agent_name}': undefined variable '{var_name}'")
                continue

            var_def = agent_var_defs[var_name]
            var_type = var_def.get("type", "int")

            checker = type_checkers.get(var_type)
            if checker and not checker(value):
                errors.append(
                    f"Agent '{agent_name}': variable '{var_name}' has wrong type. "
                    f"Expected {var_type}, got {type(value).__name__} ({value})"
                )

    return errors
```

Also update the `validate` method to call this:

```python
def validate(self, config: Dict[str, Any]) -> ValidationResult:
    errors = []
    warnings = []

    # ... existing checks ...

    # Check variable types
    errors.extend(self.validate_variable_types(config))

    return ValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )
```

**Step 4: Run test to verify it passes**

Run: `cd /home/hendrik/coding/ai-safety/apart && uv run pytest tests/unit/test_scenario_validator.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/utils/scenario_validator.py tests/unit/test_scenario_validator.py
git commit -m "feat(validation): add variable type checking"
```

---

## Phase 4: Experiment Runner (Deferred)

> **Note:** Phase 4 (Experiment Runner) and Phase 5 (Claude Skill) are deferred to a follow-up implementation plan. This plan covers the foundation needed for those features.

The experiment runner will need:
- Multi-condition execution with variable/prompt modifications
- Run aggregation and comparison
- Statistical analysis helpers
- Results persistence and retrieval

---

## Summary

This plan covers:

| Phase | Tasks | Purpose |
|-------|-------|---------|
| **Phase 1** | 1.1-1.3 | Enhanced module schema with layer/granularity |
| **Phase 2** | 2.1-2.6 | 5 core module definitions |
| **Phase 3** | 3.1-3.2 | Validation pipeline |

**Total tasks:** 11 tasks across 3 phases

**Estimated commits:** 11 atomic commits

**Test coverage:** Each task includes tests before implementation (TDD)

---

## Next Steps After This Plan

1. **Phase 4: Experiment Runner** - Multi-condition execution, statistical comparison
2. **Phase 5: Claude Skill** - `/simulation` skill with schema docs and commands
3. **Phase 6: Integration** - End-to-end testing of AI-driven scenario generation
