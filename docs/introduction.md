# Introduction

Recent work demonstrating strategic deception in LLM agents playing Diplomacy and scenario forecasting efforts like AI 2027 highlight the urgent need for controlled frameworks to evaluate multi-agent AI behaviors systematically. We introduce **apart**, a hybrid multi-agent orchestration framework combining configuration-driven game mechanics with LLM-powered simulation engines, enabling systematic comparison of heterogeneous agent-model combinations (Google Gemini, OpenAI, local Ollama models) across configurable scenarios while maintaining computational efficiency and reproducibility. The framework incorporates post-hoc behavioral analysis detecting dangerous patterns (power-seeking, deception, rule exploitation) through lightweight signal collection and LLM-as-judge evaluation, providing quantitative safety scoring with minimal performance impact (<5ms per step).

## References

- [AI Diplomacy Experiments](https://every.to/diplomacy) - Evaluation of strategic deception in LLM agents
- [AI 2027](https://ai-2027.com/) - Scenario forecasting for advanced AI systems
