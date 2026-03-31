# AStats: GSoC Project Synopsis

*An Agentic-AI Engine for Applied Statistical Workflows*

---

## 🛑 The Problem: The Gap in Agentic Statistics

Informal use suggests that modern LLMs, accessed via agentic systems, have reached a point where they can explore large datasets with human guidance. However, current global solutions struggle with:

1. **The Tooling Paradox:** Statistical practitioners often rely on recipe-driven methods (e.g., **JASP, Jamovi**) to guide their workflows. Most generic AI agents today are "black boxes" that generate code without following these robust, validated practices.
2. **Language Asymmetry:** There is a significant trade-off in the open ecosystem: **R** is statistically more sophisticated but often less familiar to LLMs' training data, while **Python** is widely understood by models but can lack the nuance of specialized R libraries.
3. **Hardware & Cost Constraints:** High-end commercial models are expensive and the SDKs for these models (like LangChain) are heavy and prone to crashing on academic **Slurm cluster** head nodes.

---

## ⚡ Our Solution: The Gen-3 Architecture

AStats (standing for **Autonomous, Augmented, Applied Statistics**) is a from-scratch harness designed to define and implement "best practice" workflows for statistical exploration.

1. **A Structured Harness for Auto-Discovery:** AStats begins with **data auto-discovery and summarization**. By profiling datasets before analysis, it ensures the agent understands the data constraints—matching the "discovery-first" approach of professional practitioners.
2. **Hybrid Language Orchestration:** We bridge the Python-R gap. AStats selects the best specialist for the task, leveraging Python's model-familiarity for cleanup and R's sophistication for complex statistical modeling.
3. **Fine-Tunable, Open-Weight First:** To reduce cost and increase predictability, AStats is built to support **open-weight models** (e.g., Llama 3.2:1B) for specialized pipelines. Our custom **Zero-SDK Router** ensures these models can run locally and efficiently.
4. **Human-Verified Rigor (The Critic Agent):** Following the "Recipe-driven" philosophy, our **Statistical Critic Agent** acts as the human-in-the-loop surrogate. It verifies that code follows valid statistical recipes (checking normality, homoscedasticity, etc.) before results are presented to the practitioner for final verification.
5. **Standalone Professional Reporting:** Results are presented as zero-dependency HTML reports with embedded visualizations, allowing practitioners to verify exploratory and confirmatory analyses in a single, high-integrity document.

---

### The Goal
AStats isn't just about code generation—it's about building the harness for **Robust Statistical Practice** in the age of agentic AI.
