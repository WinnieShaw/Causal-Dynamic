# Causal-Dynamic Interpretability Metrics — Code Repository

This repository contains the core implementation of the causal-dynamic interpretability metrics proposed in our paper, including metric computation and evaluation pipelines.

Due to size and license constraints, no pretrained language model weights are included. This repository mainly serves to demonstrate how the proposed metrics are operationalized and evaluated.

---

## Repository Structure

### Metric Computation

- `cci_computation.py`  
  Implementation of **CCI (Contextual Causal Influence)**, which measures the causal influence of prefix context on current token generation and its dynamic evolution across token positions.

- `qaci_computation.py`  
  Implementation of **QACI (Question–Answer Causal Influence)**, which quantifies the causal alignment between questions and generated answers.

- `run_cci.py`  
  Script for running CCI computation over a question set and collecting aggregated statistics.

- `run_qaci.py`  
  Script for running QACI computation using the same evaluation protocol.

- `run_pmi.py`  
  Implementation of a PMI-based correlation baseline for comparison with causal metrics.

---

### Question Pools

All evaluation questions are provided under `question_pools/` and organized in two complementary ways.

#### By Difficulty Level

Path: `question_pools/difficulty_levels/`

- `level1.json` — factual and surface-level questions  
- `level2.json` — simple relational or causal questions  
- `level3.json` — multi-step reasoning and mechanism-based questions  
- `level4.json` — system-level and process-oriented questions  
- `level5.json` — deep causal chains and cross-domain analytical questions  

These sets are used to analyze how causal dependence varies with cognitive complexity.

#### By Input Length

Path: `question_pools/length_levels/`

- `level1.json` — very short inputs  
- `level2.json` — short inputs  
- `level3.json` — medium-length inputs  
- `level4.json` — long-context inputs  
- `level5.json` — very long-context inputs  

These sets are used to study how causal constraints change with context scale during generation.

---

### Optional Fine-tuning Scripts

Located in `Fine-tune/`:

- `Qwen-train.py` — fine-tuning script for Qwen2.5-Instruct-7B model  
- `T5-base-train.py` — fine-tuning script for T5-base model  

These scripts were used in some experiments to obtain task-adapted models, but they are not required for understanding or computing the proposed metrics.

---

## Usage Notes

This repository focuses on metric implementation and evaluation protocols.

To run the full pipeline, users need to provide their own model checkpoints compatible with the HuggingFace Transformers interface.

Model training and large-scale experiments are not included in this repository.

---


