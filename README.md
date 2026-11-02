# DiffusionLLM for Tabular Data Imputation: King County House Dataset

## 📌 Overview
This project evaluates the efficacy of reasoning Large Language Models (LLMs)—specifically **Mercury-2**, a Diffusion Reasoning LLM by Inception Labs—for imputing missing tabular data. Using the King County House Data (21,611 records across 21 features), the study compares LLM-based semantic reasoning against classical statistical baselines, iterative ML algorithms, and deep generative models (CTGAN, TVAE) under severe missingness conditions.

---

## 🚨 Problem & Solution
* **The Problem:** Complex tabular datasets in real-world scenarios frequently suffer from missing values. Traditional statistical/distance-based imputation methods and deep generative models struggle when missingness is high, failing to preserve semantic domain relationships.
* **The Solution:** A semantic imputation pipeline leveraging a Diffusion Reasoning LLM with dynamic few-shot prompting. By understanding housing market domain semantics rather than relying strictly on Euclidean proximity, the LLM restores missing feature values with superior fidelity.

---

## 🔬 Methodology & Experimental Setup
* **Preprocessing & Feature Engineering:**
  * Split `bathrooms` into `bathroom_count` (quantity) and `type_bathroom` (quality: full, 3/4, half, 1/4).
  * Decomposed `date` into `sale_year` and `sale_month` to capture seasonality.
  * Isolated the `price` target variable prior to missingness generation.
* **Data Partitioning (40 / 30 / 30):**
  * **Imputation Train (40%):** Clean support set for learning restoration patterns.
  * **Imputation Test (30%):** Subjected to synthetic corruption.
  * **Holdout (30%):** Clean validation set for downstream ML efficiency evaluation.
* **Missingness Simulation:**
  * Generated 18 corrupted datasets across MCAR, MAR, and MNAR mechanisms at 10%, 25%, and 40% missingness levels.
  * Protected 40% of features as immutable **Anchor variables**.

---

## ⚙️ Imputation Pipelines
1. **Baselines & Classical Methods:** Mean, Median, KNN ($k=5$), MICE (Bayesian Ridge), and MissForest (ExtraTrees).
2. **Deep Generative Tabular Models:** TVAE and CTGAN fitted on clean data using a custom `TabularGenerativeImputer` wrapper (matching synthetic rows via minimum Euclidean distance on observed features).
3. **DLLM Semantic Pipeline (Mercury-2):**
   * Specialized domain System Prompt.
   * **Dynamic Few-Shot Prompting:** Selects 4 clean contextual records from the training set per corrupted sample.
   * **Anti-Target Leakage:** Systematically strips `price` from few-shot context examples.
   * **Asynchronous Execution:** 6-thread parallelization (`ThreadPoolExecutor`) with dynamic API key rotation for rate limits (429) and credit errors.

---

## 📊 Experimental Results (MAR 40% Extreme Stress Scenario)
Evaluated using downstream Machine Learning Efficiency via an **XGBoost** regressor predicting house prices on reconstructed datasets:

| Imputation Method | Model Category | R² Score (↑) | MAE ($ ↓) |
| :--- | :--- | :---: | :---: |
| **Clean Baseline (Oracle)** | Control Oracle | **0.880** | **$71,859** |
| **DLLM (Mercury-2)** | Diffusion Reasoning LLM | **0.749** | **$112,397** |
| **MICE** | Iterative Statistical | 0.700 | $121,903 |
| **MissForest** | Iterative Tree-based | 0.694 | $123,697 |
| **CTGAN** | Deep Generative GAN | 0.637 | $135,190 |
| **TVAE** | Deep Tabular VAE | 0.589 | $146,292 |
| **Mean / Median** | Simple Statistical | ~0.590 | >$145,000 |

---

## 🚀 Key Takeaways & Advantages
* **Semantic Reasoning vs. Euclidean Distance:** Mercury-2 handles severe missingness significantly better than statistical or deep generative baselines by understanding feature semantics.
* **Resilience to Mode Collapse:** Unlike TVAE/CTGAN, which degrade due to covariance matrix breakdown under 40% MAR missingness, the LLM pipeline maintains high structural consistency.
* **Zero Target Leakage:** Strict target isolation ensures unbiased, highly realistic tabular reconstruction.
