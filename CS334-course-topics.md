# CS334 / EE402 — Principles and Techniques of Data Science

Fall 2026–27 study reference. The five modules below follow the official course outline. Lecture slides and assigned readings add detail where the outline gives only an umbrella topic. The lecture filenames and slide numbers sometimes differ.

## Course overview

- What data science is, why it matters, and why it became widespread.
- Data science lifecycle: problem formulation, data generation and acquisition, quality assessment and cleaning, exploratory data analysis, modeling, evaluation, interpretation, and communication.
- Repeatable data pipelines and hands-on analysis with Python, NumPy, pandas, and Jupyter notebooks.
- Descriptive, causal, inferential, and predictive questions. Evidence-based impact evaluation and decision-making.

## Module 1 — Descriptive statistics, data acquisition, and tools

- Descriptive statistics and distributions. Measures of center and spread, means and medians, quantiles, and how summaries or misleading comparisons can hide important variation.
- Population, sample, census, and sampling frame. Sources of sampling bias, including coverage, selection, and nonresponse.
- Probability and non-probability samples. Convenience sampling, quota sampling, simple random sampling, stratified sampling, and cluster sampling. Election polling as a case study.
- Structured and unstructured data. Quantitative versus qualitative, discrete versus continuous, and nominal, ordinal, interval, and ratio measurement scales.
- pandas Series and DataFrames. Loading data, inspecting rows and columns, indexing, selecting, filtering, sorting, and creating columns.
- Data aggregation and the split–apply–combine workflow. Grouping with `groupby`, applying functions with `apply`, and lambda functions. The California baby-names example illustrates grouping and aggregation.

## Module 2 — Data cleaning, exploratory data analysis, and visualization

- Transforming raw data into a form suitable for analysis. Recognizing anomalies, missing data, inconsistent values, and data-quality problems.
- Five dimensions of data quality for EDA: structure, granularity, scope, temporality, and faithfulness.
- Data types and file formats, including JSON. Table keys and joins. Checking what one row represents before merging or interpreting data.
- Visual encoding and choosing a chart for a variable type or comparison. Distributions, bar charts, histograms, box plots, and scatterplots. Clear scales and axes, comparisons across groups, and misleading displays.
- Scale and functional transformations, including log and log–log plots and the Tukey–Mosteller bulge diagram. Conditioning to compare groups, and smoothing to reveal patterns.
- Text mining and string manipulation. Cleaning, splitting, and extracting fields from text with Python string methods.
- Regular expressions: matching and extraction, character classes and negation, alternation, repetition and quantifiers, anchors, and groups.

## Module 3 — Experiments, causality, and statistical inference

### Experiments and causal identification

- Association versus causation. Observational studies, randomized controlled trials, treatment and control groups, random assignment, and caveats in controlled experiments.
- Counterfactual reasoning and constructing a comparison group. Potential outcomes, observed versus unobserved outcomes, individual and average treatment effects, and the fundamental problem of causal inference.
- Quasi-experimental designs: pre–post comparisons, difference-in-differences (DiD) and parallel trends, regression discontinuity design (RDD), and instrumental variables (IV).
- IV assumptions: relevance, exogeneity, and exclusion. Treatment assignment versus take-up.
- Structural causal models (SCMs) and causal directed acyclic graphs (DAGs). Directed paths, confounders, mediators, colliders, chains, forks, and unobserved confounding.
- Conditional independence and d-separation. Open and blocked paths, conditioning on a collider or its descendant, and the consequences of adjusting for post-treatment variables.
- Back-door criterion and adjustment. Identifying an adjustment set that blocks noncausal paths without conditioning on descendants of the treatment.
- Front-door criterion and adjustment. Identifying effects through a mediator under the required path conditions when direct back-door adjustment is unavailable. Smoking, tar, and lung cancer as an illustrative example.
- Observation versus intervention, conditional probabilities versus intervention probabilities, and Pearl’s `do` operator. Modifying a causal graph under intervention. The ladder of causation: association, intervention, and counterfactuals.
- Evidence-based policy and product experimentation. Case studies named in the outline include the remedial tutoring program and Uber Express POOL.

### Statistical inference and experimental design

- Population and sample, random sampling, random variables, probability distributions, functions of random variables, and expectation.
- Estimation, sampling variability, bounds, and confidence intervals. Assessing a model by comparing observed data against simulated outcomes.
- Hypothesis testing: null and alternative hypotheses, test statistic, null distribution, simulation under the null, p-values, and interpreting evidence. Case studies include Mendelian inheritance and *United States v. Kristen Gilbert*.
- A/B testing and comparing samples. Permutation tests, repeated random sampling, bootstrap resampling, and bootstrap confidence intervals.
- Sample size, statistical power, allocation across groups, and calculating power in practice. Type I and Type II errors and the role of effect size and variability. Power under clustered designs.

## Module 4 — Machine learning

- Models and estimation: defining a model, choosing a loss function, minimizing empirical loss, making predictions, and evaluating generalization.
- Constant models and loss functions, including squared and absolute loss. Mean and median as optimal constant predictions under their respective losses.
- Simple and multiple linear regression, fitted values, residuals, and regression evaluation with root mean squared error (RMSE) and multiple R².
- Ordinary least squares (OLS): design matrix, intercept column, response and parameter vectors, squared residual norm, and geometric projection onto the feature column space. Orthogonality, normal equations, and solving for the optimal coefficients.
- Optimization: derivatives, chain rule, convexity, gradients, and gradient descent.
- Feature engineering. Prediction risk, model bias, model variance, observation variance, and the bias–variance tradeoff.
- Train–test split, k-fold cross-validation, generalization to unseen data, and regularization with ridge regression and LASSO.
- Classification and logistic regression: decision rules, evaluation metrics, linear separability, and regularization. Fake-news classification case study.

## Module 5 — Big data and ethics

- Data ethics and its role throughout the data science lifecycle. Fairness in machine learning.
- Scalable data collection, storage, and processing. Big data’s four Vs: volume, velocity, variety, and veracity.
- Distributed file systems, MapReduce, Apache Spark, and PySpark. Hands-on PySpark demonstration.

## Source guide

- **Full course scope:** `CS334F26CourseOutline.pdf` — course description, learning outcomes, and lecture schedule.
- **Causal inference:** `Lec19.pdf`, `Lec20.pdf`, `Backdoor-and-Frontdoor-Criterion.pdf`, `D-Separation Example - Causality_ Models, Reasoning and Inference _Judea Pearl_.pdf`, `On-probabilities-and-causation.pdf`, and `Probabilities-and-Causation-and-Do-Operator.pdf`.
- **Modeling and regression:** `Lec22.pdf`, `lec23.pdf`, and `lec24.pdf`.
- **Earlier topic detail:** accompanying course slides in the lecture folder, including the sampling, pandas, cleaning, visualization, text, experiments, inference, and power lectures.
