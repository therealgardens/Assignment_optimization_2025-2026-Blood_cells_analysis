# Multiclass Logistic Regression: Gradient Descent vs. Block-Coordinate Methods

**Optimization for Data Science · University of Padua · Group project (2025–2026)**

This project studies **how the choice of optimisation algorithm affects training efficiency and classification performance** in multiclass logistic regression. We implement and compare full-batch Gradient Descent (GD) with two variants of Block-Coordinate Gradient Descent (BCGD), using **synthetic data, handwritten digits (MNIST), and blood-cell images**.

The focus is not only predictive accuracy: we examine **convergence behaviour, computational cost, feature preprocessing, and sensitivity to step-size selection**.

## Optimisation problem

Given a feature matrix $A \in \mathbb{R}^{m\times d}$, class labels $y_i \in \{1,\ldots,K\}$, and parameter matrix $W \in \mathbb{R}^{d\times K}$, we minimise the multiclass negative log-likelihood:

$$
\min_{W\in\mathbb{R}^{d\times K}} f(W)
= -\sum_{i=1}^{m}\log\left(
\frac{\exp(a_i^\top w_{y_i})}
{\sum_{c=1}^{K}\exp(a_i^\top w_c)}
\right).
$$

With $P=\operatorname{softmax}(AW)$ and the one-hot label matrix $Y$, the gradient can be calculated in vectorised form as

$$
\nabla f(W) = A^\top(P-Y).
$$

This formulation allows the optimisation methods to operate efficiently using NumPy matrix operations.

## Methods compared

| Method | Update strategy | Main consideration |
| --- | --- | --- |
| **Gradient Descent (GD)** | Updates the entire parameter matrix using the full gradient | Simple baseline, but each iteration can be costly |
| **Randomised BCGD** | Updates one randomly selected row block at a time | Reduces the work performed per update |
| **Gauss–Southwell BCGD** | Updates the row block with the largest gradient norm | Prioritises promising directions; requires careful step-size selection |

An additional implementation feature is the **cached-score oracle** for Gauss–Southwell BCGD. Instead of recomputing $AW$ after every row-block update, the cached score matrix $S=AW$ is updated using only the affected column of $A$. The report derives a reduction of the score-update computation from $O(mKd)$ to $O(mK)$.

## Datasets and preprocessing

| Dataset | Description | Role in the experiments |
| --- | --- | --- |
| **Synthetic** | 1,000 generated examples, 1,000 features and 50 classes | Controlled optimisation benchmark and generalisation analysis |
| **MNIST** | Handwritten digit images (28 × 28 pixels; 10 classes) | Standard image-classification benchmark |
| **Blood Cell Images** | 17,092 images across 8 cell categories | More challenging real-world classification task |

For the blood-cell dataset, images were converted to **grayscale** and resized to **64 × 64** pixels, giving 4,096 initial features. We investigated dimensionality reduction with **PCA** and **RBF-kernel PCA**. A 150-component PCA representation retained **74.2%** of the reported variance and was selected for the experiments.

The blood-cell dataset is described in the report and available from [Kaggle](https://www.kaggle.com/datasets/unclesamulus/blood-cells-image-dataset).

## Experimental results

The following figures are reported in the final project report for a convergence tolerance of $10^{-4}$ and a maximum of **3,000 iterations**. Accuracy is measured on held-out test data; times are reported CPU times in seconds.

| Dataset | GD test accuracy | Randomised BCGD | Gauss–Southwell BCGD |
| --- | ---: | ---: | ---: |
| Synthetic | 15.7% | **16.3%** | 15.8% |
| MNIST | 89.5% | 89.6% | **89.7%** |
| Blood Cell Images | 61.0% | 66.5% | **68.4%** |

| Dataset | GD time (s) | Randomised BCGD time (s) | Gauss–Southwell time (s) |
| --- | ---: | ---: | ---: |
| Synthetic | 349 | 282 | **242** |
| MNIST | 1,330 | 1,172 | **437** |
| Blood Cell Images | 9,464 | 4,931 | **4,865** |

**Key findings**

- **Blood-cell classification:** Gauss–Southwell BCGD achieves **68.4% test accuracy**, an improvement of **7.4 percentage points** over standard GD, while taking approximately **49% less reported CPU time**.
- **MNIST:** all three approaches attain around **89.5–89.7% test accuracy**; Gauss–Southwell is approximately **3× faster** than standard GD in the reported runs.
- **Synthetic data:** training accuracies are high but test accuracies remain around 16%, indicating substantial overfitting and weak generalisation in the constructed benchmark.
- **Step-size sensitivity:** Gauss–Southwell can converge particularly quickly with a suitable step size, but its performance is sensitive to poorly calibrated learning rates.

These are outcomes of the documented experimental setup, **not universal performance guarantees**.

## Repository structure

```text
.
├── data/             # Data files and preparation assets
├── methods/          # Optimisation methods and supporting code
├── main.ipynb        # Main analysis / experiments notebook
├── requirements.txt  # Python dependencies
├── README.md
└── Optimisation_For_Data_Science__Homework_final.pdf  # Project report (upload separately)
```

### Getting started

1. Install the project dependencies with `pip install -r requirements.txt`.
2. Open **`main.ipynb`** in Jupyter or VS Code to inspect the experiments and analysis.
3. Consult **`data/`** and the notebook for input data and dataset paths; some datasets may need to be downloaded separately.

> **Note:** The README summarises the submitted report. Exact runtime requirements and data paths should be checked against the repository before attempting full reproduction.

## Full project report

The report contains the mathematical derivations, optimisation algorithms, dataset preparation, convergence analysis and additional experimental figures.

**[Read the full report (PDF)](./Optimisation_For_Data_Science__Homework_final.pdf)**

*For this link to work, upload the PDF with exactly this filename to the repository root, alongside `README.md`.*

## Team and academic context

Completed collaboratively for **Optimization for Data Science** at the **University of Padua** by:

- Juan Jose Baldelomar Cabrera
- Davide Giardini
- Emiliano Salinas Arroyo
- Sara Francesca Scully

The project and experimental results represent the work of the group.

### Reference

Nutini, J., Schmidt, M., Laradji, I. H., Friedlander, M., & Koepke, H. (2015; revised 2018). *Coordinate Descent Converges Faster with the Gauss–Southwell Rule than Random Selection*. [arXiv:1506.00552](https://arxiv.org/abs/1506.00552).
