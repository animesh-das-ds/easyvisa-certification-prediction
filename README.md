# EasyVisa — Predicting Visa Certification Outcomes

![Course](https://img.shields.io/badge/Course-Machine%20Learning%20--%202-6f42c1)
![Grade](https://img.shields.io/badge/Grade-A%2B%20(4.33)-success)
![Type](https://img.shields.io/badge/Type-Ensembles%20%C2%B7%20Classification-blue)
![Tools](https://img.shields.io/badge/Tools-scikit--learn%20%C2%B7%20AdaBoost%20%C2%B7%20Gradient%20Boosting-orange)

A classification project that predicts **whether a US work-visa application will be certified or denied**, and identifies which applicant and employer characteristics drive approval.

---

## Business Problem

The US has a continuous demand for skilled workers, and the Immigration and Nationality Act governs how foreign workers enter while protecting domestic wages and conditions. Employers and policymakers want to understand **what makes an application succeed**, so applications can be better aligned with certification standards.

## Objective

Build and tune classification models that predict visa certification, and identify the strongest predictors of approval.

## Data

| Attribute | Detail |
|---|---|
| Applications | **25,480** |
| Features | education level, job experience, job training requirement, prevailing wage, wage unit, full-time status, number of employees, employer age, region of employment, continent of origin |
| Target | `case_status` — certified / denied |

## Approach

1. **EDA** — univariate analysis of every feature, then bivariate analysis against certification.
2. **Data quality** — confirmed zero missing values and duplicates; found and removed impossible negative values in `no_of_employees`.
3. **Outlier treatment** — log-transformed `no_of_employees` and `prevailing_wage` to correct heavy right skew; capped unrealistic `employer_age` values (some firms appeared 200+ years old) using the IQR method.
4. **Feature engineering** — 80/5/15 stratified train/validation/test split, one-hot encoding, and target encoding to 0 = denied, 1 = certified.
5. **Modelling** — five ensemble/tree models — **Bagging, Decision Tree, Random Forest, AdaBoost and Gradient Boosting** — each evaluated with 5-fold stratified cross-validation using **recall** as the primary metric (to minimise false negatives on certified cases).
6. **Class-imbalance experiments** — trained the same models on the original data, on SMOTE-oversampled data, and on randomly undersampled data, then compared.
7. **Hyperparameter tuning** — `RandomizedSearchCV` with 5-fold CV optimising F1-macro, for AdaBoost (on both undersampled and original data) and Gradient Boosting.

## Key Findings

| Driver | Result |
|---|---|
| **Education is the single strongest predictor** | Certification rates rise monotonically: **Doctorate 87.2% · Master's 78.6% · Bachelor's 62.2% · High School 34.0%** |
| **Job experience gives a large boost** | **74.5%** approval with experience vs **56.1%** without |
| **Higher wages correlate with approval** | Certified applications average ~**$77,294** prevailing wage vs ~**$68,749** for denied — about $8,500 higher |
| **Region matters** | Midwest **75.5%** and South **70.0%** lead; Northeast 62.9%, West 62.3%, Island regions 60.3% |
| **Wage structure matters** | Yearly-paid roles approved **69.9%** of the time; hourly roles only **34.6%** |
| **Continent shows disparities** | Europe **79.2%**, Africa 72.1%, Asia 65.3% (Asia is the largest applicant group) |
| **Company size and age** | Limited impact — larger firms show only a slight edge; employer age nearly identical for certified (~29 yrs) and denied (~28 yrs) |
| **Employment type** | Barely matters — full-time 66.6% vs part-time 68.5% |

**Model comparison**
- **AdaBoost and Gradient Boosting** consistently outperformed Bagging and single Decision Trees.
- On the original data, AdaBoost reached the highest cross-validated recall (**0.8887**) and validation recall (**0.8473**).
- **Sampling trade-off:** SMOTE and undersampling both slightly *reduced* recall versus the original data — oversampling added synthetic variance, undersampling discarded useful majority-class information.
- **Tuned AdaBoost:** best parameters were `n_estimators = 90`, `learning_rate = 0.2`, base tree `max_depth = 3` → validation **recall 0.847, precision 0.761, F1 0.802, ROC-AUC 0.768** — the strongest overall configuration.

## Recommendations

1. **Prioritise recall** when the goal is to identify likely-certified applications, and use the **tuned AdaBoost on the original data** as the production candidate.
2. **Advise employers to align applications with the strongest approval factors** — higher education, demonstrated job experience, competitive prevailing wages, and salaried (yearly) rather than hourly structures.
3. **Recognise regional labour demand patterns** — the Midwest and South show more favourable approval trends.
4. **Treat company size and age as weak signals** — compliance and wage fairness matter far more.

## Skills Demonstrated

Ensemble methods (Bagging, Random Forest, AdaBoost, Gradient Boosting) · decision trees · cross-validation · class-imbalance handling (SMOTE, undersampling) · hyperparameter tuning with RandomizedSearchCV · log transformation and IQR capping · metric selection driven by business cost.

## Repository Structure

```
├── notebooks/
│   └── easyvisa_classification.ipynb
├── reports/
│   └── EasyVisa_Business_Report.pdf
├── screenshots/
│   ├── notebook_chart_01.png … notebook_chart_10.png
│   └── report_page_01.png … report_page_04.png
└── README.md
```

---

*PGP in Data Science & Business Analytics · Great Lakes Executive Learning / The University of Texas at Austin, McCombs School of Business. Content drawn from the project notebook, business report and course transcript.*
