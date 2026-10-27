
## Part B — Customer Segmentation and Lapse Prediction

### RFM Analysis

The RFM table is generated in `outputs/rfm.csv` from completed orders before the snapshot date of 2025-10-01. It contains 1,175 customers.

The reference statistics are a median recency of 19 days, maximum recency of 180 days, mean frequency of 9.46, and mean monetary value of INR 2,552.06. The RFM definition uses whole days between the snapshot and the most recent completed order.

### Customer Segmentation

The RFM columns (`recency`, `frequency`, and `monetary`) are standardised with `StandardScaler`. K-means is evaluated for K=2 through K=8 using `n_init=10` and `random_state=42`. K=4 is selected using the elbow, silhouette score, and business interpretability.

The scaled RFM matrix is also reduced to two dimensions with PCA. The explained variance ratios are approximately 0.634 and 0.218, accounting for 85.2% of the variance.

The output `outputs/customer_segments.csv` contains:
`customer_id`, `recency`, `frequency`, `monetary`, `segment_id`, `segment_name`, and `is_premium`.

Segment IDs are arbitrary cluster labels, not a ranking. The segment with the highest mean monetary value is marked Premium.

| Segment | Customers | Recommended action |
|---|---:|---|
| Occasional Value | 436 | Use relevant product recommendations and low-cost reminders to encourage repeat purchases. |
| Premium High-Spend | 138 | Offer personalised service and relevant loyalty benefits. |
| Loyal Frequent | 419 | Encourage continued engagement with convenience-focused offers rather than unnecessary discounts. |
| At-Risk Inactive | 182 | Test a targeted win-back campaign and monitor reactivation. |

### Lapse Prediction

The lapse cohort includes customers with at least one completed order before 2025-09-01. The label is 1 when a customer has no completed orders from 2025-09-01 through 2025-09-30, and 0 otherwise.

The cohort contains 1,156 customers, of whom 443 (38.32%) are labelled lapsed. The stratified split contains 924 training rows and 232 test rows.

Features include pre-cutoff recency, frequency, monetary value, average discount percentage, cancellation rate, average delivery time, favourite product category, city, age, gender, and tenure.

**Leakage safeguards:**
- No order dated on or after 2025-09-01 is used to construct model features.
- The segment labels from the whole-window RFM analysis are not model features.
- Numeric scaling and categorical encoding are fitted within the training pipeline and applied to test rows without fitting on them.
- No resampling is applied to validation or test rows. The class-balance experiment duplicates lapsed training rows only.
- Model selection uses training-set cross-validation. The held-out test set is reserved for final evaluation.

### Lapse Model Results

The baseline always predicts "not lapsed" and achieves approximately 0.616 accuracy, with zero F1 for the lapsed class.

The current tuned logistic regression achieves test precision 0.6444, recall 0.6517, and F1 0.6480 for the lapsed class. The tuned random forest achieves test precision 0.5895, recall 0.6292, and F1 0.6087.

The logistic regression is the current selected model because it achieves the higher held-out test F1 of these two fitted candidates. The selected model is saved to `outputs/lapse_model.joblib`, and the metrics are saved to `outputs/lapse_metrics.json`. Reloading the saved model reproduced the original test predictions exactly.

### Class-Balance Experiment

Oversampling the training data increased the minority-class training count from 354 to 570, matching the majority-class count of 570. On the unchanged test set, logistic-regression recall increased from 0.6517 to 0.8539, precision changed from 0.6444 to 0.5891, and F1 increased from 0.6480 to 0.6972.

This increases the number of lapsing customers identified, at the cost of more false-positive retention targets.

### Model Interpretation

The five most important Random Forest features in the current run are:
1. Frequency — 0.2665
2. Monetary value — 0.2440
3. Recency — 0.1134
4. Average delivery time — 0.0647
5. Tenure — 0.0566

The logistic-regression coefficients for recency, frequency, and monetary value are +0.0396, -1.6982, and +0.2846, respectively. These are conditional associations in the fitted model, not causal effects.

### Premium Retention List

The top 10 Premium customers in the held-out test set, ranked by predicted lapse probability, are written to `outputs/retention_list.csv`. Three of the 10 customers in the current list were actually labelled lapsed.

**Limitation:** The Premium flag is derived from whole-window RFM, which includes the September label window. Therefore, the retention list is illustrative and is not a leakage-free evaluation. The model's own predictive features still use only information dated before 2025-09-01.

### Reproducibility Commands

Run the following from the project root after completing Part A:

```powershell
python build_rfm.py
python evaluate_clusters.py
python build_segments.py
python build_lapse_data.py
python train_lapse_models.py
python balance_experiment.py
python interpret_lapse_models.py
python build_retention_list.py
```

### Limitations and Next Steps

- The Premium retention list uses a post-hoc whole-window segment label and is illustrative, not a leakage-free evaluation.
- Validate model stability with later time-based holdout periods before using lapse scores in production.
- Monitor false-positive campaign costs and missed lapsing customers.
- Document and evaluate support-ticket triage and policy-grounded support automation only after those workflows have been implemented and tested.

## Current Status

Part A data cleaning, SQLite storage, SQL analytics, and pandas checks have been run locally. Part B segmentation, lapse modelling, class-balance experimentation, interpretation, and retention-list generation have also been run locally.

### Part C — Support Ticket Triage Copilot

**Privacy and input guardrails**
- Redacted email addresses (55), phone numbers (91), and order IDs (132): **278 substitutions total**.
- No matching email, phone, or order-ID patterns remained after redaction.
- Empty tickets were marked `skipped_empty`; tickets exceeding the 400-token estimate were marked `flagged_long`. Neither group was sent to the model.

**Prompt comparison**

| Metric | Zero-shot | Few-shot |
|---|---:|---:|
| Classified tickets | 147 | 147 |
| Accuracy on classified tickets | 85.71% | 100% |
| Total response usage tokens | 11,584 | 21,568 |
| Tokens per classified ticket | 78.80 | 146.72 |

Few-shot prompting used approximately 86% more tokens per classified ticket, but improved measured accuracy by 14.29 percentage points on this deterministic mock-LLM evaluation. This supports using few-shot examples when classification quality is more important than token cost. These results are specific to the supplied mock model and dataset, not a guarantee of production LLM performance.

**Final few-shot results**
- `classified_json`: 135
- `classified_keyword`: 12
- `flagged_invalid_output`: 9
- `skipped_empty`: 4
- `flagged_long`: 4
- Final output: `outputs/triage_results.csv` (164 rows, eight required columns).

**Budget demonstration**
- The zero-shot run used a 6,000-token budget and produced 75 `budget_exhausted` tickets for human review.
- The reported 6,029 tokens reflect the final successful response taking the cumulative usage above the threshold; the budget check prevents the next call once usage is already at or above the limit.
- Output: `outputs/triage_budget_demo.csv`.

**Routing and human review**
- Valid classifications are routed by category and customer premium status: fraud to `security_team`, premium customers to `premium_desk`, and other valid classifications to `support_queue`.
- Empty, long, invalid-output, and budget-exhausted cases require human review and are assigned normal priority.
- Transient failures were retried with exponential backoff and jitter. Invalid model output was not silently accepted as a valid category.
- `mock_llm.py` was used offline; no hosted model API key or external API call was required.


## Part D — Policy RAG Assistant and Refund Agent

### 1. Policy chunking and retrieval

The five policy documents were split into 40-word chunks with a 10-word overlap. Each chunk records its source filename.

| Document | Chunks |
|---|---:|
| refund_policy.txt | 4 |
| delivery_policy.txt | 3 |
| escalation_policy.txt | 3 |
| payment_policy.txt | 3 |
| privacy_policy.txt | 2 |
| **Total** | **15** |

The retrieval implementation uses `TfidfVectorizer()` and cosine similarity. For the query `refund policy late delivery`, all three top-ranked chunks came from `refund_policy.txt`.

**TF-IDF explanation:** Term Frequency (TF) measures how often a term occurs in a chunk, while Inverse Document Frequency (IDF) reduces the weight of terms that appear across many chunks. A limitation is that TF-IDF depends on word overlap, so different phrasings with the same meaning may not retrieve each other well.

### 2. Retrieval and grounding evaluation

| Evaluation | Expected-document retrieval | Expected fact present | Correct fallback |
|---|---:|---:|---:|
| 10-word overlap | 12/12 | 12/12 | 2/2 |
| Zero overlap | 12/12 | 10/12 | 2/2 |

With zero overlap, only 13 chunks were produced. Two answerable questions failed the exact expected-fact check:

- **Late-delivery refund eligibility:** the answer included “receives a full refund of the order amount,” but omitted the exact phrase “more than 45 minutes.”
- **Failed payment refund timing:** the answer included “returned automatically,” but omitted “48 hours.”

These examples demonstrate how chunk boundaries can separate related terms and facts from the context selected by retrieval.

### 3. Refund agent scenario results

The agent uses five tools: `get_order`, `search_policy`, `calculate_refund`, `issue_refund`, and `escalate_to_human`. It uses the local SQLite database and `mock_llm.py`; no external model API or API key is required.

| Scenario | Status | Refund amount (INR) | Model calls | Tokens used |
|---|---|---:|---:|---:|
| S1 | REFUND | 128.25 | 5 | 1212 |
| S2 | REFUND | 135.00 | 5 | 1212 |
| S3 | ESCALATE | 1999.00 | 5 | 1199 |
| S4 | DENIED | 0.00 | 4 | 857 |
| S5 | DENIED | 0.00 | 3 | 526 |
| S6 | NOT_FOUND | 0.00 | 2 | 198 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3378 |

S1 and S2 received full refunds because their completed orders were delivered in more than 45 minutes. S3 was escalated because its refund exceeded INR 1000. S4 and S5 were denied, and S6 returned NOT_FOUND.

S7 intentionally requests a looping tool interaction and reaches the six-model-call limit. It finishes with `FALLBACK_MAX_ITERATIONS` while remaining below the default 6000-token limit.

### 4. Safety-limit demonstrations

| Demonstration | Configuration | Result |
|---|---|---|
| S1 token-limit demo | `token_limit=200` | `FALLBACK_TOKEN_LIMIT` |
| Timeout demo | `time_limit_seconds=0` | `FALLBACK_TIMEOUT` |
| S7 iteration limit | `max_iterations=6`, `token_limit=6000` | `FALLBACK_MAX_ITERATIONS` |

The agent checks elapsed time before each model call and checks accumulated model usage after each response. If the token limit is exceeded, it stops before executing any tool requested by that response.

### 5. Crash-and-resume demonstration

S1 was deliberately crashed immediately after its second tool observation. The checkpoint had already been saved.

**Tools executed across the crash and resume:**

`get_order` → `search_policy` → `calculate_refund` → `issue_refund`

The resumed run completed with `REFUND: INR 128.25 issued.`, using five model calls and 1212 tokens across the complete run. The first two tools were not repeated. Checkpoints are stored under `state/`.

Run logs are written to `outputs/agent_runs.jsonl`, and refund issuance or escalation actions are recorded in `outputs/refund_ledger.jsonl`.

### 6. ReAct versus plan-and-execute

ReAct calls the model repeatedly, using each tool observation to decide the next action. In these measured scenarios, S1 required five model calls and 1212 tokens, S5 required three calls and 526 tokens, and S6 required two calls and 198 tokens. This incurs repeated prompt and response costs.

A plan-and-execute design could ask the model for a complete action plan once and replan only if an error occurs, potentially reducing model calls and repeated prompt tokens. However, a fixed plan may be less responsive to unexpected observations, such as an order being missing or cancelled. It must still validate every tool result before issuing a refund.

For this workflow, I would choose ReAct because refund decisions depend on the order's actual status, delivery time, and calculated refund amount. Sequential tool observations allow the agent to stop or escalate based on the returned data, while iteration, token, and time limits provide safeguards against unbounded execution.
