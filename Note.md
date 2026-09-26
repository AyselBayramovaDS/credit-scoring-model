# Credit Scoring Model — Documentation

## 1. Data Loading

The data we are working with is in plain text format, with values separated by spaces. It also lacks column names and a header row. Therefore, we use the sep=" " parameter for the spaces, header=None to prevent pandas from treating the first row as a header, and since there are no column titles, we define the column names based on the data before reading the file by using names=columns.

## 2. Target Encoding (class)

The 'class' column was originally encoded as 1 (good) and 2 (bad). Since the model expects a standard binary format (0/1), I mapped these values to 0 (good) and 1 (bad). Specifically, 1 was assigned to 'bad' because our primary objective is to detect bad credit risk. This ensures that the model focuses on 1 as the target class.

## 3. Categorical Feature Mapping

To ensure the readability of the model's and SHAP's final outputs, I mapped all existing 'object' type attributes in the file using a dictionary. Additionally, I verified with 'value_counts()' that the conversion was executed correctly and without any data loss.

## 4. One-Hot Encoding

Although I previously mapped the categorical columns into human-readable text using a dictionary, the machine learning model cannot process text directly and requires numerical inputs. Therefore, I applied the pd.get_dummies() function to convert each categorical variable into a binary 0/1 (True/False) format.

This process expanded each category into separate columns (e.g., the checking_status column was transformed from 4 categories into 3 new columns). By setting the drop_first=True parameter, I excluded the reference category from each group to prevent multicollinearity. Consequently, when all remaining dummy columns for a feature are 0, it automatically represents the omitted baseline category.

As a result of this encoding, the total number of columns increased from 21 to 49. The existing numerical features (such as duration, age, and credit_amount) remained unchanged.

## 5. Column Name Cleaning

I removed/replaced special characters such as <, >=, and = from the column names because XGBoost does not support them and threw an error. Instead of simply deleting them, I replaced them with meaningful words (< → "under", >= → "over") to keep the column names both XGBoost-compatible and readable. I then replaced remaining spaces with underscores (_) and cleaned up double underscores.

## 6. Train/Test Split

X (features) and y (target/class) were separated, and the data was split into training (80%) and test (20%) sets using train_test_split, with stratify=y to preserve the original 70/30 class balance in both sets. This is important for reliable evaluation on an imbalanced dataset like this one.

## 7. Baseline Model — Logistic Regression

In the initial logistic regression model, I encountered a ConvergenceWarning because the default max_iter=1000 was insufficient for convergence. I resolved this issue by increasing max_iter to 5000. Consequently, the recall for class 1 improved from 0.80 to 0.82, and the number of False Negatives (FN) decreased from 12 to 11. Based on the cost matrix, the total financial cost was reduced from 95 to 94.

class_weight="balanced" was used to handle the class imbalance, ensuring the model pays more attention to the minority class (bad credit risk).

## 8. XGBoost Model

After Logistic Regression, I built an XGBoost model as well, as required by the task, in order to compare the two algorithms. I chose XGBoost because it performs strongly on tabular data and can capture complex, non-linear relationships between features that Logistic Regression cannot.

To handle the class imbalance, I used the scale_pos_weight parameter (the XGBoost equivalent of Logistic Regression's class_weight="balanced"): scale = (y_train == 0).sum() / (y_train == 1).sum().

Result (with the default 0.5 threshold):
- Recall (class 1, bad credit): 0.55
- Precision (class 1): 0.57
- Cost (5×FN + 1×FP): 160

This result was WORSE than Logistic Regression (recall=0.82, cost=94). The reason is that while scale_pos_weight balances the training process, the .predict() method still uses the default 0.5 threshold, which does not match our cost matrix. In the next step, I use predict_proba() to find the optimal threshold and properly evaluate XGBoost's real potential.

## 9. Threshold Selection (Cost-Based Decision)

In the model, xgb_model.predict() automatically uses a 0.5 threshold. In other words, "if the probability is over 50%, classify as bad." However, our cost matrix is at a 5:1 ratio, and we must strive to identify "bad" customers with high accuracy.

.predict_proba() returns the probabilities for both classes. Since we only need the probability of being "bad", we extract only the class 1 probabilities using [:, 1]. y_proba is the array storing the generated probabilities for each customer. Since we do not know in advance which threshold is best, we define a range of multiple thresholds.

Initially, I set the step size between probabilities to 0.05, which yielded a "best threshold" of 0.1 (with a lowest cost of 118). Finding the best result at the edge, or boundary, of the tested range creates a "boundary issue." This means we are observing the data through too narrow of a lens.

To test this suspicion, I first reduced the step size to 0.01 but kept the starting point of the range at 0.1. The result was a best_threshold of 0.11 and a cost of 115. This was slightly better than the previous result (cost = 118), but since 0.11 was still close to the 0.1 boundary, it did not fully resolve my suspicion.

Next, I restarted the range from the absolute minimum (0.01) and tested again using thresholds = np.arange(0.01, 0.9, 0.01). This time, the result changed completely, yielding a best_threshold of approximately 0.07 and a cost of 103. This confirmed that my boundary suspicion was correct. The true optimal threshold was below 0.1, which I had missed in the initial tests. Consequently, the final optimal threshold for XGBoost is 0.07, resulting in a cost (5×FN + 1×FP) of 103.

### Same process for Logistic Regression

To ensure a fairer comparison between the models, I decided to tune the threshold for the Logistic Regression model as well. Here, I proceeded with the same range (thresholds = np.arange(0.01, 0.9, 0.01)). Result: best_threshold=0.46, cost=90.

## 10. Final Model Comparison

1. Model: Logistic Regression | Optimal threshold: 0.46 | Cost: 90
2. Model: XGBoost | Optimal threshold: 0.07 | Cost: 103

Logistic Regression yielded a lower cost than XGBoost. Interestingly, the optimal threshold for LR (0.46) is close to the default 0.5, whereas for XGBoost (0.07), it is very far from the default — this suggests XGBoost's predicted probabilities may not be as well-calibrated as Logistic Regression's in this case.

As a result, I selected Logistic Regression (threshold=0.46) as the final model.

## 11. SHAP — Model Explainability

Since the final model is Logistic Regression, I used shap.LinearExplainer instead of shap.TreeExplainer (which is only for tree-based models like XGBoost). SHAP values were computed on X_test (unseen, real customers), since we want to explain how the model behaves on data it wasn't trained on, which is the more realistic and trustworthy basis for explanation.

According to the SHAP analysis, "credit_amount" and "duration" clearly emerged among the most influential features. High credit amounts and longer durations cause the model to assess a higher risk. This aligns perfectly with real-world credit practices. I also confirmed the exact direction of influence for categorical features (checking_status, credit_history) by calculating their mean SHAP values.

### Mean SHAP values (top 10)

| Feature | Mean SHAP Value | Direction |
|---|---|---|
| purpose_car_(new) | -0.070 | decreases risk |
| other_payment_plans_none | -0.055 | decreases risk |
| savings_status_unknown/no_savings_account | -0.048 | decreases risk (unexpected) |
| credit_amount | +0.045 | increases risk |
| checking_status_no_checking_account | +0.036 | increases risk |
| checking_status_balance_under_0_DM | -0.027 | decreases risk (unexpected) |
| duration | +0.022 | increases risk |
| foreign_worker_yes | +0.021 | increases risk |
| property_magnitude_unknown/no_property | +0.021 | increases risk |
| employment_4_under_7_years | -0.021 | decreases risk |

### Unexpected findings

Two results go against intuition and are worth flagging explicitly:

- **savings_status "unknown/no savings account"** — instead of increasing risk as one might expect, it actually **decreases** it (-0.048).
- **checking_status "balance < 0 DM"** (negative balance) — instead of increasing risk as one might expect, it actually **decreases** it (-0.027).

A likely explanation is that these categories represent a different demographic segment in the dataset (e.g., foreign workers, students, or customers who behave more cautiously overall), which could show different risk patterns than assumed. This is a finding worth further investigation, and it's a reminder that the model's results should not be trusted blindly — they need to be interpreted alongside real-world context.

## 12. Summary

Final model: **Logistic Regression**, threshold=0.46, cost=90 (lowest, based on 5×FN + 1×FP).

Class imbalance was handled with class_weight="balanced", and the cost asymmetry was addressed by replacing the default 0.5 threshold with a cost-based optimal threshold search. The model's decisions were explained using SHAP; most results align with real-world credit practices, while two unexpected findings were flagged separately for further scrutiny.
