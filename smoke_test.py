"""Smoke test: load the saved model, predict + explain a real employee from the test set."""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import preprocess as pre
from src import explain as ex

ROOT = os.path.dirname(os.path.abspath(__file__))
df = pre.load_data(os.path.join(ROOT, "data", "WA_Fn-UseC_-HR-Employee-Attrition.csv"))
df_clean, X_train, X_test, y_train, y_test, num_cols, cat_cols = pre.prepare_datasets(df)

pipeline = ex.load_pipeline(os.path.join(ROOT, "outputs", "models", "model.joblib"))
print(f"Model pipeline loaded: {type(pipeline.named_steps['model']).__name__}\n")

background = ex.get_background(pipeline, X_train, sample_size=50)
explainer = ex.create_explainer(pipeline, background)

row = X_test.iloc[[0]]
proba = ex.predict_proba(pipeline, row)[0]
print(f"--- Employee test#{0} ---")
print(f"Predicted probability of leaving: {proba:.1%}")
print(f"Actual outcome: {'LEFT' if y_test.iloc[0] == 1 else 'STAYED'}\n")

shap_values, feature_names = ex.explain_raw_rows(pipeline, explainer, row)
factors = ex.top_pushing_factors(shap_values, feature_names, n=3)
print("Top factors pushing toward leaving:")
for f in factors:
    print(ex.format_factor(f))

plan = ex.build_action_plan(shap_values, 0, row, feature_names, top_n=3)
print("\nRetention action plan:")
for action in plan:
    print(f"  [ ] {action}")

assert len(plan) > 0, "Action plan should not be empty"
print("\nSmoke test PASSED.")