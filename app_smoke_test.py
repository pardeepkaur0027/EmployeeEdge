"""FINAL gate: serve the real app via Flask test_client and assert output."""
import os
import sys

sys.path.insert(0, os.getcwd())
from app.app import app, ALL_FIELD_NAMES, FIELD_DEFS

client = app.test_client()

r = client.get("/")
b = r.get_data(as_text=True)
print("GET  /            ->", r.status_code, "| has-Age:",
      'name="Age"' in b, "| inputs:", b.count('name="'))

form = {}
for name, label, ftype, default, extra in FIELD_DEFS:
    if ftype == "number":
        form[name] = str(default)
    else:
        form[name] = default
assert sorted(form.keys()) == sorted(ALL_FIELD_NAMES), "form field mismatch"

r = client.post("/predict", data=form)
b = r.get_data(as_text=True)
print("POST /predict     ->", r.status_code, "| len", len(b))
print("  waterfall-b64:", "base64" in b)
print("  plan-present:", "action" in b.lower())
print("  sims-present:", "what-if" in b.lower() or "simulation" in b.lower())
print("  has-%:", ">" in b and "%" in b)

ok_gate = r.status_code == 200 and "base64" in b and "action" in b.lower()
print("\nFINAL GATE:", "PASS" if ok_gate else "FAIL")