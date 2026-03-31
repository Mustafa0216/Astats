import pandas as pd
import numpy as np
from pathlib import Path

def simulate_clinical_trial(rows: int = 1000, output_path: str = "trial_dataset.csv") -> str:
    \"\"\"
    Simulates a basic clinical trial dataset.
    Features: 'age', 'blood_pressure_baseline', 'group' (Treatment/Control).
    Outcome: 'blood_pressure_post', 'recovery_days'.
    \"\"\"
    np.random.seed(42)
    
    # Generate independent variables
    age = np.random.normal(55, 12, rows).astype(int)
    bp_baseline = np.random.normal(130, 15, rows)
    group = np.random.choice(['Control', 'Treatment'], rows, p=[0.5, 0.5])
    
    # Treatment effect (Treatment reduces BP more, reduces recovery days)
    treatment_effect_bp = np.where(group == 'Treatment', -15, -2)
    treatment_effect_days_reduction = np.where(group == 'Treatment', 3, 0)
    
    # Generate dependent variables with noise
    bp_post = bp_baseline + treatment_effect_bp + np.random.normal(0, 5, rows)
    recovery_days = np.random.normal(14, 3, rows) - treatment_effect_days_reduction
    recovery_days = np.clip(recovery_days, 1, None).astype(int)
    
    # Assemble dataframe
    df = pd.DataFrame({
        'patient_id': range(1, rows + 1),
        'age': age,
        'group': group,
        'blood_pressure_baseline': np.round(bp_baseline, 1),
        'blood_pressure_post': np.round(bp_post, 1),
        'recovery_days': recovery_days
    })
    
    # Add some realistic missing data
    missing_indices = np.random.choice(rows, size=int(rows * 0.05), replace=False)
    df.loc[missing_indices, 'blood_pressure_post'] = np.nan
    
    df.to_csv(output_path, index=False)
    return str(Path(output_path).absolute())
