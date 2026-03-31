import os
import pandas as pd
import json

def discover_dataset(filepath: str) -> str:
    """
    Auto-discovers a dataset (CSV, Excel, JSON, Parquet, Feather, SPSS, Stata)
    and generates a comprehensive markdown summary that an LLM can use to 
    understand the dataset context before writing code.
    """
    ext = os.path.splitext(filepath)[1].lower()
    
    try:
        if ext == '.csv':
            df = pd.read_csv(filepath)
        elif ext in ['.xls', '.xlsx']:
            df = pd.read_excel(filepath)
        elif ext == '.json':
            df = pd.read_json(filepath)
        elif ext == '.parquet':
            df = pd.read_parquet(filepath)
        elif ext == '.feather':
            df = pd.read_feather(filepath)
        elif ext == '.sav': # SPSS
            df = pd.read_spss(filepath)
        elif ext == '.dta': # Stata
            df = pd.read_stata(filepath)
        else:
            # Fallback to CSV/Text
            df = pd.read_csv(filepath)
    except Exception as e:
        return f"Error reading file '{filepath}': {e}\nPlease ensure the file path is correct and dependencies (like openpyxl, pyarrow, or pyreadstat) are installed if using specialized formats."

    rows, cols = df.shape
    
    summary = [
        f"### Dataset Discovery Summary: `{os.path.basename(filepath)}`",
        f"- **Rows:** {rows}",
        f"- **Columns:** {cols}",
        f"- **Format:** {ext.upper() if ext else 'Unknown (CSV fallback)'}",
        "",
        "#### Column Details"
    ]
    
    for col in df.columns:
        col_type = str(df[col].dtype)
        missing = df[col].isnull().sum()
        missing_pct = (missing / rows) * 100
        
        details = [f"- **`{col}`** (`{col_type}`, {missing} missing / {missing_pct:.1f}%):"]
        
        # Numeric Stats
        if pd.api.types.is_numeric_dtype(df[col]):
            desc = df[col].describe()
            details.append(f"  - Range: [{desc['min']}, {desc['max']}], Mean: {desc['mean']:.2f}, Std: {desc['std']:.2f}")
            details.append(f"  - Quantiles: 25%={desc['25%']:.2f}, 50%={desc['50%']:.2f}, 75%={desc['75%']:.2f}")
            
        # Categorical/Object Stats
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            unique_count = df[col].nunique()
            details.append(f"  - Unique values: {unique_count}")
            if unique_count <= 15:
                # Show top 10 most common or all if < 15
                top_vals = df[col].value_counts().head(10).to_dict()
                details.append(f"  - Sample values (Top 10): {top_vals}")
                
        summary.extend(details)
        
    summary.append("")
    summary.append("#### Data Preview (Head)")
    try:
        summary.append(df.head(5).to_markdown(index=False))
    except ImportError:
        # If tabulate is not installed for to_markdown
        summary.append("```\n" + str(df.head(5)) + "\n```")
    
    return "\n".join(summary)
