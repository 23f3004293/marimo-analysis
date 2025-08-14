# analysis.py — Marimo reactive notebook
# Role: Data Scientist demo
# Email: 23f3004293@ds.study.iitm.ac.in
#
# Run locally:
#   pip install marimo plotly pandas numpy
#   marimo run analysis.py
#
# This notebook demonstrates:
# - Variable dependencies across cells
# - An interactive slider widget
# - Dynamic markdown based on widget state
# - Commented data flow

import marimo as mo

app = mo.app()


# --- Cell 1: UI controls ----------------------------------------------------
# Exposes a slider that downstream cells depend on.
@app.cell
def __(mo=mo):
    sigma = mo.ui.slider(0.0, 5.0, step=0.1, value=1.0, label="Noise std (σ)")
    n = mo.ui.slider(50, 1000, step=50, value=300, label="Sample size (n)")
    mo.vstack([sigma, n])
    return n, sigma


# --- Cell 2: Generate synthetic data ---------------------------------------
# Depends on: sigma, n (from Cell 1)
# Produces: df (used by later cells)
@app.cell
def __(n, sigma):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    x = np.linspace(0, 10, int(n.value))
    # True relationship: y = 2.5x + 5 + ε, where ε ~ N(0, σ^2)
    eps = rng.normal(0.0, float(sigma.value), size=x.shape)
    y = 2.5 * x + 5 + eps

    df = pd.DataFrame({"x": x, "y": y})
    df.head()
    return df, np


# --- Cell 3: Fit simple linear regression ----------------------------------
# Depends on: df (from Cell 2)
# Produces: slope, intercept, r (used by plot & report)
@app.cell
def __(df, np):
    # Ordinary least squares via closed-form using numpy.polyfit (degree=1)
    slope, intercept = np.polyfit(df["x"], df["y"], 1)

    # Correlation coefficient r
    r = np.corrcoef(df["x"], df["y"])[0, 1]

    # Predicted values for plotting the fit line
    y_hat = slope * df["x"] + intercept
    return intercept, r, slope, y_hat


# --- Cell 4: Dynamic report (Markdown) -------------------------------------
# Depends on: sigma, n, slope, intercept, r
# Renders: human-readable summary that updates as the slider moves.
@app.cell
def __(intercept, n, r, sigma, slope):
    mo.md(
        f"""
### Relationship summary
- **Model**: $y = {slope:.2f}x + {intercept:.2f}$
- **Correlation**: $r = {r:.3f}$
- **Sample size**: **{n.value}**
- **Noise std (σ)**: **{sigma.value:.1f}**

_As σ increases, the points spread further from the line; correlation and the stability of the fitted slope typically decrease._
"""
    )


# --- Cell 5: Interactive visualization -------------------------------------
# Depends on: df, y_hat
# Plots: scatter of (x, y) and fitted line \n
@app.cell
def __(df, y_hat):
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_scatter(x=df["x"], y=df["y"], mode="markers", name="observations")
    fig.add_scatter(x=df["x"], y=y_hat, mode="lines", name="fit")
    fig.update_layout(
        title="Linear relationship with controllable noise",
        xaxis_title="x",
        yaxis_title="y",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig


# --- Cell 6: Data flow diagram (text) --------------------------------------
# Static documentation of dependencies to aid maintainability.
@app.cell
def __():
    mo.md(
        """
**Data flow**

`[Cell 1: UI]` → `sigma, n`  ⟶  `[Cell 2: Data]` → `df`  ⟶  `[Cell 3: Fit]` → `slope, intercept, r, y_hat`

Then:
- `[Cell 4: Report]` consumes `sigma, n, slope, intercept, r` → dynamic Markdown
- `[Cell 5: Plot]` consumes `df, y_hat` → interactive figure
"""
    )


if __name__ == "__main__":
    app.run()
