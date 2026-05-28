#!/usr/bin/env python3
"""
Generate index.html for GitHub Pages.
Run from the project root: python build_dashboard.py
"""

import json
import os
import pandas as pd


def load_daily(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    return df.groupby("date").agg({"pred": "mean", "actual": "first"}).reset_index()


dh = load_daily("models/lstm_target_daily_high_next_day_test_preds.csv")
st = load_daily("models/lstm_target_temp_next_24h_test_preds.csv")

dh["abs_error"] = (dh["pred"] - dh["actual"]).abs()
bins = [40, 55, 65, 75, 85, 110]
bin_labels = ["40–55°F", "55–65°F", "65–75°F", "75–85°F", "85°F+"]
dh["bucket"] = pd.cut(dh["actual"], bins=bins, labels=bin_labels)
bucket_mae = dh.groupby("bucket", observed=True)["abs_error"].mean().round(2)
bucket_counts = dh.groupby("bucket", observed=True)["abs_error"].count()


def jl(lst):
    return json.dumps(lst)


DATA_VARS = f"""
    const dhDates   = {jl([str(d) for d in dh["date"]])};
    const dhPred    = {jl([round(float(v), 1) for v in dh["pred"]])};
    const dhActual  = {jl([round(float(v), 1) for v in dh["actual"]])};
    const stDates   = {jl([str(d) for d in st["date"]])};
    const stPred    = {jl([round(float(v), 1) for v in st["pred"]])};
    const stActual  = {jl([round(float(v), 1) for v in st["actual"]])};
    const errLabels = {jl([str(b) for b in bucket_mae.index])};
    const errMae    = {jl([float(v) for v in bucket_mae.values])};
    const errCounts = {jl([int(v) for v in bucket_counts.values])};
"""

PLOT_CAPTIONS = {
    "1_temp_trend.png": "Temperature trend 2016–2026",
    "2_patterns.png": "Seasonal & diurnal patterns",
    "3_correlation_heatmap.png": "Feature correlation heatmap",
    "4_lag_scatters.png": "Lag feature scatters",
    "5_annual_overlay.png": "Annual temperature overlay",
    "6_distribution.png": "Temperature distribution",
    "7_predictions_target_daily_high_next_day.png": "Daily high: predicted vs actual",
    "7_predictions_target_temp_next_24h.png": "Spot 24h: predicted vs actual",
    "8_importance_target_daily_high_next_day.png": "Feature importance — daily high",
    "8_importance_target_temp_next_24h.png": "Feature importance — spot 24h",
    "9_errors_target_daily_high_next_day.png": "Error analysis — daily high",
    "9_errors_target_temp_next_24h.png": "Error analysis — spot 24h",
    "10_compare_target_daily_high_next_day.png": "Model comparison — daily high",
    "10_compare_target_temp_next_24h.png": "Model comparison — spot 24h",
    "11_calibration_target_daily_high_next_day.png": "Calibration curves — daily high",
    "11_calibration_target_temp_next_24h.png": "Calibration curves — spot 24h",
}

PLOT_GRID = "\n".join(
    f'<figure><img src="plots/{f}" alt="{PLOT_CAPTIONS.get(f, f)}" loading="lazy">'
    f"<figcaption>{PLOT_CAPTIONS.get(f, f)}</figcaption></figure>"
    for f in sorted(os.listdir("plots"))
    if f.endswith(".png")
)

# ---------- HTML template (no f-string — braces are literal) ----------
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SFO Weather Forecaster</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, -apple-system, sans-serif; background: #f1f5f9; color: #1e293b; line-height: 1.5; }

    header { background: #0f172a; color: white; padding: 3rem 1.5rem 2.5rem; text-align: center; }
    header h1 { font-size: 2rem; font-weight: 700; margin-bottom: 0.4rem; }
    header .sub { color: #94a3b8; font-size: 0.95rem; max-width: 580px; margin: 0 auto 2rem; }

    .stats { display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap; }
    .stat { background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.1); border-radius: 10px; padding: 1rem 1.5rem; text-align: center; min-width: 120px; }
    .stat .val { font-size: 1.75rem; font-weight: 700; color: #60a5fa; }
    .stat .lbl { font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.2rem; }

    main { max-width: 1080px; margin: 2rem auto; padding: 0 1.25rem; display: flex; flex-direction: column; gap: 1.5rem; }

    .card { background: white; border-radius: 12px; padding: 1.5rem; box-shadow: 0 1px 4px rgba(0,0,0,0.07); }
    .card h2 { font-size: 1rem; font-weight: 600; margin-bottom: 0.3rem; }
    .card .sub { font-size: 0.82rem; color: #64748b; margin-bottom: 1.1rem; }

    .callout { background: #eff6ff; border-left: 4px solid #3b82f6; border-radius: 0 8px 8px 0; padding: 0.9rem 1.1rem; font-size: 0.92rem; }
    .callout strong { color: #1d4ed8; }

    .tabs { display: flex; gap: 0.4rem; margin-bottom: 1rem; }
    .tab { padding: 0.35rem 0.9rem; border-radius: 6px; border: none; cursor: pointer; font-size: 0.82rem; font-weight: 500; background: #f1f5f9; color: #64748b; transition: background 0.15s, color 0.15s; }
    .tab.active { background: #3b82f6; color: white; }

    .chart-wrap { position: relative; height: 300px; }
    .chart-wrap-sm { position: relative; height: 240px; }

    table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
    th { background: #f8fafc; text-align: left; padding: 0.55rem 0.75rem; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; border-bottom: 2px solid #e2e8f0; }
    td { padding: 0.65rem 0.75rem; border-bottom: 1px solid #f1f5f9; }
    tr:last-child td { border-bottom: none; }
    .win { color: #16a34a; font-weight: 600; }
    .dim { color: #94a3b8; }

    .gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 0.75rem; }
    figure { background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
    figure img { width: 100%; display: block; cursor: zoom-in; transition: opacity 0.15s; }
    figure img:hover { opacity: 0.9; }
    figcaption { padding: 0.45rem 0.7rem; font-size: 0.77rem; color: #64748b; }

    #lb { display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.88); z-index: 100; align-items: center; justify-content: center; cursor: zoom-out; }
    #lb.open { display: flex; }
    #lb img { max-width: 92vw; max-height: 92vh; border-radius: 4px; }

    footer { text-align: center; padding: 2rem; font-size: 0.78rem; color: #94a3b8; }
    footer a { color: #60a5fa; text-decoration: none; }

    @media (max-width: 580px) {
      header h1 { font-size: 1.5rem; }
      .stat .val { font-size: 1.4rem; }
    }
  </style>
</head>
<body>

<header>
  <h1>SFO Weather Forecaster</h1>
  <p class="sub">10 years of NOAA hourly data · XGBoost vs LSTM · predicting tomorrow's temperature at San Francisco Airport</p>
  <div class="stats">
    <div class="stat">
      <div class="val">3.47°F</div>
      <div class="lbl">LSTM MAE<br>Daily High</div>
    </div>
    <div class="stat">
      <div class="val">4.04°F</div>
      <div class="lbl">XGBoost MAE<br>Daily High</div>
    </div>
    <div class="stat">
      <div class="val">4.19°F</div>
      <div class="lbl">Baseline MAE<br>Daily High</div>
    </div>
    <div class="stat">
      <div class="val">491</div>
      <div class="lbl">Test days<br>Nov 2024–May 2026</div>
    </div>
  </div>
</header>

<main>

  <div class="callout">
    On next-day high temperature, the LSTM beats XGBoost by <strong>0.57°F MAE</strong> — despite using only 11 raw features vs XGBoost's 69 hand-engineered ones. The LSTM figures out the temporal structure on its own. On 24-hour spot temperature they tie at 2.46°F.
  </div>

  <div class="card">
    <h2>Predicted vs Actual Temperature</h2>
    <p class="sub">LSTM test set · daily aggregated · hover to inspect values</p>
    <div class="tabs">
      <button class="tab active" onclick="switchChart('dh', this)">Daily High</button>
      <button class="tab" onclick="switchChart('st', this)">Spot Temp 24h</button>
    </div>
    <div class="chart-wrap">
      <canvas id="predChart"></canvas>
    </div>
  </div>

  <div class="card">
    <h2>Model Comparison</h2>
    <p class="sub">Both models beat naive persistence / climatology baselines on both targets.</p>
    <table>
      <thead>
        <tr>
          <th>Target</th>
          <th>XGBoost MAE</th>
          <th>LSTM MAE</th>
          <th>Baseline MAE</th>
          <th>vs Baseline</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Spot temp 24h ahead</td>
          <td>2.46°F</td>
          <td>2.46°F</td>
          <td>2.60°F</td>
          <td class="win">Beat by 0.14°F</td>
        </tr>
        <tr>
          <td>Tomorrow's daily high</td>
          <td class="dim">4.04°F</td>
          <td class="win">3.47°F ✓</td>
          <td>4.19°F</td>
          <td class="win">Beat by 0.72°F</td>
        </tr>
      </tbody>
    </table>
  </div>

  <div class="card">
    <h2>Error by Temperature Range</h2>
    <p class="sub">LSTM daily high MAE stratified by actual temperature. The model struggles on hot days — too few in training data. Hover a bar for sample size.</p>
    <div class="chart-wrap-sm">
      <canvas id="errChart"></canvas>
    </div>
  </div>

  <div class="card">
    <h2>All Plots</h2>
    <p class="sub">Click any image to enlarge.</p>
    <div class="gallery">
__PLOT_GRID__
    </div>
  </div>

</main>

<div id="lb" onclick="this.classList.remove('open')">
  <img id="lb-img" src="" alt="">
</div>

<footer>
  <p>Built by Jaxson Bie &nbsp;·&nbsp; <a href="https://github.com/jaxsonb" target="_blank">GitHub</a></p>
</footer>

<script>
__DATA_VARS__

  // --- Prediction chart ---
  const predCtx = document.getElementById('predChart').getContext('2d');
  const predChart = new Chart(predCtx, {
    type: 'line',
    data: {
      labels: dhDates,
      datasets: [
        {
          label: 'Actual',
          data: dhActual,
          borderColor: '#1e293b',
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.3,
          fill: false,
        },
        {
          label: 'LSTM Predicted',
          data: dhPred,
          borderColor: '#3b82f6',
          borderWidth: 1.5,
          pointRadius: 0,
          tension: 0.3,
          fill: false,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: { legend: { position: 'top', labels: { boxWidth: 12, font: { size: 12 } } } },
      scales: {
        x: { ticks: { maxTicksLimit: 8, maxRotation: 0 }, grid: { display: false } },
        y: { title: { display: true, text: 'Temperature (°F)' }, grid: { color: '#f1f5f9' } }
      }
    }
  });

  function switchChart(target, btn) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    btn.classList.add('active');
    if (target === 'dh') {
      predChart.data.labels = dhDates;
      predChart.data.datasets[0].data = dhActual;
      predChart.data.datasets[1].data = dhPred;
    } else {
      predChart.data.labels = stDates;
      predChart.data.datasets[0].data = stActual;
      predChart.data.datasets[1].data = stPred;
    }
    predChart.update();
  }

  // --- Error bar chart ---
  const errCtx = document.getElementById('errChart').getContext('2d');
  new Chart(errCtx, {
    type: 'bar',
    data: {
      labels: errLabels,
      datasets: [{
        label: 'MAE (°F)',
        data: errMae,
        backgroundColor: errMae.map(v => v > 10 ? '#ef4444' : v > 4 ? '#f97316' : '#3b82f6'),
        borderRadius: 4,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            afterLabel: ctx => `n = ${errCounts[ctx.dataIndex]} days`
          }
        }
      },
      scales: {
        x: { grid: { display: false } },
        y: { title: { display: true, text: 'MAE (°F)' }, grid: { color: '#f1f5f9' }, beginAtZero: true }
      }
    }
  });

  // --- Lightbox ---
  document.querySelectorAll('.gallery img').forEach(img => {
    img.addEventListener('click', () => {
      document.getElementById('lb-img').src = img.src;
      document.getElementById('lb').classList.add('open');
    });
  });
</script>

</body>
</html>
"""

output = HTML_TEMPLATE.replace("__DATA_VARS__", DATA_VARS).replace("__PLOT_GRID__", PLOT_GRID)

with open("index.html", "w") as f:
    f.write(output)

print("index.html written successfully.")
print("Next steps:")
print("  1. git add index.html plots/")
print("  2. git commit -m 'Add GitHub Pages dashboard'")
print("  3. git push")
print("  4. GitHub repo → Settings → Pages → Source: Deploy from branch → main / root")
