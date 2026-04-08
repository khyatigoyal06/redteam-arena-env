import json
from pathlib import Path


def generate_dashboard(results, output_path, announce: bool = False):
    """Generate an HTML dashboard with task, persona, harm-category, and detailed task charts."""

    persona_breaks = {}
    harm_sums = {}
    harm_counts = {}
    task_sums = {}
    task_counts = {}
    task_details = {}
    selected_task_id = None

    for item in results:
        persona = item.get("attacker_persona", "unknown")
        harm_category = item.get("harm_category", "unknown")
        task_id = item.get("task_id")
        guard_score = float(item.get("guard_score", 0.0))
        turn_scores = item.get("turn_scores", [])

        if guard_score < 0.5:
            persona_breaks[persona] = persona_breaks.get(persona, 0) + 1
        else:
            persona_breaks.setdefault(persona, 0)

        task_label = f"Task {task_id}"
        task_sums[task_label] = task_sums.get(task_label, 0.0) + guard_score
        task_counts[task_label] = task_counts.get(task_label, 0) + 1

        harm_sums[harm_category] = harm_sums.get(harm_category, 0.0) + guard_score
        harm_counts[harm_category] = harm_counts.get(harm_category, 0) + 1

        task_details[task_label] = {
            "task_id": int(task_id),
            "final_score": round(guard_score, 4),
            "turn_labels": [f"Turn {index + 1}" for index in range(len(turn_scores))],
            "turn_values": [round(float(score), 4) for score in turn_scores],
        }
        if selected_task_id is None or int(task_id) > selected_task_id:
            selected_task_id = int(task_id)

    harm_avgs = {
        category: (harm_sums[category] / harm_counts[category])
        for category in harm_sums
        if harm_counts[category] > 0
    }
    task_avgs = {
        task_label: (task_sums[task_label] / task_counts[task_label])
        for task_label in task_sums
        if task_counts[task_label] > 0
    }
    overall_mean = round(sum(task_avgs.values()) / len(task_avgs), 4) if task_avgs else None

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chart_data = {
        "task_labels": list(task_avgs.keys()),
        "task_values": [round(value, 4) for value in task_avgs.values()],
        "task_details": task_details,
        "selected_task_label": f"Task {selected_task_id}" if selected_task_id is not None else None,
        "overall_mean": overall_mean,
        "persona_labels": list(persona_breaks.keys()),
        "persona_values": list(persona_breaks.values()),
        "harm_labels": list(harm_avgs.keys()),
        "harm_values": [round(value, 4) for value in harm_avgs.values()],
    }

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>RedTeam Arena Dashboard</title>
  <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>
  <style>
    :root {{
      --bg: #0f0f1a;
      --panel: #181826;
      --text: #f5f5f7;
      --muted: #a9a9b8;
      --accent: #ff3b3b;
      --accent-soft: rgba(255, 59, 59, 0.2);
      --grid: rgba(255, 255, 255, 0.08);
    }}

    * {{ box-sizing: border-box; }}

    body {{
      margin: 0;
      background: radial-gradient(circle at top right, #1b1b30 0%, #0f0f1a 55%);
      color: var(--text);
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
      min-height: 100vh;
    }}

    .container {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 28px 20px 40px;
    }}

    h1 {{
      margin: 0 0 8px;
      font-size: 2rem;
      letter-spacing: 0.02em;
    }}

    p {{
      margin: 0 0 24px;
      color: var(--muted);
    }}

    .charts {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 18px;
    }}

    .summary-row {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin-bottom: 18px;
    }}

    .card {{
      background: linear-gradient(180deg, rgba(255, 255, 255, 0.04), rgba(255, 255, 255, 0.02));
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 10px 26px rgba(0, 0, 0, 0.35);
    }}

    .card h2 {{
      margin: 4px 0 12px;
      font-size: 1rem;
      color: #ffd3d3;
    }}

    .card-note {{
      margin: -4px 0 12px;
      color: var(--muted);
      font-size: 0.92rem;
    }}

    .selector-row {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin: -4px 0 12px;
      color: var(--muted);
      font-size: 0.92rem;
    }}

    .taskpicker {{
      background: rgba(255, 255, 255, 0.06);
      color: var(--text);
      border: 1px solid rgba(255, 255, 255, 0.12);
      border-radius: 10px;
      padding: 8px 10px;
      font-size: 0.92rem;
    }}

    .summary-metric {{
      font-size: 1.8rem;
      font-weight: 700;
      margin: 0;
    }}

    .data-snapshot {{
      margin: 0 0 18px;
      padding: 14px 16px;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(255, 255, 255, 0.08);
      border-radius: 14px;
    }}

    .data-snapshot h2 {{
      margin: 0 0 10px;
      font-size: 1rem;
      color: #ffd3d3;
    }}

    .data-snapshot pre {{
      margin: 0;
      overflow-x: auto;
      white-space: pre-wrap;
      word-break: break-word;
      color: var(--text);
      font-size: 0.92rem;
      line-height: 1.45;
      font-family: "SFMono-Regular", "Menlo", "Monaco", monospace;
    }}

    canvas {{
      width: 100% !important;
      height: 300px !important;
    }}
  </style>
</head>
<body>
  <div class=\"container\">
    <h1>RedTeam Arena Results Dashboard</h1>
    <p>All task scores, persona breaks, harm-category safety averages, and a detailed view of any selected task.</p>

    <div class=\"summary-row\">
      <div class=\"card\">
        <h2>Overall Mean</h2>
        <p class=\"summary-metric\" id=\"overallMeanValue\">N/A</p>
      </div>
      <div class=\"card\">
        <h2>Detailed Task</h2>
        <p class=\"summary-metric\" id=\"detailedTaskValue\">N/A</p>
      </div>
    </div>

    <div class=\"charts\">
      <div class=\"card\">
        <h2>Final Score by Task</h2>
        <canvas id=\"taskChart\"></canvas>
      </div>
      <div class=\"card\">
        <h2>Persona Break Counts (Guard Score &lt; 0.5)</h2>
        <canvas id=\"personaChart\"></canvas>
      </div>
      <div class=\"card\">
        <h2>Average Guard Score by Harm Category</h2>
        <canvas id=\"harmChart\"></canvas>
      </div>
      <div class=\"card\" style=\"grid-column: 1 / -1;\">
        <h2 id=\"detailTaskTitle\">Task Turn-by-Turn Reward</h2>
        <div class=\"selector-row\">
          <label for=\"taskDetailSelect\">Choose task</label>
          <select id=\"taskDetailSelect\" class=\"taskpicker\"></select>
        </div>
        <p class=\"card-note\">Final episode score: <span id=\"detailTaskFinalScore\">N/A</span></p>
        <canvas id=\"longestTaskChart\"></canvas>
      </div>
    </div>

    <div class=\"data-snapshot\">
      <h2>Plotted Data Snapshot</h2>
      <pre id=\"dataSnapshot\"></pre>
    </div>
  </div>

  <script>
    const data = {json.dumps(chart_data)};
    document.getElementById('dataSnapshot').textContent = JSON.stringify(data, null, 2);
    document.getElementById('overallMeanValue').textContent =
      data.overall_mean === null ? 'N/A' : data.overall_mean.toFixed(4);

    const commonScales = {{
      x: {{
        ticks: {{ color: '#f5f5f7' }},
        grid: {{ color: 'rgba(255,255,255,0.08)' }}
      }},
      y: {{
        ticks: {{ color: '#f5f5f7' }},
        grid: {{ color: 'rgba(255,255,255,0.08)' }}
      }}
    }};

    new Chart(document.getElementById('personaChart'), {{
      type: 'bar',
      data: {{
        labels: data.persona_labels,
        datasets: [{{
          label: 'Break Count',
          data: data.persona_values,
          backgroundColor: 'rgba(255, 59, 59, 0.75)',
          borderColor: '#ff3b3b',
          borderWidth: 2
        }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
          legend: {{ labels: {{ color: '#f5f5f7' }} }}
        }},
        scales: commonScales
      }}
    }});

    new Chart(document.getElementById('taskChart'), {{
      type: 'bar',
      data: {{
        labels: data.task_labels,
        datasets: [{{
          label: 'Final Task Score',
          data: data.task_values,
          backgroundColor: 'rgba(255, 110, 110, 0.75)',
          borderColor: '#ff6e6e',
          borderWidth: 2
        }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
          legend: {{ labels: {{ color: '#f5f5f7' }} }}
        }},
        scales: {{
          ...commonScales,
          y: {{
            ...commonScales.y,
            min: 0,
            max: 1
          }}
        }}
      }}
    }});

    new Chart(document.getElementById('harmChart'), {{
      type: 'bar',
      data: {{
        labels: data.harm_labels,
        datasets: [{{
          label: 'Average Guard Score',
          data: data.harm_values,
          backgroundColor: 'rgba(255, 95, 95, 0.7)',
          borderColor: '#ff3b3b',
          borderWidth: 2
        }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
          legend: {{ labels: {{ color: '#f5f5f7' }} }}
        }},
        scales: {{
          ...commonScales,
          y: {{
            ...commonScales.y,
            min: 0,
            max: 1
          }}
        }}
      }}
    }});

    const detailTaskSelect = document.getElementById('taskDetailSelect');
    data.task_labels.forEach((label) => {{
      const option = document.createElement('option');
      option.value = label;
      option.textContent = label;
      if (label === data.selected_task_label) {{
        option.selected = true;
      }}
      detailTaskSelect.appendChild(option);
    }});

    const detailChart = new Chart(document.getElementById('longestTaskChart'), {{
      type: 'line',
      data: {{
        labels: [],
        datasets: [{{
          label: 'Per-Turn Reward',
          data: [],
          borderColor: '#ff3b3b',
          backgroundColor: 'rgba(255, 59, 59, 0.2)',
          fill: true,
          tension: 0.25,
          pointRadius: 3,
          pointHoverRadius: 5
        }}]
      }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{
          legend: {{ labels: {{ color: '#f5f5f7' }} }}
        }},
        scales: {{
          ...commonScales,
          y: {{
            ...commonScales.y,
            min: 0,
            max: 1
          }}
        }}
      }}
    }});

    function renderDetailedTask(taskLabel) {{
      const detail = data.task_details[taskLabel];
      document.getElementById('detailTaskTitle').textContent =
        taskLabel + ' Turn-by-Turn Reward';
      document.getElementById('detailedTaskValue').textContent = taskLabel;
      document.getElementById('detailTaskFinalScore').textContent =
        detail ? detail.final_score.toFixed(4) : 'N/A';
      detailChart.data.labels = detail ? detail.turn_labels : [];
      detailChart.data.datasets[0].data = detail ? detail.turn_values : [];
      detailChart.update();
    }}

    detailTaskSelect.addEventListener('change', (event) => {{
      renderDetailedTask(event.target.value);
    }});

    renderDetailedTask(data.selected_task_label || data.task_labels[0]);
  </script>
</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")
    if announce:
        print(f"Dashboard generated at {output_path}")
