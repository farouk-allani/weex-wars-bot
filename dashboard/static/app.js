/* WEEX AI Wars Command Center */

const $ = (id) => document.getElementById(id);

let equityChart = null;
const REFRESH_MS = 5000;

function fmtUsd(n, digits = 2) {
  if (n == null || Number.isNaN(n)) return "—";
  const sign = n < 0 ? "-" : "";
  return sign + "$" + Math.abs(n).toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function fmtPct(n, digits = 1) {
  if (n == null || Number.isNaN(n)) return "—";
  return (n * 100).toFixed(digits) + "%";
}

function clsPnL(el, value) {
  el.classList.remove("pos", "neg");
  if (value > 0) el.classList.add("pos");
  if (value < 0) el.classList.add("neg");
}

function setText(id, text) {
  const el = $(id);
  if (el) el.textContent = text;
}

function updateLiveIndicator(bot) {
  const el = $("live-indicator");
  const text = $("live-text");
  el.classList.remove("on", "warn");
  if (bot.demo || !bot.has_state) {
    text.textContent = "Demo / bot offline";
    el.classList.add("warn");
  } else if (bot.alive) {
    text.textContent = "Live · state fresh";
    el.classList.add("on");
  } else {
    const age = bot.state_age_sec != null ? Math.round(bot.state_age_sec) + "s ago" : "stale";
    text.textContent = "State stale (" + age + ")";
    el.classList.add("warn");
  }
}

function updateKPIs(m) {
  setText("kpi-equity", fmtUsd(m.equity));
  setText("kpi-equity-sub", "start " + fmtUsd(m.initial_capital, 0) + " · open " + (m.open_count || 0));
  setText("kpi-pnl", fmtUsd(m.total_pnl));
  setText("kpi-pnl-pct", (m.pnl_pct >= 0 ? "+" : "") + (m.pnl_pct || 0).toFixed(2) + "%");
  clsPnL($("kpi-pnl"), m.total_pnl);
  clsPnL($("kpi-pnl-pct"), m.total_pnl);

  setText("kpi-wr", fmtPct(m.win_rate, 1));
  setText("kpi-wl", m.wins + "W / " + m.losses + "L");
  setText("kpi-dd", fmtPct(m.max_drawdown, 2));
  setText("kpi-trades", m.total_trades + " closed · " + (m.open_count || 0) + " open");
  setText("kpi-daily", fmtUsd(m.daily_pnl));
  clsPnL($("kpi-daily"), m.daily_pnl);

  const streak =
    m.consecutive_wins > 0
      ? m.consecutive_wins + " win streak"
      : m.consecutive_losses > 0
        ? m.consecutive_losses + " loss streak"
        : "no streak";
  setText("kpi-streak", streak);

  setText("live-balance", fmtUsd(m.balance != null ? m.balance : m.equity));
  setText("live-upnl", fmtUsd(m.unrealized_pnl || 0));
  clsPnL($("live-upnl"), m.unrealized_pnl || 0);
  setText("live-closed", fmtUsd(m.closed_pnl != null ? m.closed_pnl : 0));
  clsPnL($("live-closed"), m.closed_pnl || 0);
}

function updatePositions(positions) {
  const body = $("positions-body");
  const list = positions || [];
  setText("open-count", String(list.length));
  if (!list.length) {
    body.innerHTML = '<tr><td colspan="8" class="empty">No open positions</td></tr>';
    return;
  }
  body.innerHTML = list
    .map((p) => {
      const upnl = Number(p.unrealized_pnl || 0);
      const side = (p.side || "").toLowerCase();
      const pair = (p.symbol || "").split("/")[0] || p.symbol;
      return `<tr>
        <td class="mono">${pair}</td>
        <td class="${side === "long" ? "side-long" : "side-short"}">${side}</td>
        <td class="mono">${Number(p.entry_price).toFixed(4)}</td>
        <td class="mono">${Number(p.size).toFixed(4)}</td>
        <td class="mono ${upnl >= 0 ? "pos" : "neg"}">${fmtUsd(upnl)}</td>
        <td class="mono">${Number(p.stop_loss || 0).toFixed(4)}</td>
        <td class="mono">${Number(p.take_profit || 0).toFixed(4)}</td>
        <td>${p.strategy || "—"}</td>
      </tr>`;
    })
    .join("");
}

function updateRisk(m, riskCfg, bot) {
  const ddPct = (m.max_drawdown || 0) * 100;
  const ddLimit = (riskCfg.max_drawdown || 0.15) * 100;
  const dayLoss = m.daily_pnl < 0 ? (Math.abs(m.daily_pnl) / (m.peak_equity || m.equity || 1)) * 100 : 0;
  const dayLimit = (riskCfg.daily_loss_limit || 0.025) * 100;
  const riskTrade = (riskCfg.max_risk_per_trade || 0.012) * 100;

  setText("g-dd-val", ddPct.toFixed(2) + "%");
  setText("g-dd-limit", ddLimit.toFixed(0) + "%");
  $("g-dd").style.width = Math.min(100, (ddPct / ddLimit) * 100) + "%";

  setText("g-day-val", dayLoss.toFixed(2) + "%");
  setText("g-day-limit", dayLimit.toFixed(1) + "%");
  $("g-day").style.width = Math.min(100, (dayLoss / dayLimit) * 100) + "%";

  setText("g-risk-val", riskTrade.toFixed(1) + "%");
  $("g-risk").style.width = Math.min(100, (riskTrade / 2) * 100) + "%";

  const kill = $("flag-kill");
  kill.classList.toggle("alert", !!m.is_killed);
  kill.classList.toggle("on", !m.is_killed);

  const cd = $("flag-cd");
  const onCd = !!m.cooldown_until;
  cd.classList.toggle("alert", onCd);
  cd.classList.toggle("on", !onCd);
  cd.title = m.cooldown_until || "";

  const partial = $("flag-partial");
  partial.classList.toggle("on", !!riskCfg.partial_tp);

  setText("meta-lev", (bot.leverage || 5) + "x");
  setText("meta-tf", (bot.timeframe || "1h") + " + " + (bot.higher_timeframe || "4h"));
  setText("meta-pos", String(riskCfg.max_open_positions ?? 2));
  setText("meta-cycle", m.cycle_count != null ? String(m.cycle_count) : "—");
}

function updateChart(curve) {
  const labels = curve.map((c) => c.i);
  const data = curve.map((c) => c.equity);
  const ctx = $("equity-chart").getContext("2d");

  const gradient = ctx.createLinearGradient(0, 0, 0, 260);
  gradient.addColorStop(0, "rgba(61, 139, 253, 0.35)");
  gradient.addColorStop(1, "rgba(61, 139, 253, 0.0)");

  if (equityChart) {
    equityChart.data.labels = labels;
    equityChart.data.datasets[0].data = data;
    equityChart.update("none");
    return;
  }

  equityChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Equity",
          data,
          borderColor: "#3d8bfd",
          backgroundColor: gradient,
          borderWidth: 2,
          fill: true,
          tension: 0.25,
          pointRadius: 0,
          pointHoverRadius: 4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "rgba(8,12,20,0.92)",
          borderColor: "rgba(120,150,200,0.2)",
          borderWidth: 1,
          titleFont: { family: "IBM Plex Mono", size: 11 },
          bodyFont: { family: "IBM Plex Mono", size: 12 },
          callbacks: {
            label: (ctx) => " " + fmtUsd(ctx.parsed.y),
          },
        },
      },
      scales: {
        x: {
          display: true,
          ticks: { color: "#6b7c99", maxTicksLimit: 8, font: { family: "IBM Plex Mono", size: 10 } },
          grid: { color: "rgba(255,255,255,0.04)" },
        },
        y: {
          ticks: {
            color: "#6b7c99",
            font: { family: "IBM Plex Mono", size: 10 },
            callback: (v) => "$" + Number(v).toLocaleString(),
          },
          grid: { color: "rgba(255,255,255,0.05)" },
        },
      },
    },
  });
}

function updateTrades(trades) {
  const body = $("trades-body");
  setText("trades-count", String(trades?.length || 0));
  if (!trades || !trades.length) {
    body.innerHTML = '<tr><td colspan="5" class="empty">No trades yet</td></tr>';
    return;
  }
  body.innerHTML = trades
    .slice(0, 25)
    .map((t) => {
      const pnl = Number(t.pnl || 0);
      const side = (t.side || "").toLowerCase();
      const pair = (t.symbol || "").split("/")[0] || t.symbol;
      return `<tr>
        <td class="mono">${pair}</td>
        <td class="${side === "long" ? "side-long" : "side-short"}">${side}</td>
        <td>${t.strategy || "—"}</td>
        <td class="mono ${pnl >= 0 ? "pos" : "neg"}">${fmtUsd(pnl)}</td>
        <td class="mono" style="color:var(--muted)">${t.exit_reason || "—"}</td>
      </tr>`;
    })
    .join("");
}

function renderStatList(el, items, maxAbs) {
  if (!items || !items.length) {
    el.innerHTML = '<div class="stat-row"><span class="name" style="color:var(--muted)">No data yet</span></div>';
    return;
  }
  const peak = maxAbs || Math.max(...items.map((x) => Math.abs(x.pnl || 0)), 1);
  el.innerHTML = items
    .map((s) => {
      const pnl = s.pnl || 0;
      const width = Math.min(100, (Math.abs(pnl) / peak) * 100);
      const wr = s.win_rate != null ? fmtPct(s.win_rate, 0) : s.wins != null ? s.wins + "W" : "";
      return `<div class="stat-row">
        <span class="name">${s.name}</span>
        <span class="pnl ${pnl >= 0 ? "pos" : "neg"}">${fmtUsd(pnl)}</span>
        <span class="stat-meta">${s.trades || 0} trades · ${wr}</span>
        <div class="mini-bar"><i style="width:${width}%;background:${pnl >= 0 ? "linear-gradient(90deg,#3d8bfd,#22d3a6)" : "linear-gradient(90deg,#ff5c7a,#ff8fa3)"}"></i></div>
      </div>`;
    })
    .join("");
}

function updateLogs(payload) {
  const stream = $("log-stream");
  if (payload.path) setText("log-path", payload.path.replace(/\\/g, "/").split(/[/\\]/).slice(-2).join("/"));
  const lines = payload.lines || [];
  if (!lines.length) {
    stream.innerHTML = '<div class="log-line muted">No log lines</div>';
    return;
  }
  stream.innerHTML = lines
    .map((l) => {
      const level = (l.level || "INFO").toUpperCase();
      const ts = l.ts ? `<span style="color:#5a6b88">${l.ts}</span> ` : "";
      return `<div class="log-line ${level}">${ts}<span class="lvl">${level}</span>${escapeHtml(l.message || "")}</div>`;
    })
    .join("");
  stream.scrollTop = stream.scrollHeight;
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function updateSymbols(symbols) {
  const el = $("symbol-chips");
  if (!symbols || !symbols.length) {
    el.innerHTML = '<span class="chip">No symbols</span>';
    return;
  }
  el.innerHTML = symbols.map((s) => `<span class="chip">${s}</span>`).join("");
}

async function refresh() {
  try {
    const [overview, logs] = await Promise.all([
      fetch("/api/overview").then((r) => r.json()),
      fetch("/api/logs?lines=100").then((r) => r.json()),
    ]);

    const bot = overview.bot || {};
    const m = overview.metrics || {};
    const riskCfg = overview.risk_config || {};

    setText("bot-version", bot.version || "v8.5");
    const modePill = $("mode-pill");
    modePill.textContent = (bot.mode || "paper").toUpperCase();
    modePill.classList.toggle("paper", (bot.mode || "") === "paper");
    modePill.classList.toggle("live", (bot.mode || "") === "live");
    setText("profile-pill", bot.profile || "competition");

    $("demo-banner").classList.toggle("hidden", !m.demo);

    updateLiveIndicator(bot);
    updateKPIs(m);
    updateRisk(m, riskCfg, bot);
    updateChart(m.equity_curve || [{ i: 0, equity: m.initial_capital || 10000 }]);
    updatePositions(m.open_positions || []);
    updateTrades(m.trades || []);
    renderStatList($("strat-list"), m.strategy_stats || []);
    renderStatList($("pair-list"), m.pair_stats || []);
    updateLogs(logs);
    updateSymbols(bot.symbols || []);
  } catch (err) {
    console.error(err);
    $("live-text").textContent = "API error";
  }
}

$("refresh-btn").addEventListener("click", refresh);
refresh();
setInterval(refresh, REFRESH_MS);
