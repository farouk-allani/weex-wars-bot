/* WEEX Bot Command Center — UI + PWA client */

const $ = (id) => document.getElementById(id);
const REFRESH_MS = 5000;
let equityChart = null;

function fmtUsd(n, digits = 2) {
  if (n == null || Number.isNaN(Number(n))) return "—";
  const v = Number(n);
  const sign = v < 0 ? "-" : "";
  return (
    sign +
    "$" +
    Math.abs(v).toLocaleString(undefined, {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    })
  );
}

function fmtPct(n, digits = 1) {
  if (n == null || Number.isNaN(Number(n))) return "—";
  return (Number(n) * 100).toFixed(digits) + "%";
}

function clsPnL(el, value) {
  if (!el) return;
  el.classList.remove("pos", "neg");
  if (value > 0) el.classList.add("pos");
  if (value < 0) el.classList.add("neg");
}

function setText(id, text) {
  const el = $(id);
  if (el) el.textContent = text;
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function updateLiveIndicator(bot) {
  const el = $("live-indicator");
  const text = $("live-text");
  el.classList.remove("on", "warn");
  if (bot.demo || !bot.has_state) {
    text.textContent = "Demo / offline";
    el.classList.add("warn");
    setText("footer-status", "demo shell");
  } else if (bot.alive) {
    text.textContent = "Live";
    el.classList.add("on");
    setText("footer-status", "bot live · " + (bot.cycle_count ?? "—") + " cycles");
  } else {
    const age =
      bot.state_age_sec != null ? Math.round(bot.state_age_sec) + "s" : "stale";
    text.textContent = "Stale " + age;
    el.classList.add("warn");
    setText("footer-status", "state stale");
  }
}

function updateKPIs(m) {
  setText("kpi-equity", fmtUsd(m.equity));
  setText(
    "kpi-equity-sub",
    "start " + fmtUsd(m.initial_capital, 0) + " · open " + (m.open_count || 0)
  );
  setText("kpi-pnl", fmtUsd(m.total_pnl));
  setText(
    "kpi-pnl-pct",
    (m.pnl_pct >= 0 ? "+" : "") + Number(m.pnl_pct || 0).toFixed(2) + "%"
  );
  clsPnL($("kpi-pnl"), m.total_pnl);
  clsPnL($("kpi-pnl-pct"), m.total_pnl);

  setText("kpi-wr", fmtPct(m.win_rate, 1));
  setText("kpi-wl", (m.wins || 0) + "W / " + (m.losses || 0) + "L");
  setText("kpi-dd", fmtPct(m.max_drawdown, 2));
  setText(
    "kpi-trades",
    (m.total_trades || 0) + " closed · " + (m.open_count || 0) + " open"
  );
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

function updateRisk(m, riskCfg, bot) {
  const ddPct = (m.max_drawdown || 0) * 100;
  const ddLimit = (riskCfg.max_drawdown || 0.15) * 100;
  const dayLoss =
    m.daily_pnl < 0
      ? (Math.abs(m.daily_pnl) / (m.peak_equity || m.equity || 1)) * 100
      : 0;
  const dayLimit = (riskCfg.daily_loss_limit || 0.025) * 100;
  const riskTrade = (riskCfg.max_risk_per_trade || 0.012) * 100;

  setText("g-dd-val", ddPct.toFixed(2) + "%");
  setText("g-dd-limit", ddLimit.toFixed(0) + "%");
  $("g-dd").style.width = Math.min(100, (ddPct / Math.max(ddLimit, 0.01)) * 100) + "%";

  setText("g-day-val", dayLoss.toFixed(2) + "%");
  setText("g-day-limit", dayLimit.toFixed(1) + "%");
  $("g-day").style.width = Math.min(100, (dayLoss / Math.max(dayLimit, 0.01)) * 100) + "%";

  setText("g-risk-val", riskTrade.toFixed(1) + "%");
  $("g-risk").style.width = Math.min(100, (riskTrade / 2) * 100) + "%";

  $("flag-kill").classList.toggle("alert", !!m.is_killed);
  $("flag-kill").classList.toggle("on", !m.is_killed);
  $("flag-cd").classList.toggle("alert", !!m.cooldown_until);
  $("flag-cd").classList.toggle("on", !m.cooldown_until);
  $("flag-partial").classList.toggle("on", !!riskCfg.partial_tp);

  setText("meta-lev", (bot.leverage || 5) + "x");
  setText(
    "meta-tf",
    (bot.timeframe || "1h") + "+" + (bot.higher_timeframe || "4h")
  );
  setText("meta-pos", String(riskCfg.max_open_positions ?? 2));
  setText("meta-cycle", m.cycle_count != null ? String(m.cycle_count) : "—");
}

function updateChart(curve) {
  const labels = (curve || []).map((c) => c.i);
  const data = (curve || []).map((c) => c.equity);
  const canvas = $("equity-chart");
  if (!canvas || typeof Chart === "undefined") return;
  const ctx = canvas.getContext("2d");
  const gradient = ctx.createLinearGradient(0, 0, 0, 240);
  gradient.addColorStop(0, "rgba(61, 139, 253, 0.35)");
  gradient.addColorStop(1, "rgba(61, 139, 253, 0.0)");

  if (equityChart) {
    equityChart.data.labels = labels;
    equityChart.data.datasets[0].data = data;
    equityChart.update("none");
    return;
  }

  Chart.defaults.color = "#6b7c99";
  Chart.defaults.font.family = "IBM Plex Mono";

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
          borderWidth: 2.2,
          fill: true,
          tension: 0.28,
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
          backgroundColor: "rgba(8,12,20,0.94)",
          borderColor: "rgba(120,150,200,0.2)",
          borderWidth: 1,
          padding: 10,
          callbacks: { label: (c) => " " + fmtUsd(c.parsed.y) },
        },
      },
      scales: {
        x: {
          ticks: { maxTicksLimit: 6, font: { size: 10 } },
          grid: { color: "rgba(255,255,255,0.04)" },
        },
        y: {
          ticks: {
            font: { size: 10 },
            callback: (v) => "$" + Number(v).toLocaleString(),
          },
          grid: { color: "rgba(255,255,255,0.05)" },
        },
      },
    },
  });
}

function updatePositions(positions) {
  const body = $("positions-body");
  const list = positions || [];
  setText("open-count", String(list.length));
  if (!list.length) {
    body.innerHTML =
      '<tr><td colspan="8" class="empty">No open positions</td></tr>';
    return;
  }
  body.innerHTML = list
    .map((p) => {
      const upnl = Number(p.unrealized_pnl || 0);
      const side = (p.side || "").toLowerCase();
      const pair = (p.symbol || "").split("/")[0] || p.symbol;
      return `<tr>
        <td class="mono">${escapeHtml(pair)}</td>
        <td class="${side === "long" ? "side-long" : "side-short"}">${side}</td>
        <td class="mono">${Number(p.entry_price).toFixed(4)}</td>
        <td class="mono">${Number(p.size).toFixed(4)}</td>
        <td class="mono ${upnl >= 0 ? "pos" : "neg"}">${fmtUsd(upnl)}</td>
        <td class="mono">${Number(p.stop_loss || 0).toFixed(4)}</td>
        <td class="mono">${Number(p.take_profit || 0).toFixed(4)}</td>
        <td>${escapeHtml(p.strategy || "—")}</td>
      </tr>`;
    })
    .join("");
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
        <td class="mono">${escapeHtml(pair)}</td>
        <td class="${side === "long" ? "side-long" : "side-short"}">${side}</td>
        <td>${escapeHtml(t.strategy || "—")}</td>
        <td class="mono ${pnl >= 0 ? "pos" : "neg"}">${fmtUsd(pnl)}</td>
        <td class="mono" style="color:var(--faint)">${escapeHtml(t.exit_reason || "—")}</td>
      </tr>`;
    })
    .join("");
}

function renderStatList(el, items) {
  if (!items || !items.length) {
    el.innerHTML =
      '<div class="stat-row"><span class="name" style="color:var(--muted)">No data yet</span></div>';
    return;
  }
  const peak = Math.max(...items.map((x) => Math.abs(x.pnl || 0)), 1);
  el.innerHTML = items
    .map((s) => {
      const pnl = s.pnl || 0;
      const width = Math.min(100, (Math.abs(pnl) / peak) * 100);
      const wr =
        s.win_rate != null
          ? fmtPct(s.win_rate, 0)
          : s.wins != null
            ? s.wins + "W"
            : "";
      const bar =
        pnl >= 0
          ? "linear-gradient(90deg,#3d8bfd,#22d3a6)"
          : "linear-gradient(90deg,#ff5c7a,#ff8fa3)";
      return `<div class="stat-row">
        <span class="name">${escapeHtml(s.name)}</span>
        <span class="pnl ${pnl >= 0 ? "pos" : "neg"}">${fmtUsd(pnl)}</span>
        <span class="stat-meta">${s.trades || 0} trades · ${wr}</span>
        <div class="mini-bar"><i style="width:${width}%;background:${bar}"></i></div>
      </div>`;
    })
    .join("");
}

function updateLogs(payload) {
  const stream = $("log-stream");
  if (payload.path) {
    const short = payload.path.replace(/\\/g, "/").split("/").slice(-2).join("/");
    setText("log-path", short);
  }
  const lines = payload.lines || [];
  if (!lines.length) {
    stream.innerHTML = '<div class="log-line muted">No log lines</div>';
    return;
  }
  stream.innerHTML = lines
    .map((l) => {
      const level = (l.level || "INFO").toUpperCase();
      const ts = l.ts
        ? `<span style="color:#5a6b88">${escapeHtml(l.ts)}</span> `
        : "";
      return `<div class="log-line ${level}">${ts}<span class="lvl">${level}</span>${escapeHtml(l.message || "")}</div>`;
    })
    .join("");
  stream.scrollTop = stream.scrollHeight;
}

function updateSymbols(symbols) {
  const el = $("symbol-chips");
  if (!symbols || !symbols.length) {
    el.innerHTML = '<span class="chip">No symbols</span>';
    return;
  }
  el.innerHTML = symbols
    .map((s) => `<span class="chip">${escapeHtml(s)}</span>`)
    .join("");
}

async function refresh() {
  try {
    const [overview, logs] = await Promise.all([
      fetch("/api/overview", { cache: "no-store" }).then((r) => r.json()),
      fetch("/api/logs?lines=100", { cache: "no-store" }).then((r) => r.json()),
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
    updateChart(
      m.equity_curve || [{ i: 0, equity: m.initial_capital || 10000 }]
    );
    updatePositions(m.open_positions || []);
    updateTrades(m.trades || []);
    renderStatList($("strat-list"), m.strategy_stats || []);
    renderStatList($("pair-list"), m.pair_stats || []);
    updateLogs(logs);
    updateSymbols(bot.symbols || []);
    await refreshExportInfo();
  } catch (err) {
    console.error(err);
    $("live-text").textContent = "API error";
    setText("footer-status", "api unreachable");
  }
}

/* Mobile nav */
const menuBtn = $("menu-btn");
const mobileNav = $("mobile-nav");
if (menuBtn && mobileNav) {
  menuBtn.addEventListener("click", () => {
    const open = mobileNav.classList.toggle("open");
    mobileNav.classList.toggle("hidden", !open);
    menuBtn.setAttribute("aria-expanded", open ? "true" : "false");
  });
  mobileNav.querySelectorAll("a").forEach((a) => {
    a.addEventListener("click", () => {
      mobileNav.classList.remove("open");
      mobileNav.classList.add("hidden");
      menuBtn.setAttribute("aria-expanded", "false");
    });
  });
}

$("refresh-btn").addEventListener("click", refresh);

/* ---------- Export downloads ---------- */
let exportPeriod = "today";

function setExportPeriod(period) {
  exportPeriod = period || "today";
  document.querySelectorAll(".period-chip").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.period === exportPeriod);
  });
  const log = $("dl-log");
  const bundle = $("dl-bundle");
  if (log) log.href = `/api/export/log?period=${encodeURIComponent(exportPeriod)}`;
  if (bundle) bundle.href = `/api/export/bundle?period=${encodeURIComponent(exportPeriod)}`;
  const hint = $("export-hint");
  if (hint) {
    hint.innerHTML =
      "Selected period: <strong>" +
      exportPeriod +
      "</strong> · Prefer <strong>Bundle ZIP</strong> when sharing for review.";
  }
}

document.querySelectorAll(".period-chip").forEach((btn) => {
  btn.addEventListener("click", () => setExportPeriod(btn.dataset.period));
});
setExportPeriod("today");

async function refreshExportInfo() {
  try {
    const info = await fetch("/api/export/info", { cache: "no-store" }).then((r) =>
      r.json()
    );
    const bits = [];
    if (info.state_exists) bits.push("state " + formatBytes(info.state_size));
    else bits.push("no state");
    if (info.log_exists) bits.push("log " + formatBytes(info.log_size));
    else bits.push("no log");
    setText("export-meta", bits.join(" · "));
  } catch {
    setText("export-meta", "export n/a");
  }
}

function formatBytes(n) {
  n = Number(n) || 0;
  if (n < 1024) return n + " B";
  if (n < 1024 * 1024) return (n / 1024).toFixed(1) + " KB";
  return (n / (1024 * 1024)).toFixed(1) + " MB";
}

/* PWA service worker */
if ("serviceWorker" in navigator) {
  window.addEventListener("load", () => {
    navigator.serviceWorker.register("/sw.js").catch((e) => {
      console.warn("SW register failed", e);
    });
  });
}

refresh();
setInterval(refresh, REFRESH_MS);
