/**
 * app.js - Frontend Controller for Crowdsourced Validation Simulation.
 * Implements user submission simulation, CatBoost inference feedback,
 * peer benchmark visualizer, explore data viewer, and admin moderation.
 */

// Global state
let currentScenarios = {};
let allSubmissions = [];
let currentUserTrust = 1.0;
let currentUserId = "u_genuine_01";
let activeSubmission = null;

// Initialization
document.addEventListener("DOMContentLoaded", () => {
  fetchScenarios();
  fetchSystemStats();
  fetchSubmissions();
  fetchUserProfile();
});

// Portal Switcher (User vs Admin)
function switchPortal(portal) {
  document.querySelectorAll(".portal-view").forEach(el => el.classList.remove("active"));
  document.querySelectorAll(".nav-tab").forEach(el => el.classList.remove("active"));

  if (portal === "admin") {
    document.getElementById("admin-portal").classList.add("active");
    document.getElementById("nav-admin-tab").classList.add("active");
    loadAdminQueue();
    loadAdminReports();
    fetchSystemStats();
  } else {
    document.getElementById("user-portal").classList.add("active");
    document.getElementById("nav-user-tab").classList.add("active");
  }
}

// User Sub-Navigation Switcher
function switchUserTab(tab) {
  document.querySelectorAll(".tab-content").forEach(el => el.classList.remove("active"));
  document.querySelectorAll(".sub-nav-btn").forEach(el => el.classList.remove("active"));

  if (tab === "explore") {
    document.getElementById("user-tab-explore").classList.add("active");
    document.getElementById("btn-subnav-explore").classList.add("active");
    fetchSubmissions();
  } else if (tab === "profile") {
    document.getElementById("user-tab-profile").classList.add("active");
    document.getElementById("btn-subnav-profile").classList.add("active");
    fetchUserProfile();
  } else {
    document.getElementById("user-tab-simulate").classList.add("active");
    document.getElementById("btn-subnav-simulate").classList.add("active");
  }
}

// ==========================================================================
// Scenario Presets
// ==========================================================================

async function fetchScenarios() {
  try {
    const res = await fetch("/api/scenarios");
    if (!res.ok) return;
    currentScenarios = await res.json();
    renderScenarios(currentScenarios);
  } catch (err) {
    console.error("Failed to load scenarios:", err);
  }
}

function renderScenarios(scenarios) {
  const container = document.getElementById("scenarios-container");
  container.innerHTML = "";

  for (const [key, item] of Object.entries(scenarios)) {
    const card = document.createElement("div");
    card.className = "scenario-btn";
    card.onclick = () => loadScenario(key);

    const badgeClass = item.badge === "Genuine" ? "scenario-badge-genuine" : "scenario-badge-defect";

    card.innerHTML = `
      <div>
        <span class="scenario-badge ${badgeClass}">${item.badge}</span>
        <div class="scenario-name">${item.title}</div>
      </div>
      <div class="scenario-desc-text">${item.description}</div>
    `;
    container.appendChild(card);
  }
}

function loadScenario(key) {
  const item = currentScenarios[key];
  if (!item || !item.data) return;
  const d = item.data;

  document.getElementById("input-company").value = d.company || "";
  document.getElementById("input-jobFamily").value = d.jobFamily || "Software Engineer";
  document.getElementById("input-level").value = d.level || "";
  document.getElementById("input-location").value = d.location || "";
  document.getElementById("input-workArrangement").value = d.workArrangement || "hybrid";
  document.getElementById("input-baseSalary").value = d.baseSalary || "";
  document.getElementById("input-currency").value = d.baseSalaryCurrency || "USD";
  document.getElementById("input-totalCompensation").value = d.totalCompensation || "";
  document.getElementById("input-stock").value = d.avgAnnualStockGrantValue || "";
  document.getElementById("input-bonus").value = d.avgAnnualBonusValue || "";
  document.getElementById("input-yoe").value = d.yearsOfExperience !== undefined ? d.yearsOfExperience : "";
  document.getElementById("input-yac").value = d.yearsAtCompany !== undefined ? d.yearsAtCompany : "";
  document.getElementById("input-yal").value = d.yearsAtLevel !== undefined ? d.yearsAtLevel : "";
  document.getElementById("input-userId").value = d.userId || currentUserId;
  document.getElementById("input-ipHash").value = d.ipHash || "";
  document.getElementById("input-deviceHash").value = d.deviceHash || "";

  // Auto trigger validation
  validateCurrentForm();
}

function calculateTotalCompensation() {
  const base = parseFloat(document.getElementById("input-baseSalary").value) || 0;
  const stock = parseFloat(document.getElementById("input-stock").value) || 0;
  const bonus = parseFloat(document.getElementById("input-bonus").value) || 0;
  const tcInput = document.getElementById("input-totalCompensation");
  if (base > 0) {
    tcInput.value = base + stock + bonus;
  }
}

function handleJobFamilyChange() {
  // Suggest a default level if empty
  const lvlInput = document.getElementById("input-level");
  if (!lvlInput.value) {
    lvlInput.value = "IC2";
  }
}

function resetForm() {
  document.getElementById("submission-form").reset();
  document.getElementById("result-content").style.display = "none";
  document.getElementById("result-empty-state").style.display = "flex";
}

// ==========================================================================
// Form Submission & Validation
// ==========================================================================

function handleFormSubmit(e) {
  e.preventDefault();
  validateCurrentForm();
}

async function validateCurrentForm() {
  const btn = document.getElementById("btn-validate");
  btn.disabled = true;
  btn.innerHTML = `<span class="status-dot"></span> Evaluating Model...`;

  const payload = {
    company: document.getElementById("input-company").value.trim(),
    jobFamily: document.getElementById("input-jobFamily").value,
    title: document.getElementById("input-jobFamily").value,
    level: document.getElementById("input-level").value.trim(),
    location: document.getElementById("input-location").value.trim(),
    workArrangement: document.getElementById("input-workArrangement").value,
    baseSalary: parseFloat(document.getElementById("input-baseSalary").value) || 0,
    baseSalaryCurrency: document.getElementById("input-currency").value,
    totalCompensation: parseFloat(document.getElementById("input-totalCompensation").value) || 0,
    avgAnnualStockGrantValue: parseFloat(document.getElementById("input-stock").value) || 0,
    avgAnnualBonusValue: parseFloat(document.getElementById("input-bonus").value) || 0,
    yearsOfExperience: parseFloat(document.getElementById("input-yoe").value) || 0,
    yearsAtCompany: parseFloat(document.getElementById("input-yac").value) || 0,
    yearsAtLevel: parseFloat(document.getElementById("input-yal").value) || 0,
    userId: document.getElementById("input-userId").value.trim() || currentUserId,
    ipHash: document.getElementById("input-ipHash").value.trim() || "ip_sim_user",
    deviceHash: document.getElementById("input-deviceHash").value.trim() || "dev_sim_user",
    submittedAt: new Date().toISOString()
  };

  try {
    const res = await fetch("/api/submissions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error("Evaluation error: " + res.statusText);
    const result = await res.json();
    activeSubmission = result;
    renderValidationResult(result);
    fetchUserProfile();
    fetchSystemStats();
  } catch (err) {
    alert("Validation failed: " + err.message);
  } finally {
    btn.disabled = false;
    btn.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
        <path d="m9 12 2 2 4-4"/>
      </svg>
      Validate & Assess Quality
    `;
  }
}

function renderValidationResult(res) {
  document.getElementById("result-empty-state").style.display = "none";
  const content = document.getElementById("result-content");
  content.style.display = "block";

  // Decision Badge
  const badge = document.getElementById("res-decision-badge");
  badge.className = "decision-badge";
  const dec = res.classification || "Review";
  if (dec === "Approve" || dec === "High Quality") {
    badge.textContent = "HIGH QUALITY (APPROVE)";
    badge.classList.add("badge-approve");
  } else if (dec === "Reject" || dec === "Potentially Problematic") {
    badge.textContent = "POTENTIALLY PROBLEMATIC (REJECT)";
    badge.classList.add("badge-reject");
  } else {
    badge.textContent = "REVIEW RECOMMENDED";
    badge.classList.add("badge-review");
  }

  document.getElementById("res-submission-id").textContent = `ID: ${(res.submissionId || "").slice(0, 8)}`;

  // Quality Score & Circular Dial
  const qScore = res.qualityScore !== undefined ? res.qualityScore : 50;
  document.getElementById("res-quality-score").textContent = qScore.toFixed(1);

  const circle = document.getElementById("res-score-circle");
  circle.className = "circle";
  circle.style.strokeDasharray = `${Math.min(100, Math.max(0, qScore))}, 100`;
  if (qScore >= 75) {
    circle.classList.add("circle-green");
  } else if (qScore >= 45) {
    circle.classList.add("circle-amber");
  } else {
    circle.classList.add("circle-red");
  }

  // Mini metrics
  const pAnom = (res.p_anomaly || 0) * 100;
  document.getElementById("res-p-anomaly").textContent = `${pAnom.toFixed(1)}%`;
  document.getElementById("res-bar-anomaly").style.width = `${Math.min(100, pAnom)}%`;

  const rScore = (res.rule_score || 1.0) * 100;
  document.getElementById("res-rule-score").textContent = `${rScore.toFixed(0)}%`;
  document.getElementById("res-bar-rule").style.width = `${Math.min(100, rScore)}%`;

  const cLevel = res.confidenceScore || 90;
  document.getElementById("res-conf-level").textContent = `${cLevel.toFixed(1)}%`;
  document.getElementById("res-bar-conf").style.width = `${Math.min(100, cLevel)}%`;

  const uTrust = res.user_trust !== undefined ? res.user_trust : 1.0;
  document.getElementById("res-user-trust").textContent = uTrust.toFixed(2);
  const deltaEl = document.getElementById("res-trust-delta");
  if (dec === "Approve" || dec === "High Quality") {
    deltaEl.textContent = "+0.03 Reward";
    deltaEl.style.color = "var(--success-text)";
  } else if (dec === "Reject" || dec === "Potentially Problematic") {
    deltaEl.textContent = "-0.35 Penalty";
    deltaEl.style.color = "var(--danger-text)";
  } else {
    deltaEl.textContent = "No Change";
    deltaEl.style.color = "var(--text-muted)";
  }

  // Update top header user trust badge
  document.getElementById("header-user-trust-val").textContent = uTrust.toFixed(2);

  // Peer Benchmark Bar
  document.getElementById("res-peer-group-title").textContent = `${res.company} | ${res.jobFamily} | ${res.level}`;
  const peerMed = res.peer_median_usd;
  const userBase = res.baseSalary || 0;
  const pin = document.getElementById("res-peer-pin");
  const zScore = res.peer_z_score || 0;
  document.getElementById("res-peer-z").textContent = zScore >= 0 ? `+${zScore.toFixed(2)}` : zScore.toFixed(2);

  if (peerMed && peerMed > 0) {
    const low = Math.round(peerMed * 0.65);
    const high = Math.round(peerMed * 1.5);
    document.getElementById("res-peer-low").textContent = `$${(low / 1000).toFixed(0)}k`;
    document.getElementById("res-peer-median").textContent = `Peer Median: $${(peerMed / 1000).toFixed(0)}k USD`;
    document.getElementById("res-peer-high").textContent = `$${(high / 1000).toFixed(0)}k`;

    // Position of pin in 0 to 100%
    const ratio = (userBase - low) / (high - low);
    const clampedPct = Math.max(5, Math.min(95, ratio * 100));
    pin.style.left = `${clampedPct}%`;

    const diffPct = ((userBase - peerMed) / peerMed) * 100;
    if (Math.abs(diffPct) < 20) {
      document.getElementById("res-peer-diff-text").textContent = "Well aligned with peer median";
      document.getElementById("res-peer-diff-text").style.color = "var(--success-text)";
    } else if (diffPct > 0) {
      document.getElementById("res-peer-diff-text").textContent = `${Math.abs(diffPct).toFixed(0)}% above peer median`;
      document.getElementById("res-peer-diff-text").style.color = diffPct > 50 ? "var(--danger-text)" : "var(--warning-text)";
    } else {
      document.getElementById("res-peer-diff-text").textContent = `${Math.abs(diffPct).toFixed(0)}% below peer median`;
      document.getElementById("res-peer-diff-text").style.color = diffPct < -40 ? "var(--danger-text)" : "var(--warning-text)";
    }
  } else {
    document.getElementById("res-peer-median").textContent = "Peer group sample too small";
    pin.style.left = "50%";
    document.getElementById("res-peer-diff-text").textContent = "Assessed via global hierarchical fallback";
  }

  // Explanations List
  const reasonsList = document.getElementById("res-reasons-list");
  reasonsList.innerHTML = "";
  const reasons = res.reasons || (res.explanation ? res.explanation.split("; ") : []);
  if (reasons.length === 0) {
    reasons.push("All logical consistency and peer compensation rules satisfied.");
  }
  reasons.forEach(r => {
    const li = document.createElement("li");
    li.textContent = r;
    reasonsList.appendChild(li);
  });
}

// ==========================================================================
// Explore Data Tab
// ==========================================================================

async function fetchSubmissions() {
  try {
    const res = await fetch("/api/submissions?limit=100");
    if (!res.ok) return;
    allSubmissions = await res.json();
    renderExploreTable(allSubmissions);
  } catch (err) {
    console.error("Failed to fetch submissions:", err);
  }
}

function renderExploreTable(list) {
  const tbody = document.getElementById("explore-table-body");
  if (!list || list.length === 0) {
    tbody.innerHTML = `<tr><td colspan="10" class="text-center">No submissions recorded yet.</td></tr>`;
    return;
  }

  tbody.innerHTML = list.map(item => {
    const dec = item.classification || "Review";
    const pillClass = dec === "Approve" ? "pill-approve" : (dec === "Reject" ? "pill-reject" : "pill-review");
    const qScore = item.qualityScore ? item.qualityScore.toFixed(1) : "-";
    const pAnom = item.p_anomaly !== undefined ? `${(item.p_anomaly * 100).toFixed(1)}%` : "-";

    return `
      <tr>
        <td><span class="pill-sm ${pillClass}">${dec}</span></td>
        <td><strong>${item.company || "Unknown"}</strong></td>
        <td>${item.jobRole || item.jobFamily || "-"} <span class="text-muted">(${item.level || "-"})</span></td>
        <td>${item.location || "-"}</td>
        <td>${formatMoney(item.baseSalary, item.baseSalaryCurrency)}</td>
        <td>${formatMoney(item.totalCompensation, item.baseSalaryCurrency)}</td>
        <td>${item.yearsOfExperience || 0} yrs</td>
        <td><strong>${qScore}</strong></td>
        <td>${pAnom}</td>
        <td>
          <button class="btn btn-outline btn-sm" onclick="inspectSubmission('${item.submissionId}')">Inspect</button>
        </td>
      </tr>
    `;
  }).join("");
}

function handleExploreSearch() {
  const query = document.getElementById("explore-search").value.toLowerCase();
  const filter = document.getElementById("explore-status-filter").value;
  applyExploreFilters(query, filter);
}

function handleExploreFilter() {
  const query = document.getElementById("explore-search").value.toLowerCase();
  const filter = document.getElementById("explore-status-filter").value;
  applyExploreFilters(query, filter);
}

function applyExploreFilters(query, statusFilter) {
  let filtered = allSubmissions;
  if (statusFilter) {
    filtered = filtered.filter(s => s.classification === statusFilter);
  }
  if (query) {
    filtered = filtered.filter(s =>
      (s.company && s.company.toLowerCase().includes(query)) ||
      (s.jobFamily && s.jobFamily.toLowerCase().includes(query)) ||
      (s.jobRole && s.jobRole.toLowerCase().includes(query)) ||
      (s.location && s.location.toLowerCase().includes(query))
    );
  }
  renderExploreTable(filtered);
}

function inspectSubmission(subId) {
  const sub = allSubmissions.find(s => s.submissionId === subId);
  if (!sub) return;
  activeSubmission = sub;
  switchUserTab("simulate");
  renderValidationResult(sub);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// ==========================================================================
// User Profile & Trust
// ==========================================================================

async function fetchUserProfile() {
  try {
    const res = await fetch(`/api/user/profile?userId=${currentUserId}`);
    if (!res.ok) return;
    const data = await res.json();
    const user = data.user || {};
    const history = data.history || [];

    currentUserTrust = user.trustScore !== undefined ? user.trustScore : 1.0;
    document.getElementById("profile-trust-hero").textContent = currentUserTrust.toFixed(2);
    document.getElementById("header-user-trust-val").textContent = currentUserTrust.toFixed(2);

    const histTbody = document.getElementById("profile-history-body");
    if (history.length === 0) {
      histTbody.innerHTML = `<tr><td colspan="5" class="text-center">No previous submissions on this account.</td></tr>`;
      return;
    }

    histTbody.innerHTML = history.slice(0, 10).map(s => `
      <tr>
        <td>${(s.submittedAt || "").slice(0, 10)}</td>
        <td>${s.jobRole || s.jobFamily}</td>
        <td><strong>${s.company}</strong></td>
        <td><span class="pill-sm ${s.classification === 'Approve' ? 'pill-approve' : (s.classification === 'Reject' ? 'pill-reject' : 'pill-review')}">${s.classification}</span></td>
        <td>${s.qualityScore ? s.qualityScore.toFixed(1) : "-"}</td>
      </tr>
    `).join("");
  } catch (err) {
    console.error("Failed to load user profile:", err);
  }
}

// ==========================================================================
// Administrator Portal
// ==========================================================================

async function fetchSystemStats() {
  try {
    const res = await fetch("/api/stats");
    if (!res.ok) return;
    const stats = await res.json();

    document.getElementById("adm-stat-total").textContent = stats.totalSubmissions;
    document.getElementById("adm-stat-approved").textContent = `${stats.approvalRate}%`;
    document.getElementById("adm-stat-review").textContent = stats.reviewCount;
    document.getElementById("adm-stat-rejected").textContent = stats.rejectedCount;

    if (stats.modelLoaded) {
      document.getElementById("server-status-pill").className = "status-pill status-pill-online";
      document.getElementById("server-status-text").textContent = "CatBoost Active";
    }
  } catch (err) {
    console.error("Failed to load stats:", err);
  }
}

async function loadAdminQueue() {
  try {
    const res = await fetch("/api/submissions?limit=100");
    if (!res.ok) return;
    const subs = await res.json();
    // Filter for review or reject
    const queue = subs.filter(s => s.classification === "Review" || s.classification === "Reject");
    const tbody = document.getElementById("admin-queue-body");

    if (queue.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8" class="text-center">Moderation queue clean! No pending flagged records.</td></tr>`;
      return;
    }

    tbody.innerHTML = queue.map(s => `
      <tr>
        <td><span class="pill-sm ${s.classification === 'Reject' ? 'pill-reject' : 'pill-review'}">${s.classification}</span></td>
        <td><strong>${s.company}</strong></td>
        <td>${s.jobRole || s.jobFamily} (${s.level})</td>
        <td>${formatMoney(s.baseSalary, s.baseSalaryCurrency)}</td>
        <td>${formatMoney(s.totalCompensation, s.baseSalaryCurrency)}</td>
        <td><strong>${((s.p_anomaly || 0) * 100).toFixed(1)}%</strong></td>
        <td><small class="text-secondary">${s.explanation || "Flagged for audit"}</small></td>
        <td>
          <button class="btn btn-primary btn-sm" onclick="openOverrideModal('${s.submissionId}')">Audit</button>
        </td>
      </tr>
    `).join("");
  } catch (err) {
    console.error("Failed to load admin queue:", err);
  }
}

async function loadAdminReports() {
  try {
    const res = await fetch("/api/reports");
    if (!res.ok) return;
    const reports = await res.json();
    const tbody = document.getElementById("admin-reports-body");

    if (reports.length === 0) {
      tbody.innerHTML = `<tr><td colspan="5" class="text-center">No user discrepancy tickets filed.</td></tr>`;
      return;
    }

    tbody.innerHTML = reports.map(r => `
      <tr>
        <td><code>${(r.reportId || "").slice(0, 8)}</code></td>
        <td><strong>${r.category || r.reportType || "Feedback"}</strong></td>
        <td>${r.description || "-"}</td>
        <td><span class="pill-sm pill-review">${r.status || "Open"}</span></td>
        <td>${(r.createdAt || "").slice(0, 10)}</td>
      </tr>
    `).join("");
  } catch (err) {
    console.error("Failed to load reports:", err);
  }
}

// Modal Controllers
function openOverrideModal(subId) {
  const sub = allSubmissions.find(s => s.submissionId === subId);
  if (!sub) return;

  document.getElementById("override-sub-id").value = subId;
  document.getElementById("override-decision").value = sub.classification;
  document.getElementById("override-comments").value = "";

  document.getElementById("override-sub-details").innerHTML = `
    <div><strong>Submission:</strong> ${sub.company} - ${sub.jobRole} (${sub.level})</div>
    <div><strong>Base Salary:</strong> ${formatMoney(sub.baseSalary, sub.baseSalaryCurrency)} | <strong>TC:</strong> ${formatMoney(sub.totalCompensation, sub.baseSalaryCurrency)}</div>
    <div><strong>CatBoost Anomaly Score:</strong> ${((sub.p_anomaly || 0) * 100).toFixed(1)}% | <strong>Trust:</strong> ${sub.user_trust || 1.0}</div>
    <div><strong>Current Explanation:</strong> ${sub.explanation}</div>
  `;
  document.getElementById("admin-override-modal").style.display = "flex";
}

function closeOverrideModal() {
  document.getElementById("admin-override-modal").style.display = "none";
}

async function handleOverrideSubmit(e) {
  e.preventDefault();
  const subId = document.getElementById("override-sub-id").value;
  const decision = document.getElementById("override-decision").value;
  const comments = document.getElementById("override-comments").value;

  try {
    const res = await fetch("/api/admin/review", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ submissionId: subId, decision, comments, adminId: "admin" })
    });
    if (!res.ok) throw new Error("Override failed");
    closeOverrideModal();
    loadAdminQueue();
    fetchSystemStats();
    fetchSubmissions();
    alert("Decision override saved successfully!");
  } catch (err) {
    alert("Override error: " + err.message);
  }
}

function openReportModal() {
  document.getElementById("report-modal").style.display = "flex";
}

function closeReportModal() {
  document.getElementById("report-modal").style.display = "none";
}

async function handleReportSubmit(e) {
  e.preventDefault();
  const category = document.getElementById("report-type").value;
  const description = document.getElementById("report-desc").value;

  const payload = {
    submissionId: activeSubmission ? activeSubmission.submissionId : null,
    category,
    description,
    reportedBy: currentUserId
  };

  try {
    const res = await fetch("/api/reports", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error("Report submission failed");
    closeReportModal();
    alert("Thank you! Your feedback has been sent to the moderation queue.");
  } catch (err) {
    alert("Report error: " + err.message);
  }
}

// Helpers
function formatMoney(amount, currency = "USD") {
  if (!amount && amount !== 0) return "-";
  const num = Number(amount);
  const symbolMap = { "USD": "$", "INR": "₹", "EUR": "€", "GBP": "£", "CAD": "C$", "SGD": "S$", "AUD": "A$" };
  const sym = symbolMap[currency] || `${currency} `;
  return `${sym}${num.toLocaleString()}`;
}
