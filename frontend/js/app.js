const $ = id => document.getElementById(id), KEY = "Dr.Mail_history";
const pct = v => (v * 100).toFixed(1) + "%";
const esc = s => String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));

$("menu").onclick = () => $("links").classList.toggle("open");
$("links").onclick = () => $("links").classList.remove("open");
$("msg").oninput = () => $("count").textContent = $("msg").value.length + " / 50000";
$("clear").onclick = () => { $("msg").value = ""; $("msg").oninput(); $("result").hidden = $("err").hidden = true; };

async function loadInfo() {
  try {
    const r = await fetch("/api/model-info"); const m = await r.json();
    if (!r.ok) throw new Error(m.error);
    const items = [["Accuracy", m.accuracy], ["Precision", m.precision], ["Recall", m.recall], ["F1", m.f1]];
    $("stats").innerHTML = items.map(([n, v]) => `<div class="stat"><b>${pct(v)}</b>${n}</div>`).join("");
    const [[tn, fp], [fn, tp]] = m.confusion_matrix, d = m.dataset || {};
    $("metrics").innerHTML = `<h3>Evaluation (held-out test set)</h3>
      <p>${esc(m.model)} · ${esc(d.name)} · ${d.ham} ham / ${d.spam} spam after cleaning · train ${d.train_size} / test ${d.test_size}</p>
      <div class="cm"><span></span><div>Pred HAM</div><div>Pred SPAM</div>
      <div>True HAM</div><div style="background:#B7F36B">${tn}</div><div style="background:#FF5C5C">${fp}</div>
      <div>True SPAM</div><div style="background:#FF5C5C">${fn}</div><div style="background:#B7F36B">${tp}</div></div>
      <pre>${esc(m.report)}</pre>`;
  } catch (e) {
    $("stats").innerHTML = $("metrics").innerHTML = `<div class="card">Model metrics unavailable. Train the model first (see README).</div>`;
  }
}

$("go").onclick = async () => {
  const message = $("msg").value, err = $("err");
  err.hidden = true;
  if (!message.trim()) { err.textContent = "Please paste an email first."; err.hidden = false; return; }
  $("loader").hidden = false; $("go").disabled = true; $("result").hidden = true;
  try {
    const r = await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message }) });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || "Request failed");
    render(d); save(message, d);
  } catch (e) {
    err.textContent = e instanceof TypeError ? "Cannot reach the server." : e.message; err.hidden = false;
  } finally { $("loader").hidden = true; $("go").disabled = false; }
};

function render(d) {
  const spam = d.label === "SPAM", p = d.probabilities;
  $("result").hidden = false;
  $("result").innerHTML = `<span class="verdict ${spam ? "v-spam" : "v-ham"}">${spam ? "SPAM" : "HAM / NOT SPAM"}</span>
    <p>Model-estimated spam probability</p><div class="bar"><i style="width:${pct(p.spam)};background:#FF5C5C"></i></div>
    <p><b>SPAM ${pct(p.spam)}</b> · <b>HAM ${pct(p.ham)}</b></p>
    <p class="small">This is a machine learning prediction. Probabilities are model estimates, not measures of accuracy, and cannot guarantee an email is safe.</p>
    <button class="btn alt" onclick="$('clear').click();location.hash='analyze'">Analyze another email</button>`;
}

const getH = () => { try { return JSON.parse(localStorage.getItem(KEY)) || []; } catch { return []; } };
function save(msg, d) {
  const h = getH(); h.unshift({ p: msg.slice(0, 60).replace(/\s+/g, " "), l: d.label, t: new Date().toLocaleString() });
  try { localStorage.setItem(KEY, JSON.stringify(h.slice(0, 20))); } catch {}
  showH();
}
function showH() {
  const h = getH();
  $("hlist").innerHTML = h.length ? h.map(x => `<li><span>${esc(x.p)}…</span><span><span class="tag" style="background:${x.l === "SPAM" ? "#FF5C5C" : "#B7F36B"}">${x.l}</span> ${esc(x.t)}</span></li>`).join("") : "<li>No predictions yet.</li>";
}
$("hclear").onclick = () => { try { localStorage.removeItem(KEY); } catch {} showH(); };
showH(); loadInfo();
