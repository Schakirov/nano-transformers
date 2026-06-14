/* nano-transformers — front-end. Vanilla JS, no dependencies.
   Three jobs: (1) tab navigation, (2) the live training dashboard that polls
   web/runs/<task>.json, (3) the code / PTX / SASS viewers. */

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

/* ---------------------------------------------------------------- nav --- */
/* only the view tabs navigate; the theme toggle (.tab.toggle, no data-view)
   is wired separately below and must be excluded here. */
$$(".tab[data-view]").forEach(t => t.onclick = () => {
  $$(".tab[data-view]").forEach(x => x.classList.remove("active"));
  $$(".view").forEach(x => x.classList.remove("active"));
  t.classList.add("active");
  $("#" + t.dataset.view).classList.add("active");
  if (t.dataset.view === "gpu") loadKernel();
  if (t.dataset.view === "code") loadCode();
  if (t.dataset.view === "train") fetchRun();   // redraw now the canvases have width
});

/* ------------------------------------------------------------- theme --- */
/* The actual theme is set pre-paint by the inline script in <head>; here we
   only flip it on click and remember the choice. The glyph shows the theme
   you'll switch TO: ☀ in dark mode, ☾ in light mode. */
const themeBtn = $("#themeToggle");
function syncThemeIcon(){
  const light = document.documentElement.getAttribute("data-theme") === "light";
  themeBtn.textContent = light ? "☾" : "☀";
  themeBtn.title = light ? "Switch to dark theme" : "Switch to light theme";
}
function toggleTheme(){
  const next = document.documentElement.getAttribute("data-theme") === "light" ? "dark" : "light";
  document.documentElement.setAttribute("data-theme", next);
  try { localStorage.setItem("nt-theme", next); } catch (e) {}
  syncThemeIcon();
}
themeBtn.onclick = toggleTheme;
themeBtn.onkeydown = e => { if (e.key === "Enter" || e.key === " "){ e.preventDefault(); toggleTheme(); } };
syncThemeIcon();

/* ---------------------------------------------- tiny code highlighter --- */
function esc(s){return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");}
function highlight(src, kind){
  let h = esc(src);
  // block comments /* ... */
  h = h.replace(/\/\*[\s\S]*?\*\//g, m => `<span class="cm">${m}</span>`);
  // line comments: # for python/shell, // for c/ptx/sass
  const hash = (kind === "py" || kind === "sh");
  const lc = hash ? /(^|\n)(\s*#[^\n]*)/g : /(\/\/[^\n]*)/g;
  h = h.replace(lc, (m, a, b) => hash ? `${a}<span class="cm">${b}</span>` : `<span class="cm">${m}</span>`);
  return h;
}

/* ============================== DASHBOARD ============================== */
let task = "sort", poll = null;

$$("#taskbar .chip").forEach(c => c.onclick = () => {
  $$("#taskbar .chip").forEach(x => x.classList.remove("active"));
  c.classList.add("active"); task = c.dataset.task;
  fetchRun();                       // show last run for this task, if any
});

$("#trainBtn").onclick = async () => {
  $("#runState").innerHTML = '<span class="spin"></span> launching…';
  try {
    const r = await fetch("/api/train/" + task, { method: "POST" }).then(r => r.json());
    $("#runState").textContent = r.already ? "already running…" : "training…";
  } catch (e) { $("#runState").textContent = "could not reach server"; }
  if (poll) clearInterval(poll);
  poll = setInterval(fetchRun, 500);  // poll the live JSON twice a second
  fetchRun();
};

async function fetchRun(){
  let d;
  try { d = await fetch(`/web/runs/${task}.json?t=${Date.now()}`).then(r => r.json()); }
  catch { return; }
  render(d);
  if (d.done && poll){ clearInterval(poll); poll = null;
    $("#runState").textContent = `done · ${d.elapsed.toFixed(1)}s`; }
  else if (poll) $("#runState").innerHTML = '<span class="spin"></span> training…';
}

function render(d){
  $("#sStep").textContent  = d.step + " / " + d.max_steps;
  $("#sLoss").textContent  = d.loss.toFixed(3);
  $("#sAcc").textContent   = (d.acc * 100).toFixed(0) + "%";
  $("#sParams").textContent= (d.config.params/1000).toFixed(0) + "k";
  $("#sTime").textContent  = d.elapsed.toFixed(1);
  $("#sAcc").className = "v " + (d.acc > .8 ? "good" : "acc");

  drawCurves(d.loss_history, d.acc_history);
  drawSamples(d.samples);
  drawAttn(d.attn);
  drawHist(d.weights);
  drawWeights(d.weights);

  const ic = $("#imgCanvas"), il = $("#imgLabel");
  if (d.image){ ic.style.display = il.style.display = "block";
    drawImage(d.image); il.textContent = "input image · ground truth: " + (d.image_label||""); }
  else { ic.style.display = il.style.display = "none"; }
}

/* fit a canvas to its CSS box at device resolution */
function fit(cv){
  const w = cv.clientWidth, h = cv.height;
  if (cv.width !== w){ cv.width = w; }
  return [cv.getContext("2d"), w, h];
}

function drawCurves(loss, acc){
  const [g, W, H] = fit($("#lossChart"));
  g.clearRect(0,0,W,H);
  const pad = 28, x0 = pad, x1 = W - pad, y0 = H - pad, y1 = 10;
  if (!loss.length) return;
  const maxStep = Math.max(1, loss[loss.length-1][0]);
  const maxLoss = Math.max(...loss.map(p => p[1]), 0.1);
  const X = s => x0 + (x1-x0) * s / maxStep;
  // grid
  g.strokeStyle = "#1b2230"; g.lineWidth = 1;
  for (let i=0;i<=4;i++){ const y = y0+(y1-y0)*i/4; g.beginPath(); g.moveTo(x0,y); g.lineTo(x1,y); g.stroke(); }
  // loss (red, left scale)
  g.strokeStyle = "#ff6b6b"; g.lineWidth = 2; g.beginPath();
  loss.forEach(([s,v],i)=>{ const x=X(s), y=y0+(y1-y0)*(v/maxLoss);
    i?g.lineTo(x,y):g.moveTo(x,y); }); g.stroke();
  // accuracy (blue, 0..1)
  g.strokeStyle = "#5ad1ff"; g.beginPath();
  acc.forEach(([s,v],i)=>{ const x=X(s), y=y0+(y1-y0)*v;
    i?g.lineTo(x,y):g.moveTo(x,y); }); g.stroke();
  g.fillStyle = "#5f6b80"; g.font = "11px monospace";
  g.fillText("0", x0-4, y0+12); g.fillText("100%", x0-4, y1+4);
  g.fillText("step "+maxStep, x1-54, y0+14);
}

function drawSamples(samples){
  const el = $("#samples");
  el.innerHTML = samples.map(s =>
    `<div class="row"><span class="${s.correct?'ok':'no'}">${s.correct?'✓':'✗'}</span>`+
    `<span>${esc(s.text)}</span></div>`).join("");
}

/* generic heatmap: mode 'div' = diverging blue/red about 0, 'seq' = black→amber */
function heat(ctx, m, x, y, w, h, mode, vmin, vmax){
  const R = m.length, C = m[0].length, cw = w/C, ch = h/R;
  const lim = mode==='div' ? Math.max(Math.abs(vmin), Math.abs(vmax), 1e-6) : 0;
  for (let i=0;i<R;i++) for (let j=0;j<C;j++){
    const v = m[i][j]; let col;
    if (mode==='div'){ const t = Math.max(-1,Math.min(1, v/lim));
      col = t>=0 ? `rgb(${255},${Math.round(255*(1-t))},${Math.round(255*(1-t))})`
                 : `rgb(${Math.round(255*(1+t))},${Math.round(255*(1+t))},255)`;
    } else { const t = (v-vmin)/((vmax-vmin)||1);
      col = `rgb(${Math.round(255*Math.min(1,t*1.4))},${Math.round(190*t)},${Math.round(60*(1-t))})`; }
    ctx.fillStyle = col; ctx.fillRect(x+j*cw, y+i*ch, Math.ceil(cw), Math.ceil(ch));
  }
}

function drawAttn(a){
  const [g,W,H] = fit($("#attnCanvas")); g.clearRect(0,0,W,H);
  if (!a || !a.length) return;
  let vmin=1e9,vmax=-1e9; a.forEach(r=>r.forEach(v=>{vmin=Math.min(vmin,v);vmax=Math.max(vmax,v);}));
  const s = Math.min(W, H) - 8;
  heat(g, a, (W-s)/2, 2, s, s, 'seq', vmin, vmax);
}

function drawHist(weights){
  const [g,W,H] = fit($("#histCanvas")); g.clearRect(0,0,W,H);
  const cols = weights.length, cw = W/cols, pad = 10;
  weights.forEach((w,k)=>{
    const counts = w.hist.counts, mx = Math.max(...counts,1), x0 = k*cw+pad, ww = cw-2*pad;
    const bw = ww/counts.length;
    g.fillStyle = ["#5ad1ff","#ffb454","#48d597","#ff6b6b"][k%4];
    counts.forEach((c,i)=>{ const bh = (H-26)*c/mx; g.fillRect(x0+i*bw, H-18-bh, Math.ceil(bw), bh); });
    g.fillStyle = "#8b94a7"; g.font = "10px monospace";
    g.fillText(w.name.split(" ")[0], x0, H-4);
  });
}

let wcanvases = null;
function drawWeights(weights){
  const grid = $("#weightGrid");
  if (!wcanvases || wcanvases.length !== weights.length){
    grid.innerHTML = ""; wcanvases = weights.map(w=>{
      const wrap = document.createElement("div");
      wrap.innerHTML = `<div class="label">${esc(w.name)} · ${w.shape.join("×")} · rms <span class="rms"></span></div>`;
      const cv = document.createElement("canvas"); cv.height = 150; wrap.appendChild(cv);
      grid.appendChild(wrap); return cv;
    });
  }
  weights.forEach((w,k)=>{
    const cv = wcanvases[k]; const [g,W,H] = fit(cv); g.clearRect(0,0,W,H);
    heat(g, w.heat, 0, 0, W, H, 'div', w.vmin, w.vmax);
    cv.parentElement.querySelector(".rms").textContent = w.norm.toFixed(3);
  });
}

function drawImage(img){
  const [g,W,H] = fit($("#imgCanvas")); g.clearRect(0,0,W,H);
  const R=img.length,C=img[0].length, s=Math.min(W,H)-4, cw=s/C, ch=s/R, ox=(W-s)/2;
  for(let i=0;i<R;i++)for(let j=0;j<C;j++){ const t=img[i][j]/3;
    g.fillStyle=`rgb(${255*t|0},${255*t|0},${255*t|0})`; g.fillRect(ox+j*cw,2+i*ch,Math.ceil(cw),Math.ceil(ch)); }
}

/* ============================== GPU VIEW ============================== */
let kernel = "matmul", asm = "ptx";
$$("#kernelTabs .ft").forEach(t => t.onclick = () => {
  $$("#kernelTabs .ft").forEach(x=>x.classList.remove("active")); t.classList.add("active");
  kernel = t.dataset.k;
  $("#noteMatmul").style.display = kernel==="matmul"?"block":"none";
  $("#noteSoftmax").style.display = kernel==="softmax"?"block":"none";
  loadKernel();
});
$$("#asmTabs .ft").forEach(t => t.onclick = () => {
  $$("#asmTabs .ft").forEach(x=>x.classList.remove("active")); t.classList.add("active");
  asm = t.dataset.a; loadAsm();
});

async function loadKernel(){
  $("#srcLabel").textContent = `kernels/${kernel}.cu — the CUDA C++ you write`;
  $("#srcPane").innerHTML = await fetchText(`/kernels/${kernel}.cu`, "c");
  loadAsm(); loadBench();
}
async function loadAsm(){
  const path = `/web/asm/${kernel}.${asm}`;
  $("#asmLabel").textContent = path.slice(1);
  $("#asmPane").innerHTML = await fetchText(path, "asm");
}
async function loadBench(){
  try{ const t = await fetch("/web/asm/bench.txt?t="+Date.now()).then(r=>r.text());
    $("#benchOut").innerHTML = esc(t); }catch{}
}
async function fetchText(path, kind){
  try{ const t = await fetch(path+"?t="+Date.now()).then(r=>{if(!r.ok)throw 0;return r.text();});
    return highlight(t, kind==="c"?"c":kind); }
  catch{ return `<span class="cm">(${path} not generated yet — run kernels/build.sh and kernels/dump.sh)</span>`; }
}

/* ============================== CODE VIEW ============================== */
const FILES = ["nanograd.py","model.py","tasks.py","train.py","serve.py",
               "kernels/matmul.cu","kernels/softmax.cu","kernels/dump.sh","README.md"];
let codeInit = false;
function loadCode(){
  if (codeInit) return; codeInit = true;
  const tabs = $("#codeTabs");
  FILES.forEach((f,i)=>{ const d=document.createElement("div");
    d.className="ft"+(i===0?" active":""); d.textContent=f; d.onclick=()=>showFile(f,d);
    tabs.appendChild(d); });
  showFile(FILES[0], tabs.firstChild);
}
async function showFile(f, tab){
  $$("#codeTabs .ft").forEach(x=>x.classList.remove("active")); tab.classList.add("active");
  $("#codeLabel").textContent = f;
  const kind = f.endsWith(".py")?"py":(f.endsWith(".cu")?"c":(f.endsWith(".md")?"md":"sh"));
  $("#codePane").innerHTML = await fetchText("/"+f, kind);
}

/* try to show the latest run for the default task on first load */
fetchRun();
