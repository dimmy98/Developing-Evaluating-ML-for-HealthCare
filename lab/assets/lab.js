/* Portable model inference and local CSV exploration. No external scripts or telemetry. */
(function () {
  "use strict";
  const COLORS = ["#79b8ff", "#c2a2ff", "#45dfbf"];
  function treeValue(tree, x) {
    let node = 0;
    while (tree.left[node] !== -1) {
      node = Math.fround(x[tree.feature[node]]) <= tree.threshold[node] ? tree.left[node] : tree.right[node];
    }
    return tree.value[node];
  }
  function predictLog(model, x) {
    if (model.kind === "linear") return model.intercept + model.coef.reduce((s, c, i) => s + c * x[i], 0);
    if (model.kind === "tree") return treeValue(model.tree, x);
    return model.initial + model.learning_rate * model.trees.reduce((s, t) => s + treeValue(t, x), 0);
  }
  function predict(model, x) { return 10 ** predictLog(model, x); }
  function parseCSV(text) {
    text = text.replace(/^\uFEFF/, "");
    const rows = []; let row = [], cell = "", quoted = false, closed = false;
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (quoted) {
        if (c === '"' && text[i + 1] === '"') { cell += '"'; i++; }
        else if (c === '"') { quoted = false; closed = true; }
        else cell += c;
      } else if (c === '"' && cell === "" && !closed) quoted = true;
      else if (c === ",") { row.push(cell); cell = ""; closed = false; }
      else if (c === "\n" || c === "\r") {
        if (c === "\r" && text[i + 1] === "\n") i++;
        row.push(cell); if (row.some(v => v.trim())) rows.push(row);
        row = []; cell = ""; closed = false;
      } else {
        if (closed || c === '"') throw new Error("Malformed CSV quoting.");
        cell += c;
      }
    }
    if (quoted) throw new Error("Unclosed quoted CSV field.");
    row.push(cell); if (row.some(v => v.trim())) rows.push(row);
    if (!rows.length) throw new Error("The CSV is empty.");
    if (rows.length > 10001) throw new Error("Limit: 10,000 records per file.");
    return rows;
  }
  function validateCSV(text, bounds) {
    const rows = parseCSV(text), headers = rows.shift().map(v => v.trim().toLowerCase());
    if (new Set(headers).size !== headers.length) throw new Error("Duplicate column names are not supported.");
    const keys = ["age", "sex", "bmi", "smoker"], missing = keys.filter(k => !headers.includes(k));
    if (missing.length) throw new Error("Missing columns: " + missing.join(", "));
    const valid = [], issues = [], seen = new Set(); let invalid = 0, duplicates = 0;
    rows.forEach((cells, index) => {
      const rowNumber = index + 2;
      const fail = message => { invalid++; if (issues.length < 10) issues.push("Record " + rowNumber + ": " + message); };
      if (cells.length !== headers.length) { fail("column count does not match the header."); return; }
      const record = Object.fromEntries(headers.map((k, i) => [k, cells[i].trim().toLowerCase()]));
      const age = Number(record.age), bmi = Number(record.bmi);
      const sex = record.sex === "female" ? 0 : record.sex === "male" ? 1 : undefined;
      const smoker = record.smoker === "no" ? 0 : record.smoker === "yes" ? 1 : undefined;
      if (!record.age || !record.bmi || !Number.isFinite(age) || !Number.isFinite(bmi)) { fail("age and BMI must be finite numbers."); return; }
      if (sex === undefined || smoker === undefined) { fail("sex must be female/male and smoker must be no/yes."); return; }
      if (!Number.isInteger(age) || age < bounds.age[0] || age > bounds.age[1] || bmi < bounds.bmi[0] || bmi > bounds.bmi[1]) { fail("age or BMI is outside the training range, or age is not a whole number."); return; }
      let charges = null;
      if (headers.includes("charges")) {
        charges = Number(record.charges);
        if (!record.charges || !Number.isFinite(charges) || charges <= 0) { fail("charges must be a positive finite number when present."); return; }
      }
      const x = [age, sex, bmi, smoker], identity = JSON.stringify([x, charges]);
      if (seen.has(identity)) duplicates++;
      seen.add(identity); valid.push({row: rowNumber, x, charges});
    });
    return {valid, invalid, duplicates, total: rows.length, issues};
  }
  if (typeof module !== "undefined" && module.exports) module.exports = {predictLog, predict, parseCSV, validateCSV};
  if (typeof document === "undefined") return;
  const $ = id => document.getElementById(id);
  const money = n => new Intl.NumberFormat("en-US", {style:"currency",currency:"USD",maximumFractionDigits:0}).format(n);
  const label = f => ({age:"Age", sex:"Sex", bmi:"BMI", smoker:"Smoking status"}[f]);
  let report, csvState;
  function el(tag, text, cls) {
    const e = document.createElement(tag); if (text !== undefined) e.textContent = text; if (cls) e.className = cls; return e;
  }
  function stat(target, value, caption) {
    const d = el("div", undefined, "stat"); d.append(el("strong", value), el("span", caption)); target.append(d);
  }
  function tableRow(target, values) { const tr = el("tr"); values.forEach(v => tr.append(el("td", v))); target.append(tr); }
  function svg(content, title, desc) {
    return '<svg viewBox="0 0 560 290" role="img" aria-label="' + title + '"><title>' + title + '</title><desc>' + desc + '</desc>' + content + '</svg>';
  }
  function axes(xlabel, ylabel, xmax, ymax, xmin = 0, ymin = 0) {
    let s = '<path class="axis" fill="none" d="M62 22V244H540"/>';
    for (let i=0;i<5;i++) {
      const y=244-i*55.5, value=ymin+(ymax-ymin)*i/4;
      s+='<path class="grid" d="M62 '+y+'H540"/><text x="52" y="'+(y+4)+'" text-anchor="end">'+(Math.abs(value)>=1000 ? (value/1000).toFixed(0)+"k" : value.toFixed(0))+'</text>';
      const x=62+i*119.5, xv=xmin+(xmax-xmin)*i/4;
      s+='<text x="'+x+'" y="264" text-anchor="middle">'+(Math.abs(xv)>=1000 ? (xv/1000).toFixed(0)+"k" : xv.toFixed(0))+'</text>';
    }
    return s+'<text x="300" y="286" text-anchor="middle">'+xlabel+'</text><text x="62" y="13">'+ylabel+'</text>';
  }
  function scenario() { return [+$("age").value, +$("sex").value, +$("bmi").value, +$("smoker").value]; }
  function sensitivity(x) {
    const feature=$("sensitivity-feature").value, idx=report.features.indexOf(feature), [lo,hi]=report.bounds[feature];
    const curves=report.models.map(m => Array.from({length:65}, (_,i) => {const copy=x.slice();copy[idx]=lo+(hi-lo)*i/64;return {x:copy[idx],y:predict(m,copy)};}));
    const max=Math.max(...curves.flat().map(p=>p.y))*1.1;
    let s=axes(feature==="age"?"Age (years)":"BMI","Estimated charge · USD",hi,max,lo);
    curves.forEach((curve,i)=>{s+='<path fill="none" stroke="'+COLORS[i]+'" stroke-width="2.5" d="'+curve.map((p,j)=>(j?"L":"M")+(62+(p.x-lo)/(hi-lo)*478).toFixed(2)+" "+(244-p.y/max*222).toFixed(2)).join(" ")+'"/>';});
    $("sensitivity-chart").innerHTML=svg(s,"Model sensitivity curves","All scenario inputs except the chosen axis remain fixed. Each line represents one trained model.");
  }
  function bars(target, items, signed=false) {
    target.replaceChildren(); const max=Math.max(...items.map(v=>Math.abs(v.value)),1e-8);
    items.forEach(item=>{
      const row=el("div",undefined,"bar-item"), track=el("div",undefined,"bar-track"), fill=el("div",undefined,"bar-fill");
      fill.style.width=Math.abs(item.value)/max*100+"%"; if(signed&&item.value<0)fill.style.background=COLORS[1];
      track.append(fill); row.append(el("span",item.name),track,el("span",item.formatted,"bar-number"));target.append(row);
    });
  }
  function updateScenario() {
    const x=scenario();$("age-value").textContent=x[0];$("bmi-value").textContent=x[2].toFixed(1);
    $("predictions").replaceChildren();
    report.models.forEach((m,i)=>{const box=el("div",undefined,"prediction");box.style.setProperty("--model-color",COLORS[i]);box.append(el("span",m.name),el("strong",money(predict(m,x))),el("small","Back-transformed estimate"));$("predictions").append(box);});
    sensitivity(x);
    const m=report.models.find(m=>m.id===$("effect-model").value), current=predict(m,x);
    bars($("effects"),report.features.map((f,i)=>{const modified=x.slice();modified[i]=report.reference[i];const diff=current-predict(m,modified);return {name:label(f),value:diff,formatted:(diff>=0?"+":"−")+money(Math.abs(diff))};}),true);
  }
  function updateDiagnostics() {
    const m=report.models.find(m=>m.id===$("diagnostic-model").value), d=m.density, max=d.edges.at(-1), n=d.counts.length, peak=Math.max(...d.counts.flat(),1);
    let s=axes("Actual charge · USD","Predicted · USD",max,max);
    d.counts.forEach((row,i)=>row.forEach((count,j)=>{if(!count)return;const x=62+i/n*478,y=244-(j+1)/n*222;s+='<rect tabindex="0" x="'+x+'" y="'+y+'" width="'+(478/n)+'" height="'+(222/n)+'" fill="#45dfbf" opacity="'+(.18+.82*count/peak)+'"><title>Actual '+money(d.edges[i])+'–'+money(d.edges[i+1])+', predicted '+money(d.edges[j])+'–'+money(d.edges[j+1])+': '+count+' records</title></rect>'; }));
    s+='<path d="M62 244L540 22" fill="none" stroke="#d7e7fa" stroke-dasharray="5 5" opacity=".65"/>';
    $("density-chart").innerHTML=svg(s,"Actual versus predicted density","Binned holdout predictions. The dashed diagonal marks exact agreement.");
    $("density-summary").textContent=report.metadata.test_rows+" holdout records. Mean absolute error: "+money(m.metrics.mae_usd)+".";
    const h=m.residuals, ymax=Math.max(...h.counts)*1.12, lo=h.edges[0], hi=h.edges.at(-1);
    s=axes("Residual · USD","Records",hi,ymax,lo);
    h.counts.forEach((c,i)=>{const x=62+i/h.counts.length*478,y=244-c/ymax*222;s+='<rect tabindex="0" x="'+(x+1)+'" y="'+y+'" width="'+(478/h.counts.length-2)+'" height="'+(244-y)+'" fill="'+(h.edges[i+1]<=0?COLORS[1]:COLORS[2])+'"><title>'+money(h.edges[i])+' to '+money(h.edges[i+1])+': '+c+' records</title></rect>';});
    if(lo<0&&hi>0){const zero=62+(0-lo)/(hi-lo)*478;s+='<path d="M'+zero+' 22V244" stroke="white" stroke-dasharray="3 4"/>';}
    $("residual-chart").innerHTML=svg(s,"Residual histogram","Counts of holdout errors grouped by predicted minus actual charges.");
    $("residual-summary").textContent="Residual range: "+money(lo)+" to "+money(hi)+". Counts sum to "+h.counts.reduce((a,b)=>a+b,0)+".";
    bars($("importance"),report.features.map((f,i)=>({name:label(f),value:m.importance[i],formatted:m.importance[i].toFixed(3)+" ± "+m.importance_std[i].toFixed(3)})),true);
    $("groups").replaceChildren();m.groups.forEach(g=>tableRow($("groups"),[g.label,g.n,money(g.mae)]));
  }
  function download(name, text, type) {
    const url=URL.createObjectURL(new Blob([text],{type})), a=el("a");a.href=url;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  function showCSV(result) {
    csvState=result;$("csv-result").hidden=false;$("csv-stats").replaceChildren();
    [[result.total,"Records"],[result.valid.length,"Valid"],[result.invalid,"Invalid"],[result.duplicates,"Duplicate model rows"]].forEach(v=>stat($("csv-stats"),...v));
    $("csv-issues").replaceChildren();result.issues.forEach(issue=>$("csv-issues").append(el("li",issue)));
    if(result.invalid>10)$("csv-issues").append(el("li","Showing the first 10 validation errors."));
    if(!result.invalid)$("csv-issues").append(el("li","All records passed validation."));
    $("csv-preview").replaceChildren();result.valid.slice(0,20).forEach(v=>tableRow($("csv-preview"),[v.row,v.x[0],v.x[2],...report.models.map(m=>money(predict(m,v.x)))]));
    $("export-csv").disabled=!result.valid.length;
    $("csv-status").textContent="Validation complete. "+result.valid.length+" valid records; no data left your browser.";
  }
  async function start() {
    try {
      const response=await fetch("assets/experiment.json");if(!response.ok)throw new Error("Unable to load model data.");
      report=await response.json();if(report.schema_version!==1||report.models.length!==3)throw new Error("Unsupported model export.");
      const m=report.metadata;
      stat($("stats"),m.rows.toLocaleString(),"Unique records");stat($("stats"),m.train_rows+" / "+m.test_rows,"Train / test");
      stat($("stats"),"3","Trained models");stat($("stats"),String(m.seed),"Reproducible seed");
      ["effect-model","diagnostic-model"].forEach(id=>report.models.forEach(model=>{const opt=el("option",model.name);opt.value=model.id;$(id).append(opt);}));$("effect-model").value="boost";$("diagnostic-model").value="boost";
      ["age","bmi"].forEach(f=>{const input=$(f);input.min=report.bounds[f][0];input.max=report.bounds[f][1];});
      function reset(){report.features.forEach((f,i)=>$(f).value=report.reference[i]);updateScenario();}
      $("reset").addEventListener("click",reset);$("scenario").addEventListener("submit",e=>e.preventDefault());$("scenario").addEventListener("input",updateScenario);
      $("sensitivity-feature").addEventListener("change",updateScenario);$("effect-model").addEventListener("change",updateScenario);$("diagnostic-model").addEventListener("change",updateDiagnostics);
      document.querySelectorAll("[data-panel]").forEach(button=>button.addEventListener("click",()=>{document.querySelectorAll(".panel").forEach(p=>p.hidden=p.id!==button.dataset.panel);document.querySelectorAll("[data-panel]").forEach(b=>{b.classList.toggle("active",b===button);b.setAttribute("aria-pressed",String(b===button));});}));
      report.models.forEach(model=>{const v=model.metrics;tableRow($("metrics"),[model.name,v.train_r2.toFixed(4),v.test_r2.toFixed(4),v.test_mse_log10.toFixed(4),money(v.mae_usd)]);});
      $("reference-note").textContent="Training reference: age "+report.reference[0]+", "+(report.reference[1]?"male":"female")+", BMI "+report.reference[2].toFixed(2)+", "+(report.reference[3]?"smoker":"non-smoker")+".";
      $("provenance").textContent=m.note+" Input SHA-256: "+m.input_sha256+". Python "+m.python_version+"; scikit-learn "+m.sklearn_version+"; random seed "+m.seed+"; "+Math.round(m.test_fraction*100)+"% holdout; "+m.duplicates_removed+" duplicate record(s) removed before splitting.";
      $("sample").addEventListener("click",()=>download("example-inputs.csv","age,sex,bmi,smoker\n35,female,28,no\n50,male,32,yes\n23,female,22,no\n","text/csv"));
      $("csv-file").addEventListener("change",async e=>{
        const file=e.target.files[0];if(!file)return;$("csv-result").hidden=true;csvState=null;
        try{if(file.size>5*1024*1024)throw new Error("Limit: 5 MB per CSV.");showCSV(validateCSV(await file.text(),report.bounds));}
        catch(error){$("csv-status").textContent=error.message;}
      });
      $("export-csv").addEventListener("click",()=>{
        if(!csvState)return;
        const rows=["record,age,sex,bmi,smoker,actual_charges,linear_usd,tree_usd,boost_usd"];
        csvState.valid.forEach(v=>rows.push([v.row,v.x[0],v.x[1]?"male":"female",v.x[2],v.x[3]?"yes":"no",v.charges??"",...report.models.map(m=>predict(m,v.x).toFixed(2))].join(",")));
        download("model-predictions.csv",rows.join("\n")+"\n","text/csv");
      });
      $("workspace").hidden=false;$("load-status").hidden=true;reset();updateDiagnostics();
    } catch(error) { $("load-status").textContent="Lab could not load. "+error.message+" Serve this repository over HTTP and try again."; }
  }
  start();
}());
