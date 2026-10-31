/* The Hangar: dependency-free scene, project explorer and roaming ground crew. */
(function(){
"use strict";
const $=id=>document.getElementById(id);
const clamp=(v,min,max)=>Math.max(min,Math.min(max,v));
function flightPoint(start,end,t,height){return {x:start.x+(end.x-start.x)*t,y:start.y+(end.y-start.y)*t-Math.sin(Math.PI*t)*height};}
if(typeof module!=="undefined"&&module.exports)module.exports={clamp,flightPoint};
if(typeof document==="undefined")return;
const reduced=matchMedia("(prefers-reduced-motion: reduce)");
let roaming=!reduced.matches,following=false,night=false,report=null,flightTimer=null;
let width=innerWidth,height=innerHeight,robotSize=width<=760?68:82,robotHeight=width<=760?75:90;
let position={x:Math.max(20,width-150),y:height-robotHeight-70},journey=null,nextWalk=performance.now()+1700,lastFrame=0,lastFollow=0;
let pointer={x:width*.7,y:height*.7,active:false};
const robot=$("pip"),bubble=$("pip-bubble"),ctx=$("atmosphere").getContext("2d");
let stars=[];
function say(message){$("announcement").textContent=message;}
function setStatus(text){$("crew-status").textContent=text;}
function updateMotion(){
 $("motion-toggle").setAttribute("aria-pressed",String(roaming));
 $("motion-toggle").textContent=roaming?"PIP: ROAMING":"PIP: RESTING";
 document.body.classList.toggle("motion-paused",!roaming);
 if(!roaming){journey=null;following=false;robot.classList.remove("walking");$("flyby").classList.remove("flying");}
 $("follow-toggle").setAttribute("aria-pressed",String(following));
 $("follow-toggle").disabled=!roaming;
 setStatus(roaming?"Pip is exploring the hangar":"Pip is taking a break");
}
function setNight(value){
 night=value;document.body.classList.toggle("night",night);
 $("night-switch").setAttribute("aria-pressed",String(night));
 $("night-label").textContent=night?"HANGAR LIGHTS":"NIGHT FLIGHT";
 $("stage-condition").textContent=night?"NIGHT OPERATIONS / CLEAR SKIES":"HANGAR LIGHTS / ON";
 try{localStorage.setItem("dk-hangar-night",String(night));}catch(_){}
 say(night?"Night flight mode enabled.":"Hangar lights enabled.");
}
$("night-switch").addEventListener("click",()=>setNight(!night));
try{setNight(localStorage.getItem("dk-hangar-night")==="true");}catch(_){setNight(false);}
$("motion-toggle").addEventListener("click",()=>{roaming=!roaming;nextWalk=performance.now()+500;updateMotion();});
$("follow-toggle").addEventListener("click",()=>{following=!following;journey=null;$("follow-toggle").setAttribute("aria-pressed",String(following));setStatus(following?"Pip is following your cursor":"Pip is exploring the hangar");});
reduced.addEventListener("change",e=>{if(e.matches){roaming=false;updateMotion();}});
function size(){
 width=innerWidth;height=innerHeight;robotSize=width<=760?68:82;robotHeight=width<=760?75:90;
 position.x=clamp(position.x,8,Math.max(8,width-robotSize-8));position.y=clamp(position.y,75,Math.max(75,height-robotHeight-62));
 const dpr=Math.min(devicePixelRatio||1,2),canvas=$("atmosphere");canvas.width=width*dpr;canvas.height=height*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);
 stars=Array.from({length:75},(_,i)=>({x:((i*137.508)%1000)/1000*width,y:((i*263.731)%1000)/1000*height,r:i%7===0?1.5:.7,phase:i*.73}));
 journey=null;placeRobot();
}
function placeRobot(){
 robot.style.transform="translate3d("+position.x.toFixed(2)+"px,"+position.y.toFixed(2)+"px,0)";
 if(!bubble.hidden){
  const bw=Math.min(280,width-24),bh=bubble.offsetHeight;
  bubble.style.left=clamp(position.x-85,12,Math.max(12,width-bw-12))+"px";
  bubble.style.top=clamp(position.y-bh-10,85,Math.max(85,height-bh-65))+"px";
 }
}
function openBubble(message){
 journey=null;robot.classList.remove("walking");bubble.hidden=false;$("pip-message").textContent=message;placeRobot();
}
function closeBubble(){bubble.hidden=true;nextWalk=performance.now()+1200;}
$("pip-button").addEventListener("click",()=>{if(!bubble.hidden){closeBubble();return;}openBubble("Hi, I’m Pip, your ground crew. I explore the page, hop onto card edges, and help you board the projects. Want to try a real prediction?");});
$("meet-pip").addEventListener("click",()=>{position={x:clamp(width*.55,20,width-robotSize-20),y:clamp(250,100,height-robotHeight-80)};openBubble("Reporting for duty! Click either aircraft for a project cockpit. The night-flight switch changes the atmosphere. You can pause me using the controls at the bottom.");$("pip-button").focus({preventScroll:true});});
$("bubble-close").addEventListener("click",closeBubble);
window.addEventListener("resize",size);
window.addEventListener("scroll",()=>{journey=null;nextWalk=performance.now()+500;},{passive:true});
window.addEventListener("pointermove",e=>{
 pointer={x:e.clientX,y:e.clientY,active:true};
 $("cursor-light").style.transform="translate("+(e.clientX-250)+"px,"+(e.clientY-250)+"px)";
 robot.classList.toggle("curious",Math.hypot(e.clientX-position.x-robotSize/2,e.clientY-position.y-robotHeight/2)<150);
},{passive:true});
document.addEventListener("pointerleave",()=>pointer.active=false);
function destinations(){
 const points=[{x:clamp(30+Math.random()*(width-robotSize-60),8,width-robotSize-8),y:height-robotHeight-66,caption:"Pip is patrolling the hangar floor"}];
 document.querySelectorAll("[data-perch]").forEach(element=>{
  const r=element.getBoundingClientRect(),y=r.top-robotHeight+4;
  if(y>95&&y<height-robotHeight-70&&r.width>robotSize+30){
   points.push({x:clamp(r.left+20+Math.random()*(r.width-robotSize-40),8,width-robotSize-8),y,caption:"Pip is inspecting a card from above"});
  }
 });
 return points;
}
function beginJourney(end,now,follow){
 const distance=Math.hypot(end.x-position.x,end.y-position.y);
 journey={start:{...position},end,startTime:now,duration:clamp(distance/(follow ? 0.18 : 0.09),850,6500),arc:Math.abs(end.y-position.y)>30?Math.min(90,distance*.2):12};
 if(Math.abs(end.x-position.x)>2)robot.style.setProperty("--face",end.x>=position.x?1:-1);
 robot.classList.add("walking");if(end.caption)setStatus(end.caption);
}
function frame(now){
 requestAnimationFrame(frame);
 if(document.hidden)return;
 if(now-lastFrame<30)return;lastFrame=now;
 ctx.clearRect(0,0,width,height);
 if(night){
  stars.forEach(s=>{ctx.globalAlpha=roaming&&!reduced.matches?.2+.45*(.5+.5*Math.sin(now*.0005+s.phase)):.5;ctx.fillStyle="#b9d3ed";ctx.beginPath();ctx.arc(s.x,s.y,s.r,0,Math.PI*2);ctx.fill();});ctx.globalAlpha=1;
  if(roaming&&!reduced.matches){const t=(now%14000)/14000;ctx.strokeStyle="#8dd4ef33";ctx.beginPath();ctx.moveTo(width*(1.1-t*1.3),height*.18+t*70);ctx.lineTo(width*(1.1-t*1.3)+35,height*.18+t*70-12);ctx.stroke();}
 }
 if(roaming&&bubble.hidden&&!$("cockpit").open){
  if(following&&pointer.active&&now-lastFollow>350){
   lastFollow=now;const end={x:clamp(pointer.x-robotSize/2,8,width-robotSize-8),y:clamp(pointer.y-robotHeight-25,90,height-robotHeight-62)};
   if(Math.hypot(end.x-position.x,end.y-position.y)>55)beginJourney(end,now,true);
  }
  if(!journey&&!following&&now>nextWalk){const options=destinations();beginJourney(options[Math.floor(Math.random()*options.length)],now,false);}
  if(journey){const t=clamp((now-journey.startTime)/journey.duration,0,1),e=t*t*(3-2*t);position=flightPoint(journey.start,journey.end,e,journey.arc);placeRobot();
   if(t>=1){journey=null;nextWalk=now+1600+Math.random()*2200;robot.classList.remove("walking");}
  }
 }
}
$("launch-flight").addEventListener("click",()=>{
 if(flightTimer)return;
 setNight(true);closeBubble();const launch=$("launch-flight"),status=$("flight-status");
 launch.disabled=true;status.hidden=false;status.textContent=reduced.matches||!roaming?"DEMO FLIGHT READY / ANIMATION PAUSED":"DEMO FLIGHT / TAXI → TAKEOFF → CLEAR SKIES";
 $("flyby").classList.remove("flying");
 if(!reduced.matches&&roaming){void $("flyby").offsetWidth;$("flyby").classList.add("flying");}
 flightTimer=setTimeout(()=>{$("flyby").classList.remove("flying");status.textContent="FLIGHT COMPLETE / YOUR NEXT DESTINATION: THE ML LAB";launch.disabled=false;
  flightTimer=setTimeout(()=>{status.hidden=true;flightTimer=null;},2400);
 },reduced.matches||!roaming?1500:6000);
});
document.addEventListener("keydown",e=>{
 if(e.key==="Escape")closeBubble();
 if(e.key.toLowerCase()==="n"&&!e.ctrlKey&&!e.metaKey&&!e.altKey&&!$("cockpit").open&&!/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName))setNight(!night);
});
function clock(){const d=new Date();$("stage-clock").textContent=d.toISOString().slice(11,19)+" UTC";}
setInterval(clock,1000);clock();
function node(tag,text,cls){const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n;}
function metric(parent,label,value,unit){const box=node("div",undefined,"instrument");box.append(node("span",label),node("strong",value),node("small",unit));parent.append(box);}
function link(parent,text,url){const a=node("a",text);a.href=url;parent.append(a);}
function modelChart(parent,models){
 const chart=node("div",undefined,"cockpit-chart");chart.append(node("h3","MODEL PERFORMANCE / TEST R² · LOG₁₀ TARGET"));
 models.forEach(m=>{const row=node("div",undefined,"model-bar"),track=node("div",undefined,"track"),fill=node("div",undefined,"fill");
 fill.style.width=clamp(m.metrics.test_r2,0,1)*100+"%";track.append(fill);row.append(node("span",m.name),track,node("strong",m.metrics.test_r2.toFixed(4)));chart.append(row);});parent.append(chart);
}
function openProject(project){
 closeBubble();journey=null;robot.classList.remove("walking");
 const dialog=$("cockpit"),body=$("cockpit-body"),links=$("cockpit-links");body.replaceChildren();links.replaceChildren();
 if(project==="healthcare"){
  $("cockpit-code").textContent="AIRCRAFT 01 / HEALTHCARE REGRESSION";
  $("cockpit-title").textContent="The prediction engine.";
  $("cockpit-subtitle").textContent="Comparing regression models for medical insurance charges. MSc Data Analytics dissertation, University of Sheffield.";
  if(report){
   const instruments=node("div",undefined,"instruments");instruments.id="live-instruments";body.append(instruments);
   const choices=node("label",undefined,"model-select");choices.append(node("span","Inspect model"));const select=node("select");select.id="cockpit-model";select.setAttribute("aria-label","Select a model to inspect");
   report.models.forEach(m=>{const o=node("option",m.name);o.value=m.id;select.append(o);});select.value="boost";choices.append(select);body.append(choices);
   function gauges(){const m=report.models.find(x=>x.id===select.value);instruments.replaceChildren();metric(instruments,"TEST R²",m.metrics.test_r2.toFixed(4),"log₁₀ target scale");metric(instruments,"DOLLAR MAE","$"+Math.round(m.metrics.mae_usd).toLocaleString("en-US"),"back-transformed predictions");metric(instruments,"HOLDOUT",String(report.metadata.test_rows),"records · seed "+report.metadata.seed);}
   select.addEventListener("change",gauges);gauges();modelChart(body,report.models);
   body.append(node("p","Recomputed experiment on the public insurance dataset. These values come from the verified model export in the ML Lab. A single split does not establish generalisation; dollar estimates are back-transformed log predictions.","cockpit-note"));
  }else{
   body.append(node("p","The live experiment export is not available right now. The project dashboard and interactive lab are still available below.","cockpit-note"));
  }
  link(links,"TRY THE INTERACTIVE ML LAB ↗","../lab/");link(links,"ORIGINAL PROJECT DASHBOARD ↗","../index.html");link(links,"SOURCE CODE ↗","https://github.com/dimmy98/Developing-Evaluating-ML-for-HealthCare");
 }else{
  $("cockpit-code").textContent="AIRCRAFT 02 / NEURAL SIGNAL CLASSIFICATION";
  $("cockpit-title").textContent="Finding signal in noise.";
  $("cockpit-subtitle").textContent="A University of Sheffield team project for the HMS Kaggle competition: EEG representations, EfficientNet and two-stage training.";
  const instruments=node("div",undefined,"instruments");metric(instruments,"TARGET CLASSES","6","harmful brain activity categories");metric(instruments,"REPORTED CV","1.2527","KL loss · historical result");metric(instruments,"TRAINING","2","stages explored");body.append(instruments);
  const chart=node("div",undefined,"cockpit-chart");chart.append(node("h3","SIGNAL DISPLAY / ILLUSTRATIVE ANIMATION"));
  const svg=document.createElementNS("http://www.w3.org/2000/svg","svg");svg.setAttribute("viewBox","0 0 700 100");svg.setAttribute("class","signal-chart");svg.setAttribute("role","img");svg.setAttribute("aria-label","Decorative signal illustration; not a patient EEG");
  const path=document.createElementNS("http://www.w3.org/2000/svg","path");let d="M0 50";for(let x=0;x<700;x+=2){const y=50+Math.sin(x*.11)*6+Math.sin(x*.037)*10+(x%160>65&&x%160<85?Math.sin((x%160-65)/20*Math.PI*3)*30:0);d+="L"+x+" "+y;}path.setAttribute("d",d);svg.append(path);chart.append(svg);chart.append(node("p","The waveform is decorative. It does not show patient data or live inference.","cockpit-note"));body.append(chart);
  body.append(node("p","The team repository contains training notebooks, local inference scripts and experiment records. The CV score is a historical value from the existing portfolio, not a model rerun in this hangar.","cockpit-note"));
  link(links,"EXPLORE THE TEAM REPOSITORY ↗","https://github.com/supinfl/UoS_HBAC/tree/baseline");link(links,"TWO-STAGE TRAINING NOTEBOOK ↗","https://github.com/supinfl/UoS_HBAC/blob/baseline/2stageTrainning.ipynb");link(links,"ORIGINAL PORTFOLIO ↗","../index.html");
 }
 if(!dialog.open)dialog.showModal();dialog.scrollTop=0;setStatus("Pip is standing by at the flight deck");
}
document.querySelectorAll("[data-project]").forEach(b=>b.addEventListener("click",()=>openProject(b.dataset.project)));
document.querySelectorAll(".aircraft").forEach(b=>{b.addEventListener("pointerenter",()=>{robot.classList.add("curious");setStatus("Pip says: click to board this aircraft");});b.addEventListener("pointerleave",()=>{robot.classList.remove("curious");if(!following)setStatus("Pip is exploring the hangar");});});
$("close-cockpit").addEventListener("click",()=>$("cockpit").close());
$("cockpit").addEventListener("click",e=>{if(e.target===$("cockpit")){const r=$("cockpit").getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)$("cockpit").close();}});
$("cockpit").addEventListener("close",()=>{nextWalk=performance.now()+700;setStatus(roaming?"Pip is exploring the hangar":"Pip is taking a break");});
fetch("../lab/assets/experiment.json").then(r=>{if(!r.ok)throw Error("Export unavailable");return r.json();}).then(data=>{
 if(data.schema_version!==1||!Array.isArray(data.models)||data.models.length!==3)throw Error("Invalid model export");
 if(!data.models.every(m=>m.metrics&&Number.isFinite(m.metrics.test_r2)&&Number.isFinite(m.metrics.mae_usd)))throw Error("Invalid metrics");
 report=data;
}).catch(()=>{report=null;});
size();updateMotion();requestAnimationFrame(frame);
}());
