/* Eco Exposure Map — generic multi-layer interactive map.
 * Everything site-specific comes from window.ECO (see reference/data-schema.md).
 * UI strings live in STR (English default); override any key via ECO.config.strings.
 */
(function(){
const D=window.ECO, C=D.config;
const STR=Object.assign({
  tabs:{sum:'Summary',pol:'Pollutants',air:'Air',inc:'Incidents',obj:'Sources',water:'Water',meth:'Method'},
  polLayer:'Pollutant layer:',
  polIntro:'Every major pollutant is listed. Status says what the data allow: a calibrated map, an emissions-based layer, station values only, historic data, regional background only, or no data at all.',
  status:{field:'calibrated map',stations:'stations only',emissions:'emissions only',below_threshold:'below reporting threshold',historic:'historic only',background:'background model only',none:'no data',not_assessed:'not assessed'},
  statusNote:{field:'≥3 stations, validated by nested leave-one-out',stations:'measured, but too few stations or no valid map',emissions:'reported releases, no monitoring',below_threshold:'operating facilities reported it earlier; now below the register threshold (not zero)',historic:'only facilities no longer in the register, or measurements that stopped',background:'only the ~10 km CAMS model',none:'queried: not monitored and not reported here',not_assessed:'no station network or register was queried for this country'},
  facStatus:{reporting:'reporting',not_reporting:'not in register since',unknown_after_register_end:'register ends'},emLayer:'emissions → map',threshold:'register threshold',check:'check: unusually high',
  polCols:['Pollutant','Status','Measured (nearest)','Reported emissions','Background (CAMS)'],
  showLayer:'map',emitTitle:'Reported emitters',emitAll:'all pollutants',emitNote:'Circle area ∝ reported release of the selected pollutant (kg/yr, latest reporting year).',
  closed:'closed',kgYr:'kg/yr',massUnits:['kt','t','kg','g','mg'],perYr:'/yr',stationsN:'{n} stations',since:'since',noLayer:'—',relLvl:'relative',
  hint:'Left click: values at a point · right click: close',
  heatTitle:'Heat layer',objTitle:'Overlays',baseTitle:'Basemap',none:'No layer',
  vec:'Vector (embedded)',osm:'OSM tiles',osmNote:'Tiles load only outside claude.ai (e.g. when the HTML file is opened locally).',
  collapse:'collapse',expand:'layers ▾',opacity:'Opacity',
  fromHome:'from {home}',approx:'approximate coordinates',impacts:'Impacts',source:'source',
  upper:'upper tier',lower:'lower tier',hazard:'Major-hazard',sev:'severity',
  point:'Point',nearest:'Nearest sources:',indexLbl:'Load index',noData:'no data',
  lvl:['low','moderate','high','very high'],
  sumIntro:'Comparison of {r} m circles around each point. Click any map point for local values.',
  nearHome:'Sources nearest to {home}',incNear:'{n} documented incidents within 1.5 km of {home}.',
  airIntro:'Monthly means from official monitoring stations. A month is shown if ≥50% of hours are valid.',
  camsTitle:'Regional model (CAMS, ~10 km cell)',camsIntro:'Copernicus CAMS ensemble for the cell containing the study area — compare as background, not street level.',
  windTitle:'Wind rose',windIntro:'ERA5 reanalysis, 10 m wind. Petals show where wind blows FROM.',
  pollenTitle:'Pollen (CAMS)',year:'Year',mean:'Mean',coverage:'Coverage',exceed:'Exceedances',
  seasons:{year:'Whole year',JJA:'Jun–Aug',DJF:'Dec–Feb',MAM:'Mar–May',SON:'Sep–Nov'},
  calm:'calm',compass:['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW'],
  months:['J','F','M','A','M','J','J','A','S','O','N','D'],
  incIntro:'{n} documented episodes from press, agencies, NGOs, studies and petitions. Marker size = severity 1–5.',
  sort:'Sort:',sortOpts:{date:'newest first',dist:'closest to home',sev:'most severe'},
  objIntro:'{n} sources. Distances from {home}. Dashed outline = approximate location; hollow = closed.',
  search:'Search…',bathTitle:'Bathing water quality',
  bathIntro:'Official classification (4-season assessment). Beaches sorted north → south.',
  bathCls:{Excellent:'Excellent',Good:'Good',Sufficient:'Sufficient',Poor:'Poor'},
  monthlyBtn:'Monthly chart →',stationNote:'Annual means, µg/m³',
  probeNote:'Red = above threshold (NO₂ 20, Lden 65, Ln 55).',
  no2Model:'NO₂ model, annual',ldenLbl:'Noise Lden',lnLbl:'Night noise Ln',odourLbl:'Odour',dustLbl:'Dust',riskLbl:'Industrial risk',
  tank:'Storage tank (OSM)',contentUnknown:'content not tagged',homeLabel:null
},C.strings||{});
const HOME=C.home, AREAS=C.areas||[];
const $=s=>document.querySelector(s);
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const esc=s=>String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const tpl=(s,o)=>s.replace(/\{(\w+)\}/g,(_,k)=>o[k]!=null?o[k]:'');
function hav(a,b,c,d){const R=6371000,t=Math.PI/180;const x=Math.sin((c-a)*t/2)**2+Math.cos(a*t)*Math.cos(c*t)*Math.sin((d-b)*t/2)**2;return 2*R*Math.asin(Math.sqrt(x));}
const DEC=C.decimal||'.';
const fmtN=(v,d=1)=>v==null||isNaN(v)?'—':Number(v).toFixed(d).replace('.',DEC);
const fmtD=m=>m<1000?Math.round(m/10)*10+' m':fmtN(m/1000,m<10000?1:0)+' km';

const CATS=D.cats; const ICATS=D.icats;
D.src.forEach(s=>{s.dist=hav(HOME.lat,HOME.lon,s.lat,s.lon);if(!CATS[s.cat])s.cat=Object.keys(CATS)[0];});
D.inc.forEach(i=>i.dist=hav(HOME.lat,HOME.lon,i.lat,i.lon));
const isLine=s=>s.line===true;

/* ---------- map & basemap ---------- */
L.Path.mergeOptions({bubblingMouseEvents:false});
const map=L.map('map',{preferCanvas:true,minZoom:10,maxZoom:18}).setView(C.center||[HOME.lat,HOME.lon],C.zoom||14);
map.attributionControl.setPrefix(false);map.attributionControl.addAttribution(C.attribution||'© OpenStreetMap contributors');
[['basePane',200],['heatPane',350],['zonePane',380],['featPane',420]].forEach(([n,z])=>map.createPane(n).style.zIndex=z);
const baseR=L.canvas({pane:'basePane',padding:.4}),featR=L.canvas({pane:'featPane',padding:.3});
function dec(a){const o=[];let la=0,lo=0;for(let i=0;i<a.length;i+=2){la+=a[i];lo+=a[i+1];o.push([la/1e5,lo/1e5]);}return o;}
const B=D.base;let baseLayers={};
const ZMIN={bld:15,minor:14,rail:13,metro:13};
function buildBase(){Object.values(baseLayers).forEach(l=>map.removeLayer(l));baseLayers={};
  const P=(arr,opt)=>L.layerGroup((arr||[]).map(r=>L.polygon(dec(r),Object.assign({renderer:baseR,interactive:false,stroke:false,fillOpacity:1},opt))));
  const Ln=(arr,opt)=>L.layerGroup((arr||[]).map(r=>L.polyline(dec(r),Object.assign({renderer:baseR,interactive:false},opt))));
  const R=B.road||{};
  baseLayers.land=P(B.land,{fillColor:css('--land')});baseLayers.green=P(B.green,{fillColor:css('--green')});baseLayers.park=P(B.park,{fillColor:css('--park')});
  baseLayers.ind=P((B.industrial||[]).concat(B.commercial||[]),{fillColor:css('--ind')});baseLayers.beach=P(B.beach,{fillColor:css('--beach')});
  baseLayers.water=P(B.water,{fillColor:css('--water')});baseLayers.river=Ln(B.river,{color:css('--water'),weight:2});
  baseLayers.pier=Ln(B.pier,{color:css('--bld-edge'),weight:3});
  baseLayers.bld=P(B.building,{fillColor:css('--bld'),stroke:true,color:css('--bld-edge'),weight:.5});
  baseLayers.minor=Ln(R.minor,{color:css('--road'),weight:2.2});
  baseLayers.tert=Ln((R.tertiary||[]).concat(R.secondary||[]),{color:css('--road'),weight:3.2});
  baseLayers.prim=Ln((R.primary||[]).concat(R.trunk||[]),{color:css('--road-major'),weight:4});
  baseLayers.mw=Ln(R.motorway,{color:css('--motorway'),weight:4.5});
  baseLayers.rail=Ln(B.rail,{color:css('--rail'),weight:1.4,dashArray:'4 3'});baseLayers.metro=Ln(B.metro,{color:css('--metro'),weight:1.6,opacity:.8});
  baseLayers.runway=Ln(B.runway,{color:css('--rail'),weight:8});updateBaseZoom();}
function updateBaseZoom(){const z=map.getZoom();Object.keys(baseLayers).forEach(k=>{const l=baseLayers[k],need=z>=(ZMIN[k]||0);if(need&&!map.hasLayer(l))l.addTo(map);else if(!need&&map.hasLayer(l))map.removeLayer(l);});}
map.on('zoomend',updateBaseZoom);let osmTiles=null;

/* ---------- grids / heat layers ---------- */
const G=D.grid;const gb=L.latLngBounds([G.lat1-G.ny*G.dlat,G.lon0],[G.lat1,G.lon0+G.nx*G.dlon]);const GV={};
function loadGrid(k){return new Promise(res=>{const im=new Image();im.onload=()=>{const c=document.createElement('canvas');c.width=G.nx;c.height=G.ny;const x=c.getContext('2d');x.drawImage(im,0,0);const d=x.getImageData(0,0,G.nx,G.ny).data;const a=new Uint8Array(G.nx*G.ny);for(let i=0;i<a.length;i++)a[i]=d[i*4];GV[k]=a;res();};im.onerror=res;im.src=D.grids[k].d;});}
function gval(k,lat,lon){const a=GV[k];if(!a)return null;const x=Math.floor((lon-G.lon0)/G.dlon),y=Math.floor((G.lat1-lat)/G.dlat);if(x<0||y<0||x>=G.nx||y>=G.ny)return null;const q=a[y*G.nx+x];if(!q)return null;const g=D.grids[k];return g.lo+(q-1)/254*(g.hi-g.lo);}
const hex=h=>{h=h.replace('#','');return[parseInt(h.slice(0,2),16),parseInt(h.slice(2,4),16),parseInt(h.slice(4,6),16)];};
const HEAT=D.heat; // {key:{n,unit,stops|bands,ticks,desc,sea:true|false}}
function colorFor(h,v){if(h.bands){let c=null;for(const b of h.bands)if(v>=b[0])c=b[1];return c?[...hex(c),.62]:null;}
  const s=h.stops;if(v<s[0][0])return null;for(let i=1;i<s.length;i++){if(v<=s[i][0]){const t=(v-s[i-1][0])/(s[i][0]-s[i-1][0]);const a=hex(s[i-1][1]),b=hex(s[i][1]);return[a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t,a[2]+(b[2]-a[2])*t,s[i-1][2]+(s[i][2]-s[i-1][2])*t];}}
  const l=s[s.length-1];return[...hex(l[1]),l[2]];}
const heatCache={};
function heatURL(k){if(heatCache[k])return heatCache[k];const h=HEAT[k],a=GV[k],land=GV.land,g=D.grids[k];const c=document.createElement('canvas');c.width=G.nx;c.height=G.ny;const x=c.getContext('2d');const id=x.createImageData(G.nx,G.ny);
  for(let i=0;i<a.length;i++){if(!a[i])continue;const sea=land&&land[i]<128;if(sea&&!h.sea)continue;const v=g.lo+(a[i]-1)/254*(g.hi-g.lo);const col=colorFor(h,v);if(!col)continue;id.data.set([col[0],col[1],col[2],Math.round(col[3]*255*(sea?.45:1))],i*4);}
  x.putImageData(id,0,0);return heatCache[k]=c.toDataURL();}
let heatLayer=null,heatKey=C.defaultHeat||Object.keys(HEAT)[0],heatOpacity=.8;
function setHeat(k){heatKey=k;if(heatLayer){map.removeLayer(heatLayer);heatLayer=null;}if(k!=='none'&&GV[k])heatLayer=L.imageOverlay(heatURL(k),gb,{pane:'heatPane',opacity:heatOpacity,interactive:false}).addTo(map);drawLegend();}
function drawLegend(){const el=$('#legend');if(heatKey==='none'||!HEAT[heatKey]){el.hidden=true;return;}el.hidden=false;const h=HEAT[heatKey];let body='';
  if(h.bands)body='<div class="bands">'+h.bands.map((b,i)=>`<div style="--c:${b[1]}">${i===h.bands.length-1?'≥'+b[0]:b[0]+'–'+(h.bands[i+1][0]-1)}</div>`).join('')+'</div>';
  else{const lo=h.stops[0][0],hi=h.stops[h.stops.length-1][0];body=`<div class="ramp" style="background:linear-gradient(90deg,${h.stops.map(s=>`${s[1]} ${((s[0]-lo)/(hi-lo)*100).toFixed(0)}%`).join(',')})"></div><div class="ticks">${(h.ticks||[]).map(t=>`<span>${String(t).replace('.',DEC)}</span>`).join('')}</div>`;}
  el.innerHTML=`<div style="display:flex;justify-content:space-between;gap:10px;align-items:baseline"><b>${esc(h.n)}</b><span class="mono tiny">${esc(h.unit||'')}</span></div>${body}<div class="tiny" style="margin-top:5px;max-width:330px">${esc(h.desc||'')}</div><div class="row" style="margin-top:6px"><label class="tiny" for="opacity">${STR.opacity}</label><input id="opacity" type="range" min="0.2" max="1" step="0.05" value="${heatOpacity}"></div>`;
  $('#opacity').oninput=e=>{heatOpacity=+e.target.value;heatLayer&&heatLayer.setOpacity(heatOpacity);};}

/* ---------- overlays ---------- */
const OV={};const sevC=s=>['#9AA3A6','#E0B040','#E88A2E','#D9532F','#B22A35','#6E1F4F'][s]||'#999';
const tierTxt=t=>t==='upper'?STR.upper:STR.lower;
function srcPopup(s){const q=s.quantitative;let qh='';
  const cmp=v=>typeof v==='object'&&v!==null?JSON.stringify(v).replace(/[{}"\[\]]/g,'').replace(/,/g,', ').replace(/:/g,': '):String(v);
  if(q&&typeof q==='object')qh='<div class="kv" style="margin-top:6px">'+Object.entries(q).slice(0,10).map(([k,v])=>{const t=cmp(v);return `<dt>${esc(k.replace(/_/g,' '))}</dt><dd class="mono" style="font-size:11px">${esc(t.length>200?t.slice(0,200)+'…':t)}</dd>`;}).join('')+'</div>';
  const tags=[`<span class="tag" style="color:${CATS[s.cat].c}">${esc(CATS[s.cat].n)}</span>`,`<span class="tag">${esc(s.status||'')}</span>`];
  if(s.hazard_tier)tags.push(`<span class="tag sev" style="background:${s.hazard_tier==='upper'?'#B22A35':'#E88A2E'}">${STR.hazard} · ${tierTxt(s.hazard_tier)}</span>`);
  return `<h5>${esc(s.name)}</h5>${tags.join('')}<p class="small">${esc(s.description||'')}</p><p class="tiny" style="margin-top:4px">${STR.impacts}: ${esc((s.impacts||[]).join(', '))}</p>${qh}<p class="tiny" style="margin-top:6px">${fmtD(s.dist)} ${tpl(STR.fromHome,{home:esc(HOME.name)})}${s.approx?' · '+STR.approx:''}</p><p class="tiny">${(s.sources||[]).slice(0,4).map((u,i)=>`<a href="${esc(u)}" target="_blank" rel="noopener">${STR.source} ${i+1}</a>`).join(' · ')}</p>`;}
function incPopup(i){return `<h5>${esc(i.t)}</h5><span class="tag" style="color:${ICATS[i.c].c}">${esc(ICATS[i.c].n)}</span><span class="tag sev" style="background:${sevC(i.s)}">${STR.sev} ${i.s}/5</span><span class="tag">${esc(i.d)}</span><p class="small">${esc(i.f)}</p><p class="tiny" style="margin-top:5px">${esc(i.loc)} · ${fmtD(i.dist)}</p><p class="tiny"><a href="${esc(i.u)}" target="_blank" rel="noopener">${esc(i.src)}</a></p>`;}
function stPopup(s){const ys=C.years||['2023','2024','2025'];const rows=Object.entries(s.annual||{}).map(([p,y])=>`<dt>${esc((PM[p]&&PM[p].label)||p)}</dt><dd class="mono">${ys.map(k=>y[k]&&y[k].mean!=null?fmtN(y[k].mean):'—').join(' · ')}</dd>`).join('');
  return `<h5>${esc(s.name)}</h5><span class="tag">${esc(s.code)}</span><span class="tag">${esc(s.type)}</span><p class="tiny">${STR.stationNote} · ${ys.join(' · ')}</p><dl class="kv" style="margin-top:4px">${rows}</dl>${s.flag?`<p class="tiny" style="margin-top:6px">⚑ ${esc(s.flag)}</p>`:''}<p style="margin-top:6px"><button class="chip" data-station="${esc(s.code)}">${STR.monthlyBtn}</button></p>`;}
const BC={Excellent:'#2F8F5B',Good:'#79B04A',Sufficient:'#E0A030',Poor:'#C4392F'};
function bathPopup(b){return `<h5>${esc(b.name)}</h5><span class="tag">${esc(b.id||'')}</span><div class="kv">${Object.keys(b.class).sort().map(y=>`<dt>${y}</dt><dd>${esc(STR.bathCls[b.class[y]]||b.class[y])}</dd>`).join('')}</div>${b.url?`<p class="tiny" style="margin-top:6px"><a href="${esc(b.url)}" target="_blank" rel="noopener">profile</a></p>`:''}`;}
function buildOverlays(){
  OV.polys=L.layerGroup((D.polys||[]).map(p=>L.polygon(p.ll,{renderer:featR,color:p.color||'#D1495B',weight:1.6,dashArray:p.dashed?'6 4':null,fillColor:p.color||'#D1495B',fillOpacity:p.dashed?.04:.12,interactive:!!p.name,bubblingMouseEvents:true}).bindTooltip(p.name||'',{sticky:true})));
  const tc={fuel:'#D1495B',oil:'#D1495B',gas:'#7B4FB0',lng:'#7B4FB0',cement:'#9A5B34',water:'#1F7A99'};
  OV.tanks=L.layerGroup((B.tanks||[]).map(t=>L.circle([t[0],t[1]],{renderer:featR,bubblingMouseEvents:true,radius:Math.max(t[2],3),color:tc[t[3]]||'#B0555F',weight:1,fillOpacity:.35,fillColor:tc[t[3]]||'#D1495B'}).bindTooltip(`${STR.tank} · ${t[3]||STR.contentUnknown} · Ø≈${Math.round(t[2]*2)} m`)));
  const HZ=C.hazardRings||{upper:[250,750],lower:[150,400]};
  OV.hazard=L.layerGroup(D.src.filter(s=>s.hazard_tier).flatMap(s=>{const r=HZ[s.hazard_tier]||HZ.lower,c=s.hazard_tier==='upper'?'#B22A35':'#E88A2E';return[L.circle([s.lat,s.lon],{pane:'zonePane',renderer:featR,radius:r[0],color:c,weight:1.4,fill:false,dashArray:'2 4',interactive:false}),L.circle([s.lat,s.lon],{pane:'zonePane',renderer:featR,radius:r[1],color:c,weight:1,opacity:.6,fill:false,dashArray:'1 6',interactive:false})];}));
  OV.roads=L.layerGroup((D.roads||[]).map(r=>L.polyline(r.ll,{renderer:featR,bubblingMouseEvents:true,color:'#56616A',opacity:.75,weight:r.aadt?1.5+r.aadt/30000:2}).bindTooltip(`${esc(r.name)}${r.aadt?' · AADT ≈ '+r.aadt.toLocaleString():''}`,{sticky:true})));
  OV.src=L.layerGroup(D.src.filter(s=>!isLine(s)).map(s=>{const c=CATS[s.cat].c,closed=/closed/i.test(s.status||'');const m=L.circleMarker([s.lat,s.lon],{renderer:featR,radius:s.hazard_tier==='upper'?9:s.hazard_tier?7.5:6.5,color:c,weight:closed?1.5:2.5,fillColor:closed?css('--panel'):c,fillOpacity:.85,dashArray:s.approx?'3 2':null}).bindPopup(()=>srcPopup(s),{maxWidth:340});s.marker=m;return m;}));
  OV.inc=L.layerGroup(D.inc.map(i=>{const m=L.circleMarker([i.lat,i.lon],{renderer:featR,radius:4+i.s*1.4,color:'#fff',weight:1.5,fillColor:ICATS[i.c].c,fillOpacity:.9}).bindPopup(incPopup(i),{maxWidth:330});i.marker=m;return m;}));
  if(L.heatLayer&&D.inc.length)OV.incheat=L.heatLayer(D.inc.map(i=>[i.lat,i.lon,i.s/5]),{radius:38,blur:30,maxZoom:15,minOpacity:.25,gradient:{.2:'#6B8FB3',.4:'#8A7AB8',.6:'#D26A8A',.8:'#E8563A',1:'#FFB84A'}});
  OV.st=L.layerGroup((D.aq||[]).map(s=>{const m=L.marker([s.lat,s.lon],{icon:L.divIcon({className:'',html:'<div style="width:14px;height:14px;background:var(--accent);border:2px solid var(--panel);transform:rotate(45deg);box-shadow:0 0 0 1px var(--accent)"></div>',iconSize:[14,14],iconAnchor:[7,7]})}).bindPopup(stPopup(s));return m;}));
  OV.bath=L.layerGroup((D.bath||[]).map(b=>{const ys=Object.keys(b.class).sort();return L.circleMarker([b.lat,b.lon],{renderer:featR,radius:7,color:'#fff',weight:2,fillColor:BC[b.class[ys[ys.length-1]]]||'#999',fillOpacity:1}).bindPopup(bathPopup(b));}));
  OV.power=L.layerGroup((D.power||[]).map(p=>{const v=String(p.v||''),hv=/400000|220000|345000|500000/.test(v),kv=v.split(';').map(x=>x/1000).join('/')+' kV';return p.t==='l'?L.polyline(p.ll,{renderer:featR,bubblingMouseEvents:true,color:hv?'#7B4FB0':'#A07CC5',weight:hv?2.2:1.4,dashArray:'6 3',opacity:.85}).bindTooltip(`${kv}${p.n?' · '+esc(p.n):''}`,{sticky:true}):L.polygon(p.ll,{renderer:featR,bubblingMouseEvents:true,color:'#7B4FB0',weight:1.2,fillOpacity:.25}).bindTooltip(`${esc(p.n||'')} · ${kv}`);}));
  OV.coast=L.layerGroup((D.coast||[]).map(c=>L.polygon(c.ll,{renderer:featR,stroke:false,fillColor:c.g>=5?'#B22A35':'#E88A2E',fillOpacity:.35,interactive:false})));
  OV.lines=L.layerGroup((D.lines||[]).map(l=>L.polyline(l.ll,{renderer:featR,bubblingMouseEvents:true,color:l.color||'#40408F',weight:2,dashArray:l.dash||'8 6'}).bindTooltip(esc(l.tip||l.name||''),{sticky:true})));
  OV.emit=L.layerGroup();drawEmit('all');
  OV.areas=L.layerGroup(AREAS.flatMap(a=>[L.circle([a.lat,a.lon],{renderer:featR,radius:C.radius_m||400,color:css('--ink'),weight:1.2,dashArray:'4 4',fill:false,interactive:false}),L.marker([a.lat+(C.radius_m||400)/93000,a.lon],{icon:L.divIcon({className:'',html:`<div class="lbl-area">${esc(a.name)}</div>`,iconSize:[160,14],iconAnchor:[80,7]}),interactive:false})]));
  OV.home=L.marker([HOME.lat,HOME.lon],{zIndexOffset:1000,icon:L.divIcon({className:'',html:`<div style="display:flex;flex-direction:column;align-items:center"><div class="lbl-home">${esc(STR.homeLabel||HOME.name)}</div><div style="width:2px;height:8px;background:var(--ink)"></div><div style="width:10px;height:10px;border-radius:50%;background:var(--ink);border:2px solid var(--panel)"></div></div>`,iconSize:[160,40],iconAnchor:[80,38]})}).bindPopup(()=>probeHTML(HOME.lat,HOME.lon));
}
document.addEventListener('click',e=>{const b=e.target.closest('[data-station]');if(b&&$('#stSel')){selectTab('air');$('#stSel').value=b.dataset.station;fillPol();drawAir();}});


const PM=Object.fromEntries((D.pollutants||[]).map(p=>[p.key,p]));
function fmtKg(v){const u=STR.massUnits;return v>=1e6?fmtN(v/1e6,1)+' '+u[0]:v>=1000?fmtN(v/1000,1)+' '+u[1]:v>=1?fmtN(v,v<10?1:0)+' '+u[2]:v>=1e-3?fmtN(v*1000,1)+' '+u[3]:fmtN(v*1e6,1)+' '+u[4];}
let emitKey='all';
function emitPopup(f){const rows=Object.entries(f.air).sort((a,b)=>b[1].kg-a[1].kg).map(([p,v])=>`<dt>${esc(p)}</dt><dd class="mono">${fmtKg(v.kg)}${STR.perYr} · ${v.year}</dd>`).join('');
  return `<h5>${esc(f.name)}</h5><span class="tag">PRTR ${esc(f.activity||'')}</span><span class="tag">${STR.since} ${f.latest_year}</span><div class="kv" style="margin-top:6px">${rows}</div><p class="tiny" style="margin-top:6px">${fmtD(hav(HOME.lat,HOME.lon,f.lat,f.lon))} ${tpl(STR.fromHome,{home:esc(HOME.name)})}</p>`;}
function drawEmit(key){emitKey=key;if(!OV.emit)return;OV.emit.clearLayers();const cov=Object.fromEntries((D.coverage||[]).map(c=>[c.key,c]));
  const val=f=>key==='all'?Math.max(1,...Object.values(f.air).map(v=>v.kg)):(PRTR_CODES[key]||[]).reduce((t,c)=>t+((f.air[c]||{}).kg||0),0);
  const vs=(D.emitters||[]).map(f=>[f,val(f)]).filter(x=>x[1]>0);const mx=Math.max(1e-12,...vs.map(x=>x[1]));
  vs.forEach(([f,v])=>{const r=key==='all'?7:4+18*Math.sqrt(v/mx);L.circleMarker([f.lat,f.lon],{renderer:featR,radius:r,color:'#35205E',weight:1.5,fillColor:'#8C62BD',fillOpacity:.45}).bindPopup(emitPopup(f),{maxWidth:330}).bindTooltip(key==='all'?esc(f.name):`${esc(f.name)} · ${fmtKg(v)}${STR.perYr}`).addTo(OV.emit);});}
const PRTR_CODES=D.prtrCodes||{};
function PRTR_MATCH(code,key){return (PRTR_CODES[key]||[]).includes(code);}

/* ---------- point probe ---------- */
function band(v){if(v==null)return STR.noData;v=Math.round(v);const lo=Math.floor((v-2)/5)*5;return v>=77?'≥ 75':v<=38?'< 40':`${lo}–${lo+4}`;}
const PROBE=C.probe||[['no2','no2Model','val',' µg/m³',20],['lden','ldenLbl','band',' dB',65],['ln','lnLbl','band',' dB',55],['odour','odourLbl','lvl'],['dust','dustLbl','lvl'],['risk','riskLbl','lvl',null,.5]];
function probeHTML(lat,lon){const lvl=v=>v==null?'—':STR.lvl[v<.15?0:v<.4?1:v<.7?2:3];
  const rows=PROBE.filter(p=>D.grids[p[0]]).map(([k,lab,kind,unit,thr])=>{const v=gval(k,lat,lon);const f=thr!=null&&v!=null&&v>=thr?' style="color:var(--bad);font-weight:600"':'';return `<span>${esc(STR[lab]||lab)}</span><span${f}>${kind==='band'?band(v)+(unit||''):kind==='lvl'?lvl(v):fmtN(v)+(unit||'')}</span>`;}).join('');
  const extra=Object.keys(D.grids).filter(k=>/^(c_|e_)/.test(k)).map(k=>{const v=gval(k,lat,lon);const h=HEAT[k]||{};return `<span>${esc((h.n||k).replace(/ — emissions-based potential| \(calibrated model\)/,''))}${k.startsWith('e_')?' <span class="tiny">('+STR.relLvl+')</span>':''}</span><span>${k.startsWith('e_')?(v==null?'—':STR.lvl[v<.1?0:v<1?1:v<3?2:3]+' ('+fmtN(v,v<1?2:1)+')'):fmtN(v,2)+' '+esc(h.unit||'')}</span>`;}).join('');
  const ik=C.probeIndex||Object.keys(HEAT)[0];const ix=gval(ik,lat,lon);
  const near=D.src.filter(s=>!isLine(s)).map(s=>({s,d:hav(lat,lon,s.lat,s.lon)})).sort((a,b)=>a.d-b.d).slice(0,6);
  return `<h5>${STR.point} ${lat.toFixed(4)}, ${lon.toFixed(4)}</h5><p class="tiny">${fmtD(hav(HOME.lat,HOME.lon,lat,lon))} ${tpl(STR.fromHome,{home:esc(HOME.name)})}</p><div class="probe-v">${rows}${extra}<span><b>${STR.indexLbl}</b></span><span><b>${fmtN(ix,0)}</b> / 100</span></div><p class="tiny" style="margin:6px 0 2px">${STR.nearest}</p>${near.map(n=>`<div class="small" style="display:flex;justify-content:space-between;gap:8px"><span><span class="pip" style="background:${CATS[n.s.cat].c}"></span> ${esc(n.s.name.slice(0,46))}</span><span class="mono tiny">${fmtD(n.d)}</span></div>`).join('')}<p class="tiny" style="margin-top:6px">${STR.probeNote}</p>`;}
map.on('click',e=>L.popup({maxWidth:340}).setLatLng(e.latlng).setContent(probeHTML(e.latlng.lat,e.latlng.lng)).openOn(map));
// right click (long press on touch) closes any open popup instead of showing the browser menu
map.getContainer().addEventListener('contextmenu',e=>{e.preventDefault();map.closePopup();},true);

/* ---------- layer box ---------- */
const LAYERS=C.overlays||[['src','Pollution sources',1],['emit','Reported emitters (PRTR)',1],['polys','Site outlines',1],['tanks','Storage tanks (OSM)',1],['hazard','Major-hazard consequence rings',1],['inc','Incidents & complaints',1],['incheat','Complaint heatmap',0],['st','Air-quality stations',1],['bath','Beaches (latest class)',1],['roads','Major roads (AADT)',0],['power','Power lines & substations',0],['coast','Coastal hazard',0],['lines','Flight paths / other lines',0],['areas','Compared areas',1],['home','Home',1]];
function buildLayerBox(){const has=k=>OV[k]&&OV[k].getLayers&&OV[k].getLayers().length;
  const isPol=k=>/^(c_|e_)/.test(k);const polKeys=Object.keys(HEAT).filter(isPol);
  const hk=Object.entries(HEAT).filter(([k])=>!isPol(k)).map(([k,h])=>`<label><input type="radio" name="heat" value="${k}" ${k===heatKey?'checked':''}> ${esc(h.n)}</label>`).join('')
    +(polKeys.length?`<label><input type="radio" name="heat" value="__pol" ${isPol(heatKey)?'checked':''}> ${STR.polLayer}</label><select id="polLayerSel" aria-label="${STR.polLayer}" style="width:100%;margin:2px 0 4px">${polKeys.map(k=>`<option value="${k}" ${k===heatKey?'selected':''}>${esc(HEAT[k].n)}</option>`).join('')}</select>`:'')
    +`<label><input type="radio" name="heat" value="none"> ${STR.none}</label>`;
  const ov=LAYERS.filter(([k])=>k==='home'||k==='incheat'?OV[k]:has(k)).map(([k,n,on])=>`<label><input type="checkbox" data-ov="${k}" ${on?'checked':''}> ${esc(n)}</label>`).join('');
  $('#layerbox').innerHTML=`<button class="collapse" id="lbToggle" aria-expanded="true">${STR.collapse}</button><div class="lb-body"><h4>${STR.heatTitle}</h4>${hk}<h4>${STR.objTitle}</h4>${ov}<h4>${STR.baseTitle}</h4><label><input type="radio" name="bm" value="vec" checked> ${STR.vec}</label><label><input type="radio" name="bm" value="osm"> ${STR.osm}</label><p class="tiny">${STR.osmNote}</p></div>`;
  LAYERS.forEach(([k,n,on])=>{if(on&&OV[k])OV[k].addTo(map);});
  $('#layerbox').addEventListener('change',e=>{const t=e.target;if(t.id==='polLayerSel'){setHeat(t.value);const r=document.querySelector('input[name="heat"][value="__pol"]');if(r)r.checked=true;return;}if(t.name==='heat'){if(t.value==='__pol'){const sel=$('#polLayerSel');setHeat(sel.value);}else setHeat(t.value);}else if(t.dataset.ov){const l=OV[t.dataset.ov];t.checked?l.addTo(map):map.removeLayer(l);}else if(t.name==='bm'){if(t.value==='osm'){osmTiles=osmTiles||L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,pane:'basePane',opacity:.9});osmTiles.addTo(map);}else if(osmTiles)map.removeLayer(osmTiles);}});
  $('#lbToggle').onclick=()=>{const b=$('#layerbox'),c=b.classList.toggle('closed');b.querySelector('.lb-body').hidden=c;$('#lbToggle').textContent=c?STR.expand:STR.collapse;$('#lbToggle').setAttribute('aria-expanded',!c);};
  if(window.innerWidth<820)$('#lbToggle').click();}

/* ---------- tabs ---------- */
const TABS=Object.entries(STR.tabs).filter(([k])=>k!=='air'||(D.aq&&D.aq.length)||D.cams).filter(([k])=>k!=='water'||(D.bath&&D.bath.length)||C.water_html);
function selectTab(k){document.querySelectorAll('nav.tabs button').forEach(b=>b.setAttribute('aria-selected',b.dataset.t===k));document.querySelectorAll('.tabbody > section').forEach(s=>s.hidden=s.id!=='t-'+k);try{localStorage.setItem('eco-tab',k)}catch(e){}}

/* summary: rows from C.summaryRows [{label,key,digits,warn,bad,fmt:'range'}], findings from C.findings */
function drawSummary(){const S=D.stats;const A=AREAS.map(a=>S[a.k]||{});
  const pipOf=(r,v)=>r.warn==null||v==null?null:(r.invert?(v<=r.bad?'--bad':v<=r.warn?'--warn':'--good'):(v>=r.bad?'--bad':v>=r.warn?'--warn':'--good'));
  const cell=(r,s)=>{const v=s[r.key];if(r.fmt==='range')return `${fmtN(v,r.digits??1)} <span class="tiny">${fmtN(s[r.key+'_lo'],0)}–${fmtN(s[r.key+'_hi'],0)}</span>`;return r.bold?`<b>${fmtN(v,r.digits??1)}</b>`:fmtN(v,r.digits??1);};
  const near=D.src.filter(s=>!isLine(s)).sort((a,b)=>a.dist-b.dist).slice(0,12);
  $('#t-sum').innerHTML=`<p class="muted small">${esc(C.summaryIntro||tpl(STR.sumIntro,{r:C.radius_m||400}))}</p>
  <table class="cmp"><thead><tr><th></th>${AREAS.map(a=>`<th>${esc(a.short||a.name)}</th>`).join('')}</tr></thead><tbody>${(C.summaryRows||[]).map(r=>`<tr><td>${esc(r.label)}</td>${A.map(s=>{const p=pipOf(r,s[r.key]);return `<td><span class="cell"><span class="mono">${cell(r,s)}</span><span class="pip" style="${p?`background:var(${p})`:''}"></span></span></td>`;}).join('')}</tr>`).join('')}</tbody></table>
  ${C.summaryNote?`<p class="tiny">${esc(C.summaryNote)}</p>`:''}
  ${(C.findings||[]).map(f=>`${f.h?`<h2>${esc(f.h)}</h2>`:''}${f.title?`<div class="finding"><div class="bar" style="background:var(--${f.level||'warn'})"></div><div><b>${esc(f.title)}</b> <span class="muted">${esc(f.text||'')}</span></div></div>`:''}${f.p?`<p class="small muted">${esc(f.p)}</p>`:''}`).join('')}
  <h2>${esc(tpl(STR.nearHome,{home:HOME.name}))}</h2><div class="list">${near.map(s=>`<div class="item" data-src="${D.src.indexOf(s)}"><span class="dot" style="color:${CATS[s.cat].c};background:${CATS[s.cat].c}"></span><span class="ttl">${esc(s.name)}</span><span class="dist mono">${fmtD(s.dist)}</span><span class="meta">${esc((s.impacts||[]).slice(0,3).join(' · '))}${s.hazard_tier?' · '+STR.hazard+' '+tierTxt(s.hazard_tier):''}</span></div>`).join('')}</div>
  <p class="tiny">${esc(tpl(STR.incNear,{n:D.inc.filter(i=>i.dist<1500).length,home:HOME.name}))}</p>`;}


/* pollutants coverage */
const STC={field:'var(--accent)',stations:'#5E8F9A',emissions:'#8C62BD',below_threshold:'#B79AD6',historic:'#9AA3A6',background:'#B9BFC2',none:'var(--bad)',not_assessed:'#D08B1C'};
function drawPol(){const el=$('#t-pol');if(!el)return;const cov=D.coverage||[];const G=D.groups||{};
  const refTxt=c=>(c.refs&&c.refs.length)?`<span class="tiny">${esc(c.refs.map(r=>r.l).join(' · '))}</span>`:'';
  const meas=c=>{if(!c.obs||!c.obs.length)return c.n_stations_ever?`<span class="tiny">${tpl(STR.stationsN,{n:c.n_stations_ever})} · ${c.period[0]||''}–${c.period[1]||''}</span>`:'—';const o=c.obs[0];const r0=(c.refs||[])[0];const bad=r0&&o.mean>r0.v;
    return `<span class="mono"${bad?' style="color:var(--bad);font-weight:600"':''}>${fmtN(o.mean,o.mean<1?2:1)} ${esc(o.unit||c.unit)}</span><br><span class="tiny">${esc(o.name.slice(0,28))} · ${fmtN(o.dist_km,1)} km · ${o.year}${c.n_stations_recent>1?' · '+tpl(STR.stationsN,{n:c.n_stations_recent}):''}</span>`;};
  const emi=c=>{if(!c.emitters||!c.emitters.length)return c.threshold_kg?`— <span class="tiny">(${STR.threshold} ${fmtKg(c.threshold_kg)}${STR.perYr})</span>`:'—';const e=c.emitters[0];const tot=c.emissions_kg_recent||0;
    const st=e.fac_status==='reporting'?'':` <span class="tiny">(${STR.facStatus[e.fac_status]||e.fac_status}${e.fac_status==='not_reporting'?' '+(e.fac_last_year+1):''})</span>`;
    return `<span class="mono">${fmtKg(tot||e.kg)}${STR.perYr}</span>${tot?'':' <span class="tiny">'+e.year+'</span>'}${e.suspicious?` <span class="tiny" style="color:var(--bad)">⚠ ${STR.check}</span>`:''}<br><span class="tiny">${esc(e.name.slice(0,30))}${e.dist_km!=null?' · '+fmtN(e.dist_km,1)+' km':''}${st}${c.emitters.length>1?' +'+(c.emitters.length-1):''}</span>`;};
  const groups=[...new Set(cov.map(c=>c.group))];
  el.innerHTML=`<p class="small muted">${esc(C.polIntro||STR.polIntro)}</p>
  <div class="chips">${Object.entries(STR.status).map(([k,n])=>`<span class="chip" style="cursor:default" title="${esc(STR.statusNote[k])}"><span class="pip" style="background:${STC[k]}"></span>${n} · ${cov.filter(c=>c.status===k).length}</span>`).join('')}</div>
  <div class="row"><label class="tiny" for="emitSel">${STR.emitTitle}:</label><select id="emitSel" aria-label="emitters"><option value="all">${STR.emitAll}</option>${cov.filter(c=>c.emitters&&c.emitters.length).map(c=>`<option value="${c.key}">${esc(c.label)}</option>`).join('')}</select></div><p class="tiny">${STR.emitNote}</p>
  ${groups.map(g=>`<h3>${esc(G[g]||g)}</h3><div class="list">${cov.filter(c=>c.group===g).map(c=>`<div class="polrow">
   <div class="polhead"><b>${esc(c.label)}</b><span class="chip" style="cursor:${c.layer?'pointer':'default'};border-color:${STC[c.status]}" ${c.layer?`data-layer="${c.layer}"`:''} title="${esc(STR.statusNote[c.status]+(c.note_fit?' — '+c.note_fit:''))}"><span class="pip" style="background:${STC[c.status]}"></span>${STR.status[c.status]}${c.layer&&c.layer===c.layer2?' · '+STR.emLayer:c.layer?' → '+STR.showLayer:''}</span></div>
   ${c.layer2&&c.layer!==c.layer2?`<span class="chip" style="align-self:flex-start;border-color:#8C62BD" data-layer="${c.layer2}"><span class="pip" style="background:#8C62BD"></span>${STR.emLayer}</span>`:''}
   ${refTxt(c)}
   <div class="polcols"><div><span class="tiny">${STR.polCols[2]}</span><br>${meas(c)}</div><div><span class="tiny">${STR.polCols[3]}</span><br>${emi(c)}</div><div><span class="tiny">${STR.polCols[4]}</span><br><span class="mono">${c.cams_mean!=null?fmtN(c.cams_mean,c.cams_mean<1?2:1)+' '+esc(c.unit):'—'}</span></div></div>
   ${c.note_fit?`<p class="tiny">${esc(c.note_fit)}</p>`:''}</div>`).join('')}</div>`).join('')}
  ${C.pol_html||''}`;
  $('#emitSel').value=emitKey;$('#emitSel').onchange=e=>{drawEmit(e.target.value);if(!map.hasLayer(OV.emit)){OV.emit.addTo(map);const cb=document.querySelector('[data-ov="emit"]');if(cb)cb.checked=true;}};}
document.addEventListener('click',e=>{const b=e.target.closest('[data-layer]');if(b){const k=b.dataset.layer;setHeat(k);const pol=/^(c_|e_)/.test(k);const r=document.querySelector(`input[name="heat"][value="${pol?'__pol':k}"]`);if(r)r.checked=true;if(pol&&$('#polLayerSel'))$('#polLayerSel').value=k;if(window.innerWidth<820)$('#mapwrap').scrollIntoView({behavior:'smooth'});}});

/* charts */
function lineChart(series,refs,opt){const W=380,H=opt.h||190,m={l:34,r:8,t:10,b:22},iw=W-m.l-m.r,ih=H-m.t-m.b;
  let ymax=Math.max(1,...series.flatMap(s=>s.v.filter(v=>v!=null)),...refs.map(r=>r.v))*1.12;ymax=Math.ceil(ymax/5)*5||10;
  const x=i=>m.l+iw*(i+.5)/12,y=v=>m.t+ih*(1-v/ymax);let g='';const step=ymax>60?20:ymax>30?10:ymax>12?5:2;
  for(let v=0;v<=ymax;v+=step)g+=`<line x1="${m.l}" x2="${W-m.r}" y1="${y(v)}" y2="${y(v)}" stroke="var(--rule)"/><text x="${m.l-5}" y="${y(v)+3}" text-anchor="end">${v}</text>`;
  STR.months.forEach((mm,i)=>g+=`<text x="${x(i)}" y="${H-6}" text-anchor="middle">${mm}</text>`);
  refs.forEach(r=>{if(r.v>ymax)return;g+=`<line x1="${m.l}" x2="${W-m.r}" y1="${y(r.v)}" y2="${y(r.v)}" stroke="${r.c}" stroke-width="1.2" stroke-dasharray="4 3"/><text x="${W-m.r-2}" y="${y(r.v)-3}" text-anchor="end" style="fill:${r.c}">${esc(r.l)}</text>`;});
  series.forEach(s=>{let d='',pen=false;s.v.forEach((v,i)=>{if(v==null){pen=false;return;}d+=(pen?'L':'M')+x(i).toFixed(1)+','+y(v).toFixed(1);pen=true;});g+=`<path d="${d}" fill="none" stroke="${s.c}" stroke-width="${s.w||2}" stroke-linejoin="round" ${s.dash?`stroke-dasharray="${s.dash}"`:''}/>`;s.v.forEach((v,i)=>{if(v!=null)g+=`<circle cx="${x(i)}" cy="${y(v)}" r="2.4" fill="${s.c}"><title>${esc(s.n)} · ${i+1}: ${fmtN(v)}</title></circle>`;});});
  return `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="${esc(opt.label||'')}">${g}</svg><div class="legend-line">${series.map(s=>`<span><i class="sw" style="background:${s.c}"></i>${esc(s.n)}</span>`).join('')}</div>`;}
const YC=['#9FB8BE','#5E8F9A','#0E5E6F','#D2502B','#7B4FB0'];
const REFS=D.refs||{};
function drawAirShell(){const el=$('#t-air');if(!el)return;const st=(D.aq||[]).slice().sort((a,b)=>hav(HOME.lat,HOME.lon,a.lat,a.lon)-hav(HOME.lat,HOME.lon,b.lat,b.lon));
  const camsKeys=Object.keys(D.cams||{}).filter(k=>!/pollen/.test(k));const pollen=Object.keys(D.cams||{}).filter(k=>/pollen/.test(k));
  el.innerHTML=`${st.length?`<p class="small muted">${esc(C.airIntro||STR.airIntro)}</p><div class="row"><select id="stSel" aria-label="Station">${st.map(s=>`<option value="${esc(s.code)}">${esc(s.name)} · ${fmtD(hav(HOME.lat,HOME.lon,s.lat,s.lon))}</option>`).join('')}</select><select id="polSel" aria-label="Pollutant"></select></div><div class="chart" id="airChart"></div><div id="airAnnual"></div><p class="tiny" id="airFlag"></p>`:''}
  ${camsKeys.length?`<h2>${STR.camsTitle}</h2><p class="small muted">${STR.camsIntro}</p><div class="row"><select id="camsSel" aria-label="CAMS variable">${camsKeys.map(k=>`<option value="${k}">${k.replace(/_/g,' ')}</option>`).join('')}</select></div><div class="chart" id="camsChart"></div>`:''}
  ${D.wind?`<h2>${STR.windTitle}</h2><p class="small muted">${STR.windIntro}</p><div class="row"><select id="windSel" aria-label="Season">${Object.entries(STR.seasons).map(([k,n])=>`<option value="${k}">${n}</option>`).join('')}</select></div><div id="windRose"></div>${C.windNote?`<p class="small muted">${esc(C.windNote)}</p>`:''}`:''}
  ${pollen.length?`<h2>${STR.pollenTitle}</h2><div class="chart" id="pollenChart"></div>`:''}`;
  if(st.length){$('#stSel').value=(C.defaultStation&&st.some(s=>s.code===C.defaultStation))?C.defaultStation:st[0].code;$('#stSel').onchange=()=>{fillPol();drawAir();};$('#polSel').onchange=drawAir;fillPol();drawAir();}
  if(camsKeys.length){$('#camsSel').onchange=drawCams;drawCams();}if(D.wind){$('#windSel').onchange=drawWind;drawWind();}
  if(pollen.length)$('#pollenChart').innerHTML=lineChart(pollen.map((k,i)=>({n:k.replace('_pollen',''),c:YC[i%YC.length],v:camsMonthly(k)})),[],{h:160});}
function fillPol(){const s=D.aq.find(x=>x.code===$('#stSel').value);const order=(D.pollutants||[]).map(p=>p.key);const ps=Object.keys(s.monthly||{}).sort((a,b)=>(order.indexOf(a)+99*(order.indexOf(a)<0))-(order.indexOf(b)+99*(order.indexOf(b)<0)));const cur=$('#polSel').value;$('#polSel').innerHTML=ps.map(p=>`<option value="${esc(p)}">${esc((PM[p]&&PM[p].label)||p)}${s.units&&s.units[p]?' · '+esc(s.units[p]):''}</option>`).join('');if(ps.includes(cur))$('#polSel').value=cur;}
function drawAir(){const s=D.aq.find(x=>x.code===$('#stSel').value),p=$('#polSel').value,mo=(s.monthly||{})[p]||{};const ys=Object.keys(mo).sort().slice(-5);
  $('#airChart').innerHTML=lineChart(ys.map((y,i)=>({n:y,c:YC[(i+5-ys.length)%YC.length],v:mo[y],w:i===ys.length-2?2.6:1.8,dash:i===ys.length-1&&C.lastYearPartial?'4 3':null})),REFS[p]||[],{label:p});
  const an=(s.annual||{})[p]||{};$('#airAnnual').innerHTML=`<table class="cmp"><thead><tr><th>${STR.year}</th><th>${STR.mean}</th><th>${STR.coverage}</th><th>${STR.exceed}</th></tr></thead><tbody>${Object.keys(an).sort().reverse().slice(0,8).map(y=>{const a=an[y];const ex=Object.entries(a).filter(([k])=>/days|hours/.test(k)).map(([k,v])=>`${k.replace(/_/g,' ')}: ${v}`).join('; ');return `<tr><td>${y}</td><td class="mono">${fmtN(a.mean)}</td><td class="mono">${a.coverage_pct??'—'}%</td><td class="tiny" style="text-align:right">${esc(ex)}</td></tr>`;}).join('')}</tbody></table>`;
  $('#airFlag').textContent=s.flag?'⚑ '+s.flag:'';}
function camsMonthly(k){const m=D.cams[k]||{},yrs=C.camsYears||null,out=Array.from({length:12},()=>[]);Object.entries(m).forEach(([ym,v])=>{const y=ym.slice(0,4);if(!yrs||yrs.includes(y))out[+ym.slice(5)-1].push(v);});return out.map(a=>a.length?a.reduce((x,y)=>x+y,0)/a.length:null);}
function drawCams(){const k=$('#camsSel').value,map2={pm2_5:'pm25',pm10:'pm10',nitrogen_dioxide:'no2',ozone:'o3',sulphur_dioxide:'so2',ammonia:'nh3',formaldehyde:'hcho'};$('#camsChart').innerHTML=lineChart([{n:'CAMS',c:'#0E5E6F',v:camsMonthly(k),w:2.6}],REFS[map2[k]]||[],{label:k});}
function drawWind(){const r=D.wind[$('#windSel').value];const W=300,Cc=150,R=118,max=Math.max(...r.bins.map(b=>b.reduce((a,c)=>a+c,0))),top=Math.ceil(max/5)*5||5,sc=R/top;const cols=['#BFD8DC','#6FA8B3','#2E7A89','#0E4D5A'];let g='';
  for(let k=5;k<=top;k+=5)g+=`<circle cx="${Cc}" cy="${Cc}" r="${k*sc}" fill="none" stroke="var(--rule)"/><text x="${Cc+3}" y="${Cc-k*sc-2}">${k}%</text>`;
  r.bins.forEach((b,i)=>{let acc=0;const a0=(i*22.5-9)*Math.PI/180,a1=(i*22.5+9)*Math.PI/180;b.forEach((v,j)=>{const r0=acc*sc,r1=(acc+v)*sc;acc+=v;if(v<=0)return;const p=(a,rr)=>`${(Cc+rr*Math.sin(a)).toFixed(1)},${(Cc-rr*Math.cos(a)).toFixed(1)}`;g+=`<path d="M${p(a0,r0)}L${p(a0,r1)}A${r1},${r1} 0 0 1 ${p(a1,r1)}L${p(a1,r0)}A${r0},${r0} 0 0 0 ${p(a0,r0)}Z" fill="${cols[j]}"><title>${STR.compass[i]}: ${fmtN(b.reduce((a,c)=>a+c,0))}%</title></path>`;});
    if(i%2===0)g+=`<text x="${Cc+(R+16)*Math.sin(i*22.5*Math.PI/180)}" y="${Cc-(R+16)*Math.cos(i*22.5*Math.PI/180)+3}" text-anchor="middle">${STR.compass[i]}</text>`;});
  $('#windRose').innerHTML=`<svg viewBox="0 0 ${W} ${W}" width="100%" style="max-width:320px" role="img" aria-label="${STR.windTitle}">${g}</svg><div class="legend-line">${['<10','10–20','20–30','≥30 km/h'].map((l,j)=>`<span><i class="sw" style="background:${cols[j]};height:8px"></i>${l}</span>`).join('')}<span>${STR.calm} ${fmtN(r.calm)}%</span></div>`;}

/* incidents */
let incOn=new Set(Object.keys(ICATS)),incSort='date';
function drawInc(){const list=D.inc.filter(i=>incOn.has(i.c)).sort((a,b)=>incSort==='date'?b.y-a.y||String(b.d).localeCompare(String(a.d)):incSort==='dist'?a.dist-b.dist:b.s-a.s);
  const ys=D.inc.map(i=>i.y).filter(Boolean),y0=Math.min(...ys,2020)-1,y1=Math.max(...ys,2020)+1,W=380,H=90,cats=Object.keys(ICATS),x=y=>14+(W-28)*(y-y0)/(y1-y0);
  let tl='';const stp=(y1-y0)>20?5:2;for(let y=Math.ceil(y0/stp)*stp;y<=y1;y+=stp)tl+=`<line x1="${x(y)}" x2="${x(y)}" y1="2" y2="${H-14}" stroke="var(--rule)"/><text x="${x(y)}" y="${H-3}" text-anchor="middle">${y}</text>`;
  D.inc.forEach(i=>{const j=cats.indexOf(i.c);tl+=`<circle cx="${x(i.y)+((i.t.length%5)-2)*1.5}" cy="${10+j*(60/cats.length)}" r="${1.5+i.s*.8}" fill="${ICATS[i.c].c}" opacity="${incOn.has(i.c)?.85:.15}"><title>${esc(i.d+' · '+i.t)}</title></circle>`;});
  $('#t-inc').innerHTML=`<p class="small muted">${esc(tpl(STR.incIntro,{n:D.inc.length}))}</p><div class="chips">${cats.map(c=>`<button class="chip" data-ic="${c}" aria-pressed="${incOn.has(c)}"><span class="pip" style="background:${ICATS[c].c}"></span>${esc(ICATS[c].n)} · ${D.inc.filter(i=>i.c===c).length}</button>`).join('')}</div>
  <svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="timeline">${tl}</svg><div class="row"><span class="tiny">${STR.sort}</span><select id="incSort" aria-label="sort">${Object.entries(STR.sortOpts).map(([k,n])=>`<option value="${k}">${n}</option>`).join('')}</select></div>
  <div class="list">${list.map(i=>`<div class="item" data-inc="${D.inc.indexOf(i)}"><span class="dot" style="color:${ICATS[i.c].c};background:${ICATS[i.c].c}"></span><span class="ttl">${esc(i.t)}</span><span class="dist mono">${esc(String(i.d).slice(0,10))}</span><span class="meta">${esc(i.f.slice(0,150))}${i.f.length>150?'…':''}<br><span class="tiny">${esc(i.loc)} · ${fmtD(i.dist)} · <a href="${esc(i.u)}" target="_blank" rel="noopener">${esc(i.src)}</a></span></span></div>`).join('')}</div>`;
  $('#incSort').value=incSort;$('#incSort').onchange=e=>{incSort=e.target.value;drawInc();};}
/* sources */
let objQ='';
function drawObj(){const q=objQ.toLowerCase(),items=D.src.filter(s=>!q||(s.name+' '+s.category+' '+(s.description||'')).toLowerCase().includes(q));
  const groups=Object.keys(CATS).map(c=>[c,items.filter(s=>s.cat===c).sort((a,b)=>a.dist-b.dist)]).filter(g=>g[1].length);
  $('#t-obj').innerHTML=`<p class="small muted">${esc(tpl(STR.objIntro,{n:D.src.length,home:HOME.name}))}</p><input type="search" id="objQ" placeholder="${STR.search}" value="${esc(objQ)}" aria-label="search">
  ${groups.map(([c,arr])=>`<h3 style="color:${CATS[c].c}">${esc(CATS[c].n)} · ${arr.length}</h3><div class="list">${arr.map(s=>`<div class="item" data-src="${D.src.indexOf(s)}"><span class="dot" style="color:${CATS[c].c};background:${/closed/i.test(s.status||'')?'transparent':CATS[c].c}"></span><span class="ttl">${esc(s.name)}</span><span class="dist mono">${fmtD(s.dist)}</span><span class="meta">${esc(s.status||'')}${s.hazard_tier?' · '+STR.hazard+' '+tierTxt(s.hazard_tier):''} · ${esc((s.impacts||[]).slice(0,3).join(', '))}</span></div>`).join('')}</div>`).join('')}`;
  $('#objQ').oninput=e=>{objQ=e.target.value;const pos=e.target.selectionStart;drawObj();const n=$('#objQ');n.focus();n.setSelectionRange(pos,pos);};}
/* water */
function drawWater(){const el=$('#t-water');if(!el)return;const bs=(D.bath||[]).slice().sort((a,b)=>b.lat-a.lat);const ys=[...new Set(bs.flatMap(b=>Object.keys(b.class)))].sort().slice(-7);const ab={Excellent:'E',Good:'G',Sufficient:'S',Poor:'P'};
  el.innerHTML=`${bs.length?`<h2>${STR.bathTitle}</h2><p class="small muted">${STR.bathIntro}</p><div class="bathgrid" style="grid-template-columns:minmax(120px,1fr) repeat(${ys.length},26px)"><span></span>${ys.map(y=>`<span class="h">${y.slice(2)}</span>`).join('')}${bs.map(b=>`<span class="small">${esc(b.name)}</span>${ys.map(y=>{const c=b.class[y];return `<span class="bq" style="background:${BC[c]||'var(--chip)'}" title="${y}: ${esc(c||'—')}">${ab[c]||''}</span>`;}).join('')}`).join('')}</div>`:''}${C.water_html||''}`;}
function drawMeth(){const c=D.calib||[];$('#t-meth').innerHTML=`${C.method_html||''}${c.length?`<table class="cmp"><thead><tr><th>Station</th><th>Observed</th><th>Model</th><th>Leave-one-out</th></tr></thead><tbody>${c.map((r,i)=>`<tr><td>${esc(r.name)}</td><td class="mono">${fmtN(r.obs)}</td><td class="mono">${fmtN(r.mod)}</td><td class="mono">${D.loo&&D.loo[i]?fmtN(D.loo[i].pred):'—'}</td></tr>`).join('')}</tbody></table>`:''}${C.sources_html||''}`;}

/* wiring */
document.addEventListener('click',e=>{const it=e.target.closest('.item');if(it&&!e.target.closest('a')){
  if(it.dataset.src!=null){const s=D.src[+it.dataset.src];map.flyTo([s.lat,s.lon],Math.max(map.getZoom(),15),{duration:.6});if(s.marker){if(!map.hasLayer(OV.src))OV.src.addTo(map);setTimeout(()=>s.marker.openPopup(),650);}}
  if(it.dataset.inc!=null){const i=D.inc[+it.dataset.inc];if(!map.hasLayer(OV.inc)){OV.inc.addTo(map);const cb=document.querySelector('[data-ov="inc"]');if(cb)cb.checked=true;}map.flyTo([i.lat,i.lon],Math.max(map.getZoom(),15),{duration:.6});setTimeout(()=>i.marker.openPopup(),650);}
  if(window.innerWidth<820)$('#mapwrap').scrollIntoView({behavior:'smooth'});}
  const ch=e.target.closest('[data-ic]');if(ch){const k=ch.dataset.ic;incOn.has(k)?incOn.delete(k):incOn.add(k);drawInc();}});
async function init(){
  document.title=C.title;$('#ttl').textContent=C.title;$('#sub').textContent=C.subtitle||'';$('.hint').textContent=STR.hint;document.documentElement.lang=C.lang||'en';
  $('nav.tabs').innerHTML=TABS.map(([k,n])=>`<button role="tab" data-t="${k}" aria-selected="false">${esc(n)}</button>`).join('');
  $('.tabbody').innerHTML=TABS.map(([k])=>`<section id="t-${k}" hidden></section>`).join('');
  $('nav.tabs').onclick=e=>{const b=e.target.closest('button');if(b)selectTab(b.dataset.t);};
  buildBase();await Promise.all(Object.keys(D.grids).map(loadGrid));buildOverlays();buildLayerBox();setHeat(heatKey);
  drawSummary();drawPol();drawAirShell();drawInc();drawObj();drawWater();drawMeth();
  let t='sum';try{t=localStorage.getItem('eco-tab')||'sum'}catch(e){}const hs=(location.hash||'').slice(1);if(TABS.some(x=>x[0]===hs))t=hs;selectTab(TABS.some(x=>x[0]===t)?t:'sum');
  window.addEventListener('hashchange',()=>{const h=location.hash.slice(1);if(TABS.some(x=>x[0]===h))selectTab(h);});
  const re=()=>buildBase();const mq=window.matchMedia('(prefers-color-scheme: dark)');mq.addEventListener&&mq.addEventListener('change',re);
  new MutationObserver(re).observe(document.documentElement,{attributes:true,attributeFilter:['data-theme']});}
init();
})();
