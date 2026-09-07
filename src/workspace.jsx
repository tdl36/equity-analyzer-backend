import * as React from 'react';
import { GROUPS, viewLabel, viewGroup, companyIndex, parseTimestamp } from './workspace-model.mjs';
const { useState, useEffect, useRef } = React;

function Icon({name, ...props}) {
  const paths = { today: 'M3 4h18v16H3z M3 9h18 M8 2v4 M16 2v4 M7 13h3 M14 13h3 M7 17h3', companies: 'M4 21V5h10v16 M14 11h6v10 M8 9h2 M8 13h2 M8 17h2 M2 21h20', library: 'M3 4h5v16H3z M10 4h5v16h-5z M17 5l4-1 3 15-4 1z', create: 'M12 3v18 M3 12h18', automations: 'M13 2L4 14h7l-1 8 10-13h-7z', search: 'M10 3a7 7 0 1 0 0 14 7 7 0 0 0 0-14 M15 15l6 6', menu: 'M4 6h16 M4 12h16 M4 18h16', close:'M6 6l12 12 M18 6L6 18', arrow: 'M5 12h14 M14 7l5 5-5 5', settings:'M4 7h16 M4 17h16 M9 4v6 M15 14v6', chat:'M3 4h18v13H8l-5 4z' };
  return <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...props}><path d={paths[name] || paths.today}/></svg>;
}
export function WorkspaceShell({active, onNavigate, themeControl, local, health, ticker, onCompany}) {
  const [menu, setMenu] = useState(false);
  const [search, setSearch] = useState('');
  const dialog = useRef(null);
  const trigger = useRef(null);
  const group = viewGroup(active);
  const go = id => { setMenu(false); setSearch(''); onNavigate(id); };
  useEffect(() => {
    if (!menu) return;
    const before = document.activeElement;
    dialog.current?.querySelector('input')?.focus();
    const key = e => {
      if (e.key === 'Escape') { setMenu(false); return; }
      if (e.key !== 'Tab') return;
      const elements = [...dialog.current.querySelectorAll('button,input,a')].filter(x => !x.disabled && x.getClientRects().length);
      const first=elements[0], last=elements[elements.length-1];
      if (e.shiftKey && document.activeElement===first) { e.preventDefault(); last?.focus(); }
      if (!e.shiftKey && document.activeElement===last) { e.preventDefault(); first?.focus(); }
    };
    document.addEventListener('keydown',key);
    return () => { document.removeEventListener('keydown',key); before?.focus(); };
  },[menu]);
  useEffect(() => {
    const handler = e => { if ((e.metaKey || e.ctrlKey) && e.key==='k') { e.preventDefault(); setMenu(x=>!x); } };
    document.addEventListener('keydown',handler); return () => document.removeEventListener('keydown',handler);
  },[]);
  const unavailable = health && (health.status==='stale' || health.agentSeenEver===false);
  return <>
    <a className="workspace-skip" href="#workspace-content" onClick={e=>{e.preventDefault();document.getElementById('workspace-content')?.focus();}}>Skip to content</a>
    <aside className="workspace-sidebar" aria-label="Workspace navigation">
      <button className="workspace-brand" onClick={()=>go('today')} aria-label="Charlie home"><span className="workspace-monogram">C</span><span>Charlie<small>RESEARCH WORKSPACE</small></span></button>
      <button ref={trigger} className="workspace-search" onClick={()=>setMenu(true)}><Icon name="search"/><span>Find a workspace</span><kbd>⌘ K</kbd></button>
      <nav aria-label="Main navigation">{GROUPS.map(g=><div key={g.id} className="workspace-nav-group">
        <button className={'workspace-nav-item '+(group?.id===g.id?'is-active':'')} onClick={()=>go(g.id)} aria-current={active===g.id?'page':undefined}><Icon name={g.id}/>{g.label}<span className="nav-indicator"/></button>
        {group?.id===g.id && <div className="workspace-subnav">{g.items.map(([id,label])=><button key={id} onClick={()=>go(id)} aria-current={active===id?'page':undefined}>{label}</button>)}</div>}
      </div>)}</nav>
      <div className="workspace-sidebar-foot"><button onClick={()=>go('chat')}><Icon name="chat"/>Ask Charlie</button><button onClick={()=>go('settings')} aria-current={active==='settings'?'page':undefined}><Icon name="settings"/>Settings</button><div className="workspace-theme">{themeControl}</div><small>Evidence. Conviction. Perspective.</small></div>
    </aside>
    <header className="workspace-header"><div className="workspace-breadcrumb"><span>{group?.label || 'Workspace'}</span><span>/</span><strong>{active===group?.id ? group.description : viewLabel(active)}</strong></div>
      <div className="workspace-header-actions">{ticker && <button className="workspace-ticker" onClick={()=>onCompany(ticker)}>{ticker}</button>}<span className="workspace-local">{local ? 'TEST DATA' : 'LIVE DATA'}</span>
      <details className="workspace-health"><summary><span className={'workspace-status-dot '+(unavailable?'is-warning':'')}/><span>{local?'Local session':health ? unavailable?'Agent offline':'Agent connected':'Checking agent'}</span></summary><div role="status"><strong>{local?'Development environment':unavailable?'Local agent unavailable':'Research services'}</strong><p>{local?'You are using the local backend. The Mac agent reports to production, so its local heartbeat may be stale.':unavailable?'Tasks that need files on your Mac may wait until the agent reconnects. You can continue reading saved research.':'Your Mac agent is connected.'}</p><button onClick={()=>go('settings')}>View settings <Icon name="arrow"/></button></div></details>
      <button className="workspace-menu-trigger" onClick={()=>setMenu(true)} aria-label="Open workspace menu"><Icon name="menu"/></button></div>
    </header>
    <nav className="workspace-mobile-nav" aria-label="Mobile navigation">{GROUPS.slice(0,4).map(g=><button key={g.id} onClick={()=>go(g.id)} aria-current={group?.id===g.id?'page':undefined}><Icon name={g.id}/><span>{g.label}</span></button>)}<button onClick={()=>setMenu(true)} aria-label="All workspaces"><Icon name="menu"/><span>More</span></button></nav>
    {menu && <div className="workspace-dialog-scrim" onClick={()=>setMenu(false)}><section className="workspace-command" role="dialog" aria-modal="true" aria-labelledby="workspace-menu-title" ref={dialog} onClick={e=>e.stopPropagation()}><div className="workspace-command-head"><h2 id="workspace-menu-title">Your workspaces</h2><button onClick={()=>setMenu(false)} aria-label="Close workspace menu"><Icon name="close"/></button></div><label className="workspace-command-search"><Icon name="search"/><input placeholder="Find a tool or workflow…" aria-label="Search workspaces" value={search} onChange={e=>setSearch(e.target.value)}/></label><div className="workspace-command-results">{GROUPS.map(g=>{
      const items=[[g.id,g.label],...g.items].filter(i=>i[1].toLowerCase().includes(search.toLowerCase()));
      return items.length ? <div key={g.id}><h3>{g.label}</h3>{items.map(([id,label])=><button key={id} onClick={()=>go(id)}><span>{label}</span>{active===id?<small>Current</small>:<Icon name="arrow"/>}</button>)}</div>:null;
    })}{'settings'.includes(search.toLowerCase())&&<button onClick={()=>go('settings')}>Settings</button>}{!GROUPS.some(g=>[[g.id,g.label],...g.items].some(i=>i[1].toLowerCase().includes(search.toLowerCase())))&&!'settings'.includes(search.toLowerCase())&&<p>No matching workspace. Try “review” or “summary”.</p>}</div></section></div>}
  </>;
}
const dateLabel = v => parseTimestamp(v)?.toLocaleDateString('en-US',{month:'short',day:'numeric',timeZone:'America/New_York'}) || 'Date unavailable';
function PageHeading({eyebrow,title,children,action}) { return <div className="workspace-page-heading"><div><p className="workspace-eyebrow">{eyebrow}</p><h1>{title}</h1><p className="workspace-lead">{children}</p></div>{action}</div>; }
function Empty({children}) { return <p className="workspace-empty">{children}</p>; }
export function TodayWorkspace({analyses,overviews,summaries,alerts,meetings,onNavigate,onSummary,onCompany}) {
  const companies=companyIndex(analyses,overviews);
  const recent=[...summaries].sort((a,b)=>(parseTimestamp(b.createdAt)?.getTime()||0)-(parseTimestamp(a.createdAt)?.getTime()||0)).slice(0,5);
  const pending=alerts.filter(a=>!['dismissed','done','archived'].includes(a.status));
  const upcoming=(meetings||[]).filter(m=>m.meeting_date && m.meeting_date>=new Date().toLocaleDateString('en-CA')).sort((a,b)=>a.meeting_date.localeCompare(b.meeting_date)).slice(0,3);
  return <div className="workspace-page"><PageHeading eyebrow={new Date().toLocaleDateString('en-US',{weekday:'long',month:'long',day:'numeric'})} title="A clearer view of what matters." action={<button className="workspace-primary" onClick={()=>onNavigate('create')}>Create research <Icon name="arrow"/></button>}>Revisit your companies, follow the evidence, and move your research forward.</PageHeading>
    <div className="workspace-metrics">{[[companies.length,'Companies in your library','companies'],[summaries.length,'Saved source summaries','library'],[pending.length,'Alerts to review','alerts']].map(([n,label,id])=><button key={id} onClick={()=>onNavigate(id)}><strong>{n}</strong><span>{label}</span><Icon name="arrow"/></button>)}</div>
    <div className="workspace-home-grid"><section className="workspace-panel"><div className="workspace-section-heading"><h2>Continue your research</h2><button onClick={()=>onNavigate('library')}>Open library ↗</button></div>{recent.length?recent.map(s=><button className="workspace-document-row" key={s.id} onClick={()=>onSummary(s)}><span className="workspace-document-icon"><Icon name="library"/></span><span><strong>{s.title || 'Untitled document'}</strong><small>{s.topic || 'General'} · {s.docType || s.sourceType || 'Research'}</small></span><time>{dateLabel(s.createdAt)}</time><Icon name="arrow"/></button>):<Empty>Your saved research will appear here. Start by adding a document to the Library.</Empty>}</section>
    <div className="workspace-home-side"><section className="workspace-panel"><p className="workspace-eyebrow">NEXT STEP</p><h2>What are you working on?</h2>{[['companies','Revisit an investment view'],['meetingprep','Prepare for a management meeting'],['explain','Understand a difficult document']].map(([id,label])=><button className="workspace-link-row" key={id} onClick={()=>onNavigate(id)}>{label}<Icon name="arrow"/></button>)}</section><section className="workspace-panel"><h2>Upcoming meetings</h2>{upcoming.length?upcoming.map(m=><button className="workspace-link-row" key={m.id} onClick={()=>onNavigate('meetingprep')}>{m.ticker} <time>{dateLabel(m.meeting_date)}</time></button>):<Empty>No upcoming meetings in this workspace. Your saved meeting preparation is available in Create.</Empty>}</section></div></div>
    <section className="workspace-panel"><div className="workspace-section-heading"><h2>Company quick access</h2><button onClick={()=>onNavigate('companies')}>All companies ↗</button></div><div className="workspace-company-chips">{companies.slice(0,12).map(c=><button key={c.ticker} onClick={()=>onCompany(c.ticker)}>{c.ticker}<small>{c.company!==c.ticker?c.company:c.hasThesis?'Thesis saved':'Overview saved'}</small></button>)}</div>{!companies.length&&<Empty>Build your coverage by researching your first company.</Empty>}</section>
  </div>;
}
export function CompaniesWorkspace({analyses,overviews,ticker,onCompany,onNavigate}) {
  const [query,setQuery]=useState('');
  const companies=companyIndex(analyses,overviews), selected=companies.find(c=>c.ticker===ticker);
  const visible=companies.filter(c=>(c.ticker+' '+c.company).toLowerCase().includes(query.toLowerCase()));
  return <div className="workspace-page"><PageHeading eyebrow="YOUR COVERAGE" title="Companies" action={<button className="workspace-primary" onClick={()=>onNavigate('create')}>Research a company <Icon name="arrow"/></button>}>One place to return to the evidence and investment view behind each name.</PageHeading>
    {ticker&&<section className="workspace-company-focus"><div><p className="workspace-eyebrow">CURRENT COMPANY</p><h2>{ticker} <span>{selected?.company!==ticker?selected?.company:''}</span></h2><p>{selected ? [selected.hasOverview&&'Overview available',selected.hasThesis&&'Thesis available'].filter(Boolean).join(' · ') : 'Start research or add evidence for this company.'}</p></div><div className="workspace-company-actions">{[['overview','Overview'],['portfolio','Thesis'],['summary','Evidence'],['review','Review'],['onepager','One-pager']].map(([id,label])=><button key={id} onClick={()=>onCompany(ticker,id)}>{label}<Icon name="arrow"/></button>)}</div></section>}
    <div className="workspace-list-toolbar"><label className="workspace-filter"><Icon name="search"/><input aria-label="Search companies" placeholder="Search ticker or company…" value={query} onChange={e=>setQuery(e.target.value)}/></label><span>{visible.length} companies</span></div>
    <div className="workspace-company-grid">{visible.map(c=><button className={'workspace-company-card '+(ticker===c.ticker?'is-selected':'')} key={c.ticker} onClick={()=>onCompany(c.ticker)}><span className="workspace-company-card-top"><strong>{c.ticker}</strong><Icon name="arrow"/></span><span>{c.company}</span><small>{c.hasThesis?'Thesis + research':'Company overview'}</small></button>)}</div>{!visible.length&&<Empty>No matching companies. Try another name, or create new research.</Empty>}
  </div>;
}
export function LibraryWorkspace({summaries,onSummary,onNavigate}) {
  const [query,setQuery]=useState('');
  const [filter,setFilter]=useState('all');
  const matches=[...summaries].filter(s=>(`${s.title} ${s.topic} ${s.docType} ${s.sourceType}`).toLowerCase().includes(query.toLowerCase())&&(filter==='all'||s.sourceType===filter)).sort((a,b)=>(parseTimestamp(b.createdAt)?.getTime()||0)-(parseTimestamp(a.createdAt)?.getTime()||0));
  return <div className="workspace-page"><PageHeading eyebrow="THE EVIDENCE BEHIND THE VIEW" title="Research library" action={<button className="workspace-primary" onClick={()=>onNavigate('summary')}>Add or manage documents <Icon name="arrow"/></button>}>Search your saved summaries by company, title, or source type.</PageHeading><div className="workspace-list-toolbar"><label className="workspace-filter"><Icon name="search"/><input aria-label="Search library" placeholder="Search documents, companies, topics…" value={query} onChange={e=>setQuery(e.target.value)}/></label><select aria-label="Filter by source" value={filter} onChange={e=>setFilter(e.target.value)}><option value="all">All sources</option>{[...new Set(summaries.map(s=>s.sourceType).filter(Boolean))].map(v=><option key={v} value={v}>{v}</option>)}</select><span>{matches.length} documents</span></div><section className="workspace-panel">{matches.slice(0,100).map(s=><button className="workspace-document-row" key={s.id} onClick={()=>onSummary(s)}><span className="workspace-document-icon"><Icon name="library"/></span><span><strong>{s.title || 'Untitled document'}</strong><small>{s.topic || 'General'} · {s.docType || s.sourceType || 'Document'}</small></span><time>{dateLabel(s.createdAt)}</time><Icon name="arrow"/></button>)}{!matches.length&&<Empty>No documents match this search. Try a company ticker or clear the source filter.</Empty>}{matches.length>100&&<p className="workspace-empty">Showing the 100 most recent matches. Narrow your search to find older documents.</p>}</section><button className="workspace-link-row" onClick={()=>onNavigate('research')}>Browse research documents and folders <Icon name="arrow"/></button></div>;
}
const CREATIONS=[['deepdive','Company research brief','A fresh research pass with a one-page brief, two-page report, and investment memo.','Ticker · Web research'],['review','Investment review','Revisit the thesis, scenarios, and what changed using your stored evidence.','Ticker · Saved sources'],['meetingprep','Meeting preparation','Turn documents into focused questions for management.','Company · Source documents'],['onepager','Investment one-pager','A visual reference for the stock debate and key signposts.','Ticker · Existing + web research'],['slides','Presentation','Build and refine a slide narrative from your research.','Documents · Editable slides'],['formats','Thesis export','Present a saved thesis as a scorecard, IC memo, or another decision format.','Saved thesis · Template'],['explain','Document explanation','Understand dense language, assumptions, and financial terminology.','Text · PDF · Screenshots'],['studio','Creative studio','Explore infographics, mind maps, and learning materials.','Sources · Output format']];
export function CreateWorkspace({ticker,onTicker,onNavigate}) { return <div className="workspace-page"><PageHeading eyebrow="FROM EVIDENCE TO OUTPUT" title="What would you like to create?">Choose the outcome first. You can refine sources, depth, and model in the workspace.</PageHeading><label className="workspace-create-context">Company context <input aria-label="Company context ticker" placeholder="Ticker (optional)" value={ticker} onChange={e=>onTicker(e.target.value.toUpperCase())}/><span>Carried into company research tools</span></label><div className="workspace-create-grid">{CREATIONS.map(([id,title,description,meta],i)=><button key={id} onClick={()=>onNavigate(id)} className="workspace-create-card"><span className="workspace-create-number">0{i+1}</span><h2>{title}</h2><p>{description}</p><div><small>{meta}</small><Icon name="arrow"/></div></button>)}</div></div>; }
export function AutomationsWorkspace({onNavigate,local}) { return <div className="workspace-page"><PageHeading eyebrow="KEEP YOUR RESEARCH CURRENT" title="Automations">Follow signals, review proposed work, and manage research that runs in the background.</PageHeading>{local&&<p className="workspace-notice">Local development session. Mac-agent jobs are connected to production; review configuration before starting a run.</p>}<div className="workspace-create-grid">{[['desk','Research desk','Coordinate company teams, review aging theses, and track research runs.'],['analysts','Analyst team','Review pending investigations, coverage, and earnings activities.'],['pipeline','Research pipeline','Generate notes and refresh investment theses across your universe.'],['feed','Podcast feed','Filter new episodes and material mentions of companies you cover.'],['agents','Research agents','Configure multi-perspective analysis and catalyst investigations.']].map(([id,title,description])=><button className="workspace-create-card" key={id} onClick={()=>onNavigate(id)}><Icon name="automations"/><h2>{title}</h2><p>{description}</p><div><small>Open workspace</small><Icon name="arrow"/></div></button>)}</div></div>; }
// Server documents contain global CSS. A script-free frame prevents them from
// changing the host application while preserving their authored tables/styles.
export function ResearchDocument({html,title='Investment review'}) {
  const frame=useRef(null), observer=useRef(null);
  const [height,setHeight]=useState(800);
  useEffect(()=>()=>observer.current?.disconnect(),[]);
  const loaded=()=>{ observer.current?.disconnect(); const doc=frame.current?.contentDocument; if(!doc)return; const measure=()=>setHeight(Math.ceil(doc.documentElement.getBoundingClientRect().height)+24); measure(); observer.current=new ResizeObserver(measure); observer.current.observe(doc.documentElement); };
  const source=`<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"><style>html{background:#fff;color:#1e293b;}body{margin:0;padding:24px;box-sizing:border-box;color:#1e293b;}html,body{height:auto;min-height:0;}img{max-width:100%;}table{max-width:100%;}*{box-sizing:border-box;}@media(max-width:600px){body{padding:16px;}table{font-size:11px!important;}}</style></head><body>${html || '<p>No document is available for this review.</p>'}</body></html>`;
  return <iframe className="workspace-document-frame" ref={frame} title={title} sandbox="allow-same-origin" srcDoc={source} onLoad={loaded} style={{height}}/>;
}
