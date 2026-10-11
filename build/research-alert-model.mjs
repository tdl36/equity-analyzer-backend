export function researchAlerts(data,dismissed={}) {
 return (data?.items||[]).filter(r=>['research_assignment','stock_analysis_runs','company_research_runs','investment_committee_runs'].includes(r.workflow)&&['attention','failed','needs_auth'].includes(r.status))
 .map(r=>({...r,alertKey:JSON.stringify([r.id,r.status,r.error||r.step||''])}))
 .sort((a,b)=>(a.workflow==='research_assignment'?0:1)-(b.workflow==='research_assignment'?0:1))
 .filter((r,i,all)=>all.findIndex(a=>a.ticker===r.ticker)===i).filter(r=>!dismissed[r.alertKey]);
}

export function incompleteResearchAlerts(data) {
 return !!data?.unavailable?.length || (data?.truncated||[]).some(label=>
  ['Research workflows','Stock analysis','Company research','Investment committee'].some(prefix=>label===prefix||label===prefix+' (attention)'));
}
