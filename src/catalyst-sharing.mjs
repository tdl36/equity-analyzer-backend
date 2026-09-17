export const SHARE_VIEWS = ['pm','comprehensive','qa'];
export function shareableViews(value,sections){
  return SHARE_VIEWS.filter(key=>sections[key] && (key==='qa'?value.qa?.status==='ready':!value.editorialVersion||!!value.editorial?.[key]));
}
export function emailContent(value,sections,scope,labels){
  const keys=shareableViews(value,sections).filter(key=>scope==='all'||key===scope);
  if(!keys.length)throw Error('This note is not ready to email.');
  return keys.map(key=>`<div style="font-family:Calibri,Carlito,Arial,sans-serif;font-size:11pt;line-height:1.55"><h1 style="font-size:11pt;font-weight:700">${labels[key]}</h1>${sections[key]}</div>`).join('<hr style="border:0;border-top:1px solid #ddd;margin:28px 0">');
}

// Remove labels from exported HTML, not merely hide them with CSS.
export function qaSpeakerVisibility(html,show=true){
  if(show)return html;
  return html.replace(/<p data-qa-speaker="asker">[\s\S]*?<\/p>/g,'').replace(/<strong data-qa-speaker="respondent">[\s\S]*?<\/strong>/g,'');
}
