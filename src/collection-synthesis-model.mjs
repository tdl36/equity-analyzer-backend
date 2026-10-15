import {parseTimestamp} from './workspace-model.mjs';
function modeState(mode,now,analyst=false) {
 if(!mode||typeof mode!=='object')return null;
 const earnings=mode.earnings??mode.enabled;
 if(analyst ? typeof earnings!=='boolean'||(mode.takeaways!=null&&typeof mode.takeaways!=='boolean') : typeof mode.enabled!=='boolean')return null;
 const enabled=analyst?(earnings||mode.takeaways===true):mode.enabled;
 if(!enabled)return false;
 if(mode.expires_at){const expiry=parseTimestamp(mode.expires_at);if(!expiry)return null;return expiry.getTime()>now;}
 return true;
}
export function synthesisRows(data={},now=Date.now()) {
 const label=value=>value===null?'Status unavailable':value?'On':'Off';
 const analysts=data.analysts?.analysts;
 const states=Array.isArray(analysts)?analysts.map(a=>modeState(a.autoMode,now,true)):null;
 const on=states?.filter(s=>s===true).length||0;
 const unknown=states===null||states.includes(null);
 return [
  {label:'Event folder synthesis',status:label(modeState(data.catalysts,now)),href:'#view=catalysts',control:'Catalyst notes → Auto-Pilot'},
  {label:'New files → thesis proposals',status:label(modeState(data.intake,now)),href:'#view=desk&section=evidence',control:'Evidence & changes → New files'},
  {label:'Analyst earnings & takeaways',status:unknown?(on?`${on} analysts on; other settings unavailable`:'Status unavailable'):on?`${on} of ${states.length} analysts on`:'Off for all analysts',href:'#view=analysts',control:'Analyst team → select analyst → Auto Mode'},
 ];
}
