export const assignmentOutputs=[['summary_note','Summary note'],['stock_summary','Stock summary'],['thesis','Thesis proposal'],['visual','One-pager'],['meeting_prep','Meeting questions']];
export const fallbackAssignmentDefaults={outputs:['summary_note','stock_summary','thesis','visual'],horizon:'12–24 months'};
export function checkedAssignmentDefaults(value) {
 if(!Array.isArray(value?.outputs)||!value.outputs.length||value.outputs.length>assignmentOutputs.length||
    new Set(value.outputs).size!==value.outputs.length||value.outputs.some(id=>!assignmentOutputs.some(([key])=>key===id))||
    typeof value.horizon!=='string'||!value.horizon.trim()||value.horizon.length>100)throw Error('Saved assignment defaults are unavailable.');
 return {outputs:[...value.outputs],horizon:value.horizon};
}
export function outputNames(ids) {
 return (ids||[]).map(id=>assignmentOutputs.find(([key])=>key===id)?.[1]||'Unknown output').join(', ');
}
