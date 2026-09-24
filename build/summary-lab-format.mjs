// Escape source text before adding a small, predictable document vocabulary.
export const escapeHtml = value => String(value ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
const inline = value => escapeHtml(value).replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>').replace(/__([^_]+)__/g,'<strong>$1</strong>').replace(/`([^`]+)`/g,'<code>$1</code>').replace(/\*([^*\n]+)\*/g,'<em>$1</em>');
export const looksLikeHtmlDocument = value => /<\/?(?:p|h[1-6]|ul|ol|li|blockquote|table|thead|tbody|tr|th|td|strong|em|br|hr)\b/i.test(String(value ?? ''));
export function youtubeLanguagePayload(generateKorean=false,koreanOnly=false){
 const enabled=Boolean(generateKorean);
 return {generateKorean:enabled,koreanOnly:enabled&&Boolean(koreanOnly),outputMode:enabled?(koreanOnly?'korean_only':'korean_bilingual'):'english'};
}
// A completed transcription may already carry the automatic Summary Lab
// experiment Charlie starts for SUMMARIES-folder audio. Adopt that experiment
// instead of starting a second one: both run the same paid multi-pass review
// over the same transcript.
export function labFanoutPlan(job={}){
 const summaryId=job?.summaryId||'';
 if(!summaryId)return{adoptId:'',start:false,error:'The transcript finished but no saved Summary was returned.'};
 const adoptId=job?.summaryLabId||'';
 return{adoptId,start:!adoptId,error:'',summaryId};
}

// The status line used to read "3 / 3 source parts reviewed · running", which
// looks finished while four of five sections are still unverified. A section is
// drafted, checked against every source part, then revised — so the text on
// screen is a draft until its section appears in completedSections, and a
// cross-section review runs after the last one.
export function labProgressSummary(row={},sectionCount=5){
 const state=row?.state||{};
 const parts=Object.keys(state.parts||{}).length;
 const totalParts=state.totalParts||0;
 const verified=(state.completedSections||[]).length;
 if(row?.status==='complete')return{detail:`${sectionCount} section${sectionCount===1?'':'s'} verified against ${totalParts||parts} source part${(totalParts||parts)===1?'':'s'}`,blocked:''};
 const phase=(!totalParts||parts<totalParts)?'reading the source'
  :verified<sectionCount?'verifying drafts against the source'
  :'final cross-section review';
 return{
  detail:`${parts} of ${totalParts||'—'} source parts reviewed · ${verified} of ${sectionCount} sections verified · ${phase}`,
  blocked:'Each section stays a draft until it has been checked against every source part and revised, so Email all and Copy all unlock when the experiment finishes. Copy section works on any section already verified.'
 };
}

export function labDocument(text='') {
 const lines=text.replace(/\r\n/g,'\n').split('\n'); let html='',paragraph=[],list='';
 const flush=()=>{if(paragraph.length){html+='<p>'+inline(paragraph.join(' '))+'</p>';paragraph=[];}};
 const close=()=>{if(list){html+=`</${list}>`;list='';}};
 for(const line of lines){const s=line.trim();if(/^```/.test(s))continue;
 if(!s){flush();close();continue;}
 const heading=s.match(/^#{1,6}\s+(.+)$/); const item=s.match(/^([-*+]\s+|\d+[.)]\s+)(.+)$/);
 if(heading){flush();close();html+='<h3>'+inline(heading[1])+'</h3>';}
 else if(/^([-*_])\1{2,}$/.test(s)){flush();close();html+='<hr>';}
 else if(item){flush();const kind=/^\d/.test(item[1])?'ol':'ul';if(list!==kind){close();html+=`<${kind}>`;list=kind;}html+='<li>'+inline(item[2])+'</li>';}
 else if(s.startsWith('> ')){flush();close();html+='<blockquote>'+inline(s.slice(2))+'</blockquote>';}
 else {close();paragraph.push(s);}
	}flush();close();return html;
}

// Summary output exists in two historical forms: HTML from the original
// pipeline and Markdown/plain text from the source-reviewed pipeline. Keep one
// display path for both. HTML is used only when the caller supplies Charlie's
// page sanitizer; otherwise it is escaped through the Markdown renderer.
export function documentHtml(text='',sanitizeHtml){
 const value=String(text??'');
 return looksLikeHtmlDocument(value)&&typeof sanitizeHtml==='function'?sanitizeHtml(value):labDocument(value);
}

// Source parts are cited inline as [P1], several in a row where a claim draws
// on more than one. The citation has to survive -- it is how a reader checks a
// claim against the transcript -- but a note carrying hundreds of bracketed
// markers is hard to read. Render each run as one superscript the reader can
// switch off, and leave the stored text and the emailed note untouched.
export function sourceMarkers(html='', mode='reader') {
 return String(html??'').split(/(<[^>]*>)/).map(segment => segment.startsWith('<') ? segment
  : segment.replace(/[ \t]*(?:\[P\d+\])+/g, run => {
     if (mode === 'strip') return '';
     const parts=[...new Set([...run.matchAll(/\[P(\d+)\]/g)].map(m=>m[1]))];
     const label=`Source part${parts.length>1?'s':''} ${parts.join(', ')}`;
     // Email clients do not load the page stylesheet, so a copy that keeps its
     // citations has to carry the styling inline.
     return mode === 'inline'
      ? `<sup style="font-size:9px;font-weight:700;color:#8a8577" title="${label}">${parts.join(',')}</sup>`
      : `<sup class="src" title="${label}">${parts.join(',')}</sup>`;
    })).join('');
}

export function emailDocument(title,sections,sanitizeHtml,markers='strip'){return `<div style="font-family:Calibri,Carlito,Arial,sans-serif;font-size:11pt;line-height:1.55;color:#202020"><h1 style="font-size:11pt;font-weight:700;margin:0 0 18px">${escapeHtml(title)}</h1>`+sections.map(([label,text])=>`<section style="margin:0 0 24px"><h2 style="font-size:11pt;font-weight:700;margin:0 0 8px;padding-bottom:4px;border-bottom:1px solid #d9d5cc">${escapeHtml(label)}</h2>${sourceMarkers(documentHtml(text,sanitizeHtml),markers)}</section>`).join('')+'</div>';}
