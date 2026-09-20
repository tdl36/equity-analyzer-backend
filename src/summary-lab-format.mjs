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

export function emailDocument(title,sections,sanitizeHtml){return `<div style="font-family:Calibri,Carlito,Arial,sans-serif;font-size:11pt;line-height:1.55;color:#202020"><h1 style="font-size:11pt;font-weight:700;margin:0 0 18px">${escapeHtml(title)}</h1>`+sections.map(([label,text])=>`<section style="margin:0 0 24px"><h2 style="font-size:11pt;font-weight:700;margin:0 0 8px;padding-bottom:4px;border-bottom:1px solid #d9d5cc">${escapeHtml(label)}</h2>${documentHtml(text,sanitizeHtml)}</section>`).join('')+'</div>';}
