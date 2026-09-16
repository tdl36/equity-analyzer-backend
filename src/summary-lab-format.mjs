// Escape source text before adding a small, predictable document vocabulary.
export const escapeHtml = value => String(value ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
const inline = value => escapeHtml(value).replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>').replace(/__([^_]+)__/g,'<strong>$1</strong>').replace(/`([^`]+)`/g,'<code>$1</code>').replace(/\*([^*\n]+)\*/g,'<em>$1</em>');
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
export function emailDocument(title,sections){return `<div style="font-family:Calibri,Carlito,Arial,sans-serif;font-size:11pt;line-height:1.5;color:#202020"><h1 style="font-size:11pt;font-weight:700">${escapeHtml(title)}</h1>`+sections.map(([label,text])=>`<h2 style="font-size:11pt;font-weight:700;margin-top:24px">${escapeHtml(label)}</h2>${labDocument(text)}`).join('')+'</div>';}
