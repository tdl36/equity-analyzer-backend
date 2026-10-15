// Browser-session recovery only. Never submits, rebases or accepts research.
export function readEditBuffer(storage,key) {
 const raw=storage.getItem(key);
 if(raw===null)return null;
 if(raw.length>2100000)throw Error('Local note recovery exceeds its size limit. Preserve this session.');
 let value;try{value=JSON.parse(raw);}catch{throw Error('Local note recovery is unreadable. Preserve this session.');}
 if(!value||Object.keys(value).sort().join(',')!=='markdown,revision,title'||!Number.isSafeInteger(value.revision)||value.revision<1||typeof value.title!=='string'||value.title.length>1000||typeof value.markdown!=='string'||value.markdown.length>1000000)throw Error('Local note recovery is invalid. Preserve this session.');
 return value;
}
export function writeEditBuffer(storage,key,value) {
 const raw=JSON.stringify(value);
 // Validate without changing an existing recovery copy first.
 readEditBuffer({getItem:()=>raw},key);
 storage.setItem(key,raw);
 if(storage.getItem(key)!==raw)throw Error('Local note recovery could not be verified. Keep this screen open and save a revision.');
}
