import test from 'node:test';import assert from 'node:assert/strict';
import {emptyDraft,draftKey,validDraft,readDraft,writeDraft} from '../../src/processing-proposal-draft.mjs';
const storage=()=>{const data=new Map();return {getItem:k=>data.get(k),setItem:(k,v)=>data.set(k,v),removeItem:k=>data.delete(k)}};
test('authoring choices survive reload and remain collection/API scoped',()=>{
 const store=storage(),key=draftKey('local','one'),d={...emptyDraft(),selected:['original.pdf'],supplements:['12'],fullNote:true};
 writeDraft(store,key,d);assert.deepEqual(readDraft(store,key).value,d);assert.equal(readDraft(store,draftKey('prod','one')).value,null);assert.equal(readDraft(store,draftKey('local','two')).value,null);
 writeDraft(store,key,emptyDraft());assert.equal(readDraft(store,key).value,null);
});
test('restored authoring never contains request identity, approval, hashes or source bodies',()=>{
 for(const extra of [{requestId:'uuid'},{approved:true},{digest:'a'.repeat(64)},{file_data:'private'}])assert.equal(validDraft({...emptyDraft(),...extra}),null);
 const d={...emptyDraft(),selected:['one']},clean=validDraft(d);clean.selected.push('two');assert.deepEqual(d.selected,['one']);
});
test('malformed, oversized, cross-purpose and quota-failed drafts stay explicit',()=>{
 for(const change of [{selected:['one','one']},{supplements:['../12']},{fullNote:'true'},{instruction:'x'.repeat(1801)},{purpose:'event_note',outputs:['thesis']},{version:2}])assert.equal(validDraft({...emptyDraft(),...change}),null);
 const store=storage();store.setItem('bad','{bad');assert.ok(readDraft(store,'bad').error);
 const denied={getItem(){throw Error('denied')},setItem(){throw Error('quota')},removeItem(){throw Error('denied')}};
 assert.ok(readDraft(denied,'bad').error);assert.throws(()=>writeDraft(denied,'key',{...emptyDraft(),selected:['one']}),/quota/);
});
