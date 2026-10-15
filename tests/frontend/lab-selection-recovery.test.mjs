import test from 'node:test';
import assert from 'node:assert/strict';
import {readSelectionDraft,writeSelectionDraft,selectionRecoveryKey} from '../../src/lab-selection-recovery.mjs';
const storage=()=>{const data=new Map();return {getItem:k=>data.get(k)||null,setItem:(k,v)=>data.set(k,v)};};
test('source drafts preserve both task selections and baseline identity',()=>{
 const s=storage(),key=selectionRecoveryKey('api','ABT');
 writeSelectionDraft(s,key,'ABT',4,{thesis:['charlie:thesis_1'],note:['icloud:folder/file.pdf']});
 assert.deepEqual(readSelectionDraft(s,key,'ABT'),{version:1,ticker:'ABT',revision:4,selection:{thesis:['charlie:thesis_1'],note:['icloud:folder/file.pdf']}});
 assert.notEqual(key,selectionRecoveryKey('other-api','ABT'));assert.notEqual(key,selectionRecoveryKey('api','CVS'));
 assert.throws(()=>readSelectionDraft(s,key,'CVS'));
});
test('malformed and unconfirmed storage never claims recoverability',()=>{
 const s=storage();s.setItem('key','broken');assert.throws(()=>readSelectionDraft(s,'key','ABT'));
 assert.throws(()=>writeSelectionDraft({setItem(){},getItem(){return null;}},'key','ABT',1,{thesis:[],note:[]}));
 assert.throws(()=>writeSelectionDraft(s,'key','ABT',1,{thesis:['duplicate','duplicate'],note:[]}));
});
