import {test} from 'node:test';
import assert from 'node:assert/strict';
import {editionSelection,basisLabel,claimLabel} from '../../src/onepager-edition-model.mjs';
import {editionReviewState} from '../../src/onepager-edition-model.mjs';
const pack={status:'pending_only',assignmentId:'a',runId:'r',fingerprint:'f',fundamentalsDigest:'d',parentId:null};
const rows=[{id:'c1',fingerprint:'f',stale:false,proposal:{path:'/summary/view'}},{id:'c2',fingerprint:'f',stale:false,proposal:{path:'/summary/other'}}];
test('decided editions never offer a pending preview',()=>{
 assert.equal(editionReviewState({status:'pending',stale:false}).canPreview,true);
 assert.equal(editionReviewState({status:'pending',stale:true}).canPreview,false);
 for(const action of ['accept','reject'])assert.equal(editionReviewState({decision:{action},stale:false}).canPreview,false);
 assert.match(editionReviewState({decision:{action:'accept'},currentAccepted:true}).label,/current review/);
 assert.match(editionReviewState({decision:{action:'accept'},currentAccepted:false}).label,/prior review/);
 assert.equal(editionReviewState({status:'accepted'}).canPreview,false);
 assert.equal(editionReviewState({decision:{action:'unknown'}}).canPreview,false);
});
test('edition payload binds ordered selection without altering originals',()=>{const before=JSON.stringify(rows);const ids=['c2','c1'];const p=editionSelection(pack,rows,ids,'request');assert.deepEqual(p,{requestId:'request',assignmentId:'a',runId:'r',fingerprint:'f',fundamentalsDigest:'d',parentId:null,correctionIds:ids});p.correctionIds.pop();assert.equal(ids.length,2);assert.equal(JSON.stringify(rows),before);assert.equal(p.status,undefined);});
test('missing, stale, duplicate and same-claim corrections cannot be selected',()=>{for(const ids of [[],['unknown'],['c1','c1'],Array(21).fill('c1')])assert.throws(()=>editionSelection(pack,rows,ids,'x'));assert.throws(()=>editionSelection(pack,[{...rows[0],stale:true}],['c1'],'x'));assert.throws(()=>editionSelection(pack,[{...rows[0],fingerprint:'old'}],['c1'],'x'));assert.throws(()=>editionSelection(pack,[rows[0],{...rows[1],proposal:rows[0].proposal}],['c1','c2'],'x'));});
test('review labels distinguish facts and interpretation',()=>{assert.equal(basisLabel('interpretation'),'Interpretation');assert.equal(basisLabel('broker_estimate'),'Broker estimate');assert.equal(basisLabel('approved'),'Claim type unavailable');assert.equal(claimLabel('/scenario_analysis/cases/0/eps_or_fcf'),'scenario analysis · cases · 1 · eps or fcf');});

test('pending preview identity cannot be replaced by another edition or acceptance flag',async()=>{
 const {verifiedEditionPreview}=await import('../../src/onepager-edition-model.mjs');
 const value={id:'e',assignmentId:'a',runId:'r',ticker:'TEST',status:'pending',layout:'illustrated',html:'<p>Pending</p>'};
 assert.equal(verifiedEditionPreview(value,'e','a','r','TEST'),value);
 for(const patch of [{id:'other'},{assignmentId:'other'},{runId:'other'},{ticker:'OTHER'},{status:'accepted'},{layout:'other'},{html:''}])assert.throws(()=>verifiedEditionPreview({...value,...patch},'e','a','r','TEST'));
});
