import test from 'node:test';
import assert from 'node:assert/strict';
import {thesisFields,setText,revisionPackage,correctionBody} from '../../src/package-editor-model.mjs';
const analysis={thesis:{summary:'Old',pillars:[{id:'p1',title:'Pillar',description:'Evidence',sources:[{sourceId:'s'}]}]},signposts:[{id:'s1',metric:'Growth',target:'5%',timeframe:'2027'}],threats:[{id:'r1',threat:'Risk',triggerPoints:'Miss'}],conclusion:'Conditional'};
test('edit only narrative fields, preserving evidence, IDs and original',()=>{
 const next=setText(analysis,['thesis','pillars',0,'description'],'Revised');
 assert.equal(next.thesis.pillars[0].id,'p1');assert.deepEqual(next.thesis.pillars[0].sources,analysis.thesis.pillars[0].sources);assert.equal(analysis.thesis.pillars[0].description,'Evidence');
 assert.throws(()=>setText(analysis,['thesis','pillars',0,'sources'],'fake'));
 assert.equal(thesisFields(next).length,9);
});
test('revision is deterministic for replay and never claims approval',()=>{
 const draft={id:'old',package:{analysis,baseline:{hash:'live'},sourceRegister:[{id:'s'}]}};
 const one=revisionPackage(draft,analysis,'request');assert.deepEqual(one,revisionPackage(draft,analysis,'request'));assert.equal(one.baseline.hash,'live');assert.equal(one.provenance.editorialRevision.parentDraftId,'old');assert.equal(draft.package.provenance,undefined);
});
test('onepager retains evidence without promoting editorial changes to verification',()=>{
 const body=correctionBody({proposal:{path:'/summary/one_liner',basis:'interpretation',sourceId:'s',excerpt:'original passage',locator:'page 1',before:'old',reviewOrigin:'investor',additionalEvidence:[{sourceId:'s2',excerpt:'more evidence',locator:'page 2',filename:'original.pdf',sha256:'hash',originalPage:2}]}},'new','f','id');
 assert.equal(body.excerpt,'original passage');assert.deepEqual(body.additionalEvidence,[{sourceId:'s2',excerpt:'more evidence',locator:'page 2',originalPage:2}]);assert.equal(body.reviewOrigin,undefined);assert.match(body.reason,/no independent fact-check/);assert.throws(()=>correctionBody({proposal:{}},'x'.repeat(301),'f','id'));
});
