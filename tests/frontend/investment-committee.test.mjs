import test from 'node:test';
import assert from 'node:assert/strict';
import {committeeDocument} from '../../src/investment-committee-model.mjs';
const run={id:'synthetic',ticker:'SYNTH',status:'attention',model:'fixture',baseline:{revision:4},sources:[{id:'s',filename:'<original>',originalHash:'frozen-sha'}],state:{sections:[{id:'downside',title:'Downside case',claims:[{id:'downside:0',basis:'interpretation',review:'needs_review',statement:'<script>dissent</script>',evidence:[]}],gaps:['Missing data']}],challenges:[{id:'challenge-0',claimIds:['downside:0'],question:'Unresolved issue'}],responses:[{challengeId:'challenge-0',status:'contested',reason:'Conflicting evidence',nextTest:'Next filing',proposedChange:'Do not accept yet'}]}};
test('committee export preserves dissent, partial status, source identity and unaccepted proposals',()=>{
 const html=committeeDocument(run);
 for(const text of ['attention','case R4','&lt;script&gt;','needs_review','contested','frozen-sha','No investor approval','Do not accept yet'])assert.ok(html.includes(text),text);
 assert.ok(!html.includes('<script>'));
});
test('committee renders named challenge and clearly labeled response without case mutation controls',async()=>{
 const {build}=await import('esbuild');const {createRequire}=await import('node:module');const require=createRequire(import.meta.url);
 const React=require('react'),{renderToStaticMarkup}=require('react-dom/server');
 const bundle=await build({entryPoints:['src/investment-committee.jsx'],bundle:true,write:false,platform:'node',format:'cjs',external:['react']});
 const module={exports:{}};new Function('require','module','exports',bundle.outputFiles[0].text)(require,module,module.exports);
 const html=renderToStaticMarkup(React.createElement(module.exports.CommitteeFindings,{run}));
 assert.match(html,/not five independent people/);assert.match(html,/Unaccepted proposal/);assert.match(html,/Downside case/);assert.ok(!html.includes('<button'));
});
