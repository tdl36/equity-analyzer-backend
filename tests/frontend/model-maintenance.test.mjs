import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {build} from 'esbuild';
import {createRequire} from 'node:module';
const require=createRequire(import.meta.url);
const React=require('react'),{renderToStaticMarkup}=require('react-dom/server');
const bundle=await build({entryPoints:['src/model-maintenance.jsx'],bundle:true,write:false,platform:'node',format:'cjs',external:['react']});
const module={exports:{}};new Function('require','module','exports',bundle.outputFiles[0].text)(require,module,module.exports);
const {ModelPolicyView}=module.exports;
const render=(props)=>renderToStaticMarkup(React.createElement(ModelPolicyView,props));
test('model policy shows uncertainty, retirement and failed review without hiding last good configuration',()=>{
 const r=JSON.parse(readFileSync('model_registry.json','utf8'));
 const html=render({data:{...r,revision:'fixture',policy:'Per-job bills vary',maintenanceDependency:'Mac and Codex required',lastCheck:{checkedAt:'2026-10-05',status:'attention',summary:'Provider unavailable <script>bad()</script>'},notices:[{model:'gpt-image-1',date:'2026-10-23',message:'Replacement needs review'}]}});
 for(const text of ['claude-sonnet-5-5','claude-opus-4-6','Per-job bills vary','Mac and Codex required','2026-10-23','attention','&lt;script&gt;'])assert.ok(html.includes(text),text);
 assert.ok(!html.includes('<script>'));
});
test('loading and failure are explicit',()=>{
 assert.match(render({data:null}),/Loading model policy/);
 assert.match(render({data:null,error:'Status unavailable'}),/role="alert"/);
 assert.ok(!render({data:null,error:'Status unavailable'}).includes('Loading model policy'));
});
