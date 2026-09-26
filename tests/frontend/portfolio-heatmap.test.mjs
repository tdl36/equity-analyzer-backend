import test from 'node:test';
import assert from 'node:assert/strict';
import {layout,groupedLayout,parseHoldings,tileColor,formatReturn} from '../../src/portfolio-heatmap-model.mjs';
test('treemap preserves relative area without overlap',()=>{
 const rows=layout([{size:50},{size:30},{size:15},{size:5}],0,0,800,600);
 assert.equal(rows.length,4);for(const r of rows)assert.ok(Math.abs(r.width*r.height/480000-r.size/100)<1e-8);
 for(let i=0;i<rows.length;i++)for(let j=i+1;j<rows.length;j++){const a=rows[i],b=rows[j];assert.ok(a.x+a.width<=b.x+1e-8||b.x+b.width<=a.x+1e-8||a.y+a.height<=b.y+1e-8||b.y+b.height<=a.y+1e-8);}
});
test('short weights use absolute area and remain signed',()=>{const g=groupedLayout([{ticker:'A',weight:-5,sector:'Health'},{ticker:'B',weight:10,sector:'Tech'}]);assert.equal(g.reduce((s,r)=>s+r.size,0),15);assert.equal(g.find(r=>r.name==='Health').tiles[0].weight,-5);});
test('CSV parses quoted company names, percentages and spreadsheet TSV',()=>{assert.equal(parseHoldings('ticker,weight,company\r\nABT,5%,"Abbott, Inc."')[0].company,'Abbott, Inc.');assert.equal(parseHoldings('ticker\tweight\nABT\t-3')[0].weight,-3);});
test('invalid and duplicate holdings never silently overwrite',()=>{for(const t of ['ticker,weight\nABT,','ticker,weight\nABT,0','ticker,weight\nABT,2\nABT,4','ticker,weight\nABT,Infinity'])assert.throws(()=>parseHoldings(t));});
test('missing return is not zero',()=>{assert.equal(formatReturn(null),'No data');assert.equal(formatReturn(0),'0.00%');assert.notEqual(tileColor(null),tileColor(0));});
