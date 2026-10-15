import test from 'node:test';
import assert from 'node:assert/strict';
import {synthesisRows} from '../../src/collection-synthesis-model.mjs';
test('unavailable reads never claim synthesis is paused',()=>{
 assert.ok(synthesisRows().every(r=>r.status==='Status unavailable'));
 const rows=synthesisRows({catalysts:{enabled:false},analysts:{analysts:[{autoMode:{enabled:true}},{}]}});
 assert.equal(rows[0].status,'Off');assert.match(rows[2].status,/1 analysts on; other settings unavailable/);
});
test('independent analyst switches and UTC expiry are reflected',()=>{
 const now=Date.parse('2026-10-15T14:00:00Z');
 const data={catalysts:{enabled:false},intake:{enabled:false},analysts:{analysts:[
  {autoMode:{enabled:true,earnings:false,takeaways:false}},
  {autoMode:{enabled:false,takeaways:true,expires_at:'2026-10-15T13:59:00'}},
 ]}};
 assert.equal(synthesisRows(data,now)[2].status,'Off for all analysts');
 data.analysts.analysts[1].autoMode.expires_at='2026-10-15T14:01:00';
 assert.equal(synthesisRows(data,now)[2].status,'1 of 2 analysts on');
 data.analysts.analysts[1].autoMode.expires_at='invalid';
 assert.equal(synthesisRows(data,now)[2].status,'Status unavailable');
});
