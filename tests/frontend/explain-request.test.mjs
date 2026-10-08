import test from 'node:test';
import assert from 'node:assert/strict';
import {explainRequest} from '../../src/explain-request.mjs';

test('screenshot dispatch uses app proxy and preserves payload without retry', async () => {
    const options = {method:'POST',body:JSON.stringify({attachments:[{fileData:'c2NyZWVuc2hvdA==',mimeType:'image/png'}]})};
    let calls=0;
    await explainRequest('', '', options, async (url, passed) => {
        calls++; assert.equal(url,'/api/decipher'); assert.equal(passed,options);
        return new Response('{"jobId":"test"}');
    });
    assert.equal(calls,1);
    calls=0;
    await assert.rejects(explainRequest('', '', options, async () => {calls++;throw new TypeError('Load failed');}), /may have started/);
    assert.equal(calls,1);
});
test('local backend and follow-up polling use the same transport', async () => {
    await explainRequest('http://127.0.0.1:5000','/followup/test',{},async url=>{
        assert.equal(url,'http://127.0.0.1:5000/api/decipher/followup/test');return new Response('{}');
    });
});
test('proxy HTML failures and expired sessions are actionable',async()=>{
    for(const [status,match] of [[502,/temporarily unavailable/],[413,/too large/],[404,/no longer available/]]) {
        await assert.rejects(explainRequest('','/test',{},async()=>new Response('<html>error</html>',{status})), e=>e.status===status && match.test(e.message));
    }
});
