import copy
import hashlib
import unittest
from research_source_budget import fit_sources,MARKER


def source(size,ident):
    text=(ident+' '+'Evidence text. '*size)[:size]
    return {'id':ident,'filename':ident+'.txt','originalHash':ident,
            'text':text,'extractionHash':hashlib.sha256(text.encode()).hexdigest()}


class SourceBudgetTests(unittest.TestCase):
    def test_small_pack_unchanged_and_not_mutated(self):
        pack=[source(100,'a'),source(500,'b')];original=copy.deepcopy(pack)
        self.assertEqual(fit_sources(pack),original);self.assertEqual(pack,original)

    def test_large_pack_fair_bounded_exact_ranges_and_hashes(self):
        pack=[source(n,str(i)) for i,n in enumerate([45000,9000,10000,36000,58000,26000])]
        result=fit_sources(pack)
        self.assertLessEqual(sum(len(s['text']) for s in result),160000)
        self.assertEqual(result,fit_sources(pack))
        self.assertEqual(result[1],pack[1])
        for original,s in zip(pack,result):
            self.assertEqual(s['originalHash'],original['originalHash'])
            if 'coverage' not in s:continue
            c=s['coverage'];ranges=c['includedRanges']
            self.assertEqual(s['text'],MARKER.join(original['text'][a:b] for a,b in ranges))
            self.assertEqual(c['omittedCharacters'],ranges[1][0]-ranges[0][1])
            self.assertEqual(c['fullExtractionHash'],original['extractionHash'])
            self.assertEqual(s['extractionHash'],hashlib.sha256(s['text'].encode()).hexdigest())
            self.assertNotEqual(s['text'],original['text'])

    def test_unreadable_source_not_silently_dropped(self):
        with self.assertRaises(ValueError):fit_sources([source(0,'empty')])
