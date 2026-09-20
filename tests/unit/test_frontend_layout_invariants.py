"""Source invariants for two defects that were invisible until a user hit them."""
import re
import unittest
from pathlib import Path

APP = Path('src/app.jsx').read_text()
LAB = Path('src/summary-lab.jsx').read_text()


class ScrollToTopTests(unittest.TestCase):
    """The document listener behind the button used to fire only for elements
    carrying the Tailwind class overflow-y-auto, which excluded every workspace
    styled with its own CSS. Summary Lab had no button at all."""

    def test_a_document_level_listener_catches_every_view(self):
        self.assertIn("document.addEventListener('scroll', onAnyScroll, true)", APP)
        self.assertIn("document.removeEventListener('scroll', onAnyScroll, true)", APP)

    def test_it_uses_the_capture_phase_because_scroll_does_not_bubble(self):
        listener = APP[APP.index('const onAnyScroll'):APP.index("document.addEventListener('scroll'")]
        self.assertIn('scrollContainerRef.current = el', listener)
        self.assertIn('setShowScrollTop(el.scrollTop > 300)', listener)

    def test_it_does_not_gate_on_a_styling_class(self):
        self.assertEqual(APP.count("addEventListener('scroll'"), 1)
        self.assertNotIn("includes('overflow-y-auto')", APP)

    def test_the_button_still_exists_and_is_reachable(self):
        self.assertIn('title="Scroll to top"', APP)
        self.assertIn('onClick={scrollToTop}', APP)


class SummaryLabNarrowScreenTests(unittest.TestCase):
    """At 375px the page, panel and reader each added padding, leaving about
    70% of the screen for text."""

    def narrow_block(self):
        match = re.search(r'@media\(max-width:600px\)\{(.+?)\}\}', LAB, re.S)
        self.assertIsNotNone(match, 'no narrow-screen block in Summary Lab')
        return match.group(0)

    def test_narrow_rules_come_after_the_900px_block(self):
        # Equal specificity: whichever media block is written last wins, so a
        # 600px block placed first is silently undone at 375px.
        self.assertLess(LAB.index('@media(max-width:900px)'),
                        LAB.index('@media(max-width:600px)'))

    def test_each_nested_container_is_tightened(self):
        block = self.narrow_block()
        for selector in ('.summary-lab{padding:10px 8px',
                         '.summary-lab .panel{padding:14px 10px',
                         '.summary-lab .reader{padding:16px 12px',
                         '.summary-lab .section-body{padding:12px 8px'):
            self.assertIn(selector, block, selector)

    def test_the_bottom_gutter_clears_the_mobile_navigation(self):
        self.assertIn('112px', self.narrow_block())
