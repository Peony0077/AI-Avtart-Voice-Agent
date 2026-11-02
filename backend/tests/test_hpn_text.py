"""Port of HPN backend/apps/helper/tests/test_text.py: reading a question (app/hpn/text.py is HPN's own file).
Unchanged except the imports (unittest instead of Django; has_card_number/mask_private live in app.hpn.engine)."""
import unittest

from app.hpn import text as T
from app.hpn.engine import CARD_RE, mask_private


class NormaliseTests(unittest.TestCase):
    def test_case_accents_punctuation_and_spaces(self):
        self.assertEqual(T.normalise("  What's the PRICE   of a Wédding?!  "),
                         "what is the price of a wedding")

    def test_contractions_are_spelled_out(self):
        self.assertEqual(T.normalise("I'm sure you can't"), "i am sure you can not")
        self.assertEqual(T.normalise("dont   u do bday"), "do not you do birthday")

    def test_only_the_first_500_characters_are_read(self):
        long = "wedding " + "x" * 2000 + " foam"
        self.assertIn("wedding", T.normalise(long))
        self.assertNotIn("foam", T.normalise(long))

    def test_emoji_only_is_empty(self):
        self.assertEqual(T.normalise("😀🎉!!"), "")


class ConceptTests(unittest.TestCase):
    def test_synonyms_share_an_idea(self):
        for phrase in ("how much", "what does it cost", "your rates", "a quote", "fees",
                       "what do you charge", "precio"):
            self.assertIn("price", T.concepts(T.normalise(phrase)), phrase)
        for phrase in ("the bride", "our reception", "getting married", "boda"):
            self.assertIn("wedding", T.concepts(T.normalise(phrase)), phrase)
        for phrase in ("this evening", "today", "tonight"):
            self.assertIn("tonight", T.concepts(T.normalise(phrase)), phrase)
        self.assertIn("foam", T.concepts("bubbles"))
        self.assertIn("karaoke", T.concepts("where can i sing"))
        self.assertIn("bingo", T.concepts("jukebox"))
        self.assertIn("dj", T.concepts("deejay"))
        self.assertIn("book", T.concepts("can we reserve"))
        self.assertIn("request", T.concepts("play my song"))
        self.assertIn("contact", T.concepts("your phone"))
        self.assertIn("where", T.concepts("which bar"))

    def test_whole_words_only(self):
        # "rate" must not fire inside "celebrate".
        self.assertNotIn("price", T.concepts(T.normalise("we celebrate")))


class GreetingTests(unittest.TestCase):
    def test_a_greeting_is_stripped_from_the_question(self):
        self.assertEqual(T.strip_greeting("hi there how much is a wedding"),
                         (True, "how much is a wedding"))
        self.assertEqual(T.strip_greeting("good morning"), (True, ""))
        self.assertEqual(T.strip_greeting("hola"), (True, ""))
        self.assertEqual(T.strip_greeting("hey guys"), (True, ""))
        self.assertEqual(T.strip_greeting("how much is a wedding"),
                         (False, "how much is a wedding"))

    def test_spanish_is_noticed(self):
        self.assertTrue(T.is_spanish("hola cuanto cuesta una boda"))
        self.assertFalse(T.is_spanish("how much is a wedding"))


class GibberishTests(unittest.TestCase):
    def test_key_mashing(self):
        for mash in ("asdfgh", "jjjjjj", "qwrtzp kdjfg"):
            self.assertTrue(T.looks_like_gibberish(mash, set()), mash)

    def test_ordinary_words_are_not(self):
        self.assertFalse(T.looks_like_gibberish("what about llamas", set()))
        self.assertFalse(T.looks_like_gibberish("foam", {"foam"}))


class MaskingTests(unittest.TestCase):
    def test_card_numbers_are_recognised_and_masked(self):
        for card in ("4111 1111 1111 1111", "4111-1111-1111-1111", "4111111111111111",
                     "378282246310005"):
            self.assertTrue(has_card_number(f"my card is {card} thanks"), card)
            self.assertNotIn(card, mask_private(f"my card is {card}"))
            self.assertIn("[number removed]", mask_private(card))

    def test_phone_numbers_and_emails_are_masked(self):
        masked = mask_private("call me on (832) 555-0101 or jo@example.com")
        self.assertNotIn("555-0101", masked)
        self.assertNotIn("jo@example.com", masked)

    def test_dates_and_guest_counts_survive(self):
        self.assertEqual(mask_private("150 guests on 6/14/2027"), "150 guests on 6/14/2027")
        self.assertFalse(has_card_number("150 guests on 6/14/2027"))


def has_card_number(text):
    return bool(CARD_RE.search(text or ""))
