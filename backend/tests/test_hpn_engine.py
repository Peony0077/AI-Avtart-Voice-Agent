"""Port of HPN's Event-helper tests (backend/apps/helper/tests: test_accuracy_round2, test_smalltalk, test_live,
test_knowledge_base) against the avatar's engine (app/hpn), with HPN's test schedule (hpn_fixture.py).

The same questions must get the same answers as on h-townparty.com. Where the avatar deliberately differs, the
test says so in a comment starting "AVATAR:" (who is answering, the DJs' team app left out, no live DJ rota,
the website-helper-only topics rewritten in knowledge/hpn/avatar_overrides.json).
"""
import json
import re

from app.hpn import engine
from app.hpn.knowledge import Offering, Topic
from tests.hpn_fixture import AFTERNOON, EMAIL, KNOWLEDGE, MORNING, PHONE, THURSDAY_8PM, HelperTestCase

# ---------------------------------------------------------------- from HPN test_accuracy_round2.py

class RoutingTests(HelperTestCase):
    def intent(self, text, last=None):
        return self.ask(text, last=last)["intent"]

    def test_a_code_is_a_question_for_us_not_programming(self):
        # AVATAR: the DJs' team app topics are left out (customers only): never the team sign-in answer
        self.assertNotIn(self.intent("My sign-in code didn't work"), ("topic:team-sign-in", "topic:team-app"))
        self.assertEqual(self.intent("What is the QR code on the table?"), "topic:venue-qr")
        self.assertEqual(self.intent("tell me a joke"), "smalltalk.offtopic")

    def test_the_extra_hours_chip_gets_the_extra_hour_prices(self):
        self.assertEqual(self.intent("Extra hours"), "prices.extra_hours")

    def test_a_goodbye_is_only_a_goodbye_when_nothing_else_is_asked(self):
        for text in ("good night", "thanks, good night", "ok thanks, bye", "later"):
            self.assertEqual(self.intent(text), "smalltalk.bye", text)
        self.assertEqual(self.intent("thanks for your help"), "smalltalk.thanks")
        self.assertTrue(self.intent("thanks, what is on this weekend").startswith("nights."))
        self.assertEqual(self.intent("will you be at Molly's Pub Tomball later"), "nights.week")

    def test_a_venue_by_name_gets_its_own_nights(self):
        reply = self.ask("is there karaoke at mollys pub tomball")
        self.assertTrue(reply["cards"])
        self.assertTrue(all(c["title"] == "Molly's Pub Tomball" for c in reply["cards"]))
        self.assertIn("at Molly's Pub Tomball", reply["text"])
        # A shortened name only when it is still two words: "Tony's" alone is not a venue.
        self.assertTrue(all(c["title"] == "Tony's Tavern"
                            for c in self.ask("what nights at tonys tavern")["cards"]))

    def test_where_are_you_is_the_contact_answer(self):
        reply = self.ask("where are you located?")
        self.assertEqual(reply["intent"], "contact")
        self.assertIn("Tomball", reply["text"])

    def test_a_call_back_is_not_promised(self):
        for text in ("can you call me back", "my number is 832 555 1234 call me"):
            reply = self.ask(text)
            self.assertEqual(reply["intent"], "smalltalk.callback", text)
            self.assertNotIn("Of course", reply["text"])
            self.assertIn("contact form", reply["text"])

    def test_misspelt_wedding_and_follow_ups(self):
        self.assertEqual(self.intent("how much for a weding"), "prices.wedding")
        self.assertEqual(self.intent("what about foam?", last="prices.party"), "prices.foam")
        self.assertEqual(self.intent("what about weddings?", last="prices.foam"),
                         "prices.wedding")

    def test_an_event_named_and_asked_for_gets_its_packages(self):
        reply = self.ask("necesito un dj para una fiesta")
        self.assertEqual(reply["intent"], "prices.party")
        self.assertTrue(reply["text"].startswith("Yes, we can help with that."))
        # Not for a question about the event that nobody has answered.
        self.assertNotEqual(self.intent("Does the foam damage grass?"), "prices.foam")

    def test_bar_mitzvah_team_schedule_and_who_are_you_guys(self):
        self.assertEqual(self.intent("Do you do bar mitzvahs?"), "topic:parties")
        # AVATAR: the team app topic is left out (customers only)
        self.assertNotEqual(self.intent("Where is my team schedule?"), "topic:team-app")
        self.assertEqual(self.intent("Who are you guys?"), "topic:about")
        self.assertEqual(self.intent("are you a bot?"), "smalltalk.who")

    def test_a_song_for_the_event_is_not_tonights_djs(self):
        self.assertEqual(self.intent("Do you take song requests from guests?"),
                         "topic:guest-requests")
        self.assertEqual(self.intent("Can you play our first dance song?"),
                         "topic:planning-form")
        self.assertEqual(self.intent("request a song"), "djs.now")

    def test_paying_to_jump_the_queue_is_no(self):
        reply = self.ask("Can I pay to get my song played sooner?")
        self.assertEqual(reply["intent"], "topic:request-priority")
        self.assertTrue(reply["text"].startswith("No."))

    def test_a_price_question_about_an_add_on_gets_its_price(self):
        reply = self.ask("dancing on a cloud price")
        self.assertEqual(reply["intent"], "prices.addon")
        self.assertEqual([c["title"] for c in reply["cards"]], ["Dancing on a Cloud"])

    def test_extras_that_are_not_wedding_extras_say_so(self):
        reply = self.ask("how much is uplighting for my wedding")
        self.assertIn("isn't a wedding extra", reply["text"])
        self.assertNotIn("Craig talks extras", reply["text"])


class NoSourceTests(HelperTestCase):
    """Nobody has written these down, so the team confirms them; the helper never guesses."""

    def test_each_goes_to_the_team(self):
        for text, concept in (
                ("Is there a cover charge?", "policy_cover"),
                ("how much is it at Molly's Pub Tomball", "policy_cover"),
                ("how much is karaoke tonight", "policy_cover"),
                ("Do I need to be 21 for karaoke?", "policy_age"),
                ("Can kids come to bingo night?", "policy_age"),
                ("Do I need a permit for foam?", "policy_permit"),
                ("Can the DJ drink at our wedding?", "policy_conduct"),
                ("what's the dress code for the DJ", "policy_conduct"),
                ("Do you play tejano?", "policy_genre"),
                ("What kind of music do you play?", "policy_genre"),
                ("When do you arrive?", "policy_arrival"),
                ("Is foam safe for eyes?", "policy_health"),
                ("Is foam slippery?", "policy_health"),
                ("Can I pick my DJ?", "policy_dj_choice")):
            reply = self.ask(text)
            self.assertEqual(reply["intent"], f"smalltalk.{concept}", text)
            self.assertFalse(reply["confident"], text)
            self.assertTrue(reply["text"].startswith("That's one for the team"), text)
            self.assertNotIn("$", reply["text"], text)

    def test_the_karaoke_add_on_still_has_its_price(self):
        self.assertEqual(self.ask("how much is karaoke")["intent"], "prices.addon")

    def test_a_foam_fundraiser_with_tickets_is_still_the_fundraiser_topic(self):
        self.assertEqual(self.ask("can we sell tickets for a foam fundraiser")["intent"],
                         "topic:foam-fundraiser")


class ContentTests(HelperTestCase):
    def text(self, question):
        return self.ask(question)["text"]

    def test_scope_of_every_package_claims(self):
        self.assertIn("Every wedding and party package includes a professional DJ and emcee",
                      self.text("Do you MC?"))
        self.assertIn("Every wedding and party package includes a wireless microphone",
                      self.text("Do you have a mic?"))
        self.assertNotIn("Delivery and setup within our standard service area are included",
                         self.text("is there a travel fee"))

    def test_planning_form_reminders_match_the_code(self):
        text = self.text("what is the planning form")
        self.assertIn("a week before it locks and again the day before", text)
        self.assertIn("the day before your event", text)

    def test_no_unsourced_promises(self):
        self.assertNotIn("particular DJ", self.text("Who are your DJs?"))
        self.assertNotIn("the whole time", self.text("is the foam safe?"))
        self.assertIn("younger than 3", self.text("Is foam OK for toddlers?"))

    def test_not_wedding_extras(self):
        self.assertIn("aren't offered for weddings",
                      self.text("Can you project our names at our wedding?"))
        self.assertIn("Video Guest Book and the Guest Book Booth are for parties",
                      self.text("guest book booth for my wedding"))

    def test_helper_privacy_says_exactly_what_is_removed(self):
        text = self.text("What happens to what I type here?")
        self.assertNotIn("without your name", text)
        self.assertIn("card numbers, phone numbers and email addresses are taken out", text)

    def test_new_topics(self):
        for question, slug in (("Do you use AI?", "ai-check"),
                               ("What happened to the AI chat?", "concierge-replaced"),
                               ("Can I send photos of the venue?", "attachments"),
                               ("What should I put in my enquiry?", "enquiry-details"),
                               ("Does the app track my location?", "app-location"),
                               ("Do you use cookies?", "cookies"),
                               ("Do you bring an assistant?", "event-technician"),
                               ("I forgot my password", "sign-in")):
            self.assertEqual(self.ask(question)["intent"], f"topic:{slug}", question)
        self.assertIn("up to 8 MB each", self.text("Can I send photos of the venue?"))
        self.assertIn("Download app", self.text("Is there an app?"))

    def test_no_answer_names_another_platform(self):
        # Read inside the iPhone and Android apps too: Apple 2.3.10 (store check, 29 Oct 2026).
        banned = re.compile(r"android|google play|play store|app store|iphone|apple maps|google maps", re.I)
        for topic in self.kb.topics:
            self.assertIsNone(banned.search(self.kb.fill(topic.answer_template)), topic.slug)
        for question in ("Is there an app?", "Does the app track my location?", "hi"):
            self.assertIsNone(banned.search(self.text(question)), question)

# ---------------------------------------------------------------- from HPN test_smalltalk.py

class GreetingTests(HelperTestCase):
    def test_the_opening_message_greets_by_houston_time_of_day(self):
        self.assertTrue(self.engine.start(MORNING).text.startswith("Good morning!"))
        self.assertTrue(self.engine.start(AFTERNOON).text.startswith("Good afternoon!"))
        self.assertTrue(self.engine.start(THURSDAY_8PM).text.startswith("Good evening!"))
        opening = self.engine.start(MORNING).as_dict()
        self.assertEqual(opening["kind"], "greeting")
        # AVATAR: she says her name ("I'm the Houston Party Network Event helper" on the website)
        self.assertIn("I'm Emma, Houston Party Network's AI Agent. I answer from our own "
                      "prices and schedule.", opening["text"])
        self.assertEqual(opening["chips"][:6], ["Plan an event", "Prices and packages",
                                                "What's on tonight?", "Request a song",
                                                "Weddings", "Foam parties"])

    def test_greetings(self):
        for hello in ("hi", "Hello!", "hey", "howdy", "good morning", "Good afternoon",
                      "good evening", "hola", "yo", "hi there"):
            reply = self.ask(hello, now=MORNING)
            self.assertEqual(reply["intent"], "greeting", hello)
            self.assertTrue(reply["chips"], hello)
        self.assertTrue(self.ask("hi", now=MORNING)["text"].startswith("Good morning!"))
        self.assertTrue(self.ask("hi", now=THURSDAY_8PM)["text"].startswith("Good evening!"))

    def test_a_greeting_with_a_question_answers_the_question_after_a_short_hello(self):
        reply = self.ask("hi, how much is a wedding?", now=AFTERNOON)
        self.assertEqual(reply["intent"], "prices.wedding")
        self.assertTrue(reply["text"].startswith("Good afternoon!\n\n"))
        self.assertTrue(reply["cards"])
        reply = self.ask("Hey there! what's on tonight?")
        self.assertEqual(reply["intent"], "nights.tonight")

    def test_hola_with_a_question_answers_in_english_with_a_spanish_line(self):
        reply = self.ask("hola, precio boda")
        self.assertEqual(reply["intent"], "prices.wedding")
        self.assertTrue(reply["text"].startswith("¡Hola!"))
        self.assertIn("llámenos al " + PHONE, reply["text"])
        self.assertIn("gracias", "gracias")
        self.assertEqual(self.ask("gracias")["intent"], "smalltalk.thanks")
        self.assertEqual(self.ask("adios")["intent"], "smalltalk.bye")
        self.assertIn("escríbanos a " + EMAIL, self.ask("fiesta precio")["text"])


class ConversationTests(HelperTestCase):
    def test_how_are_you(self):
        for text in ("how are you?", "How's it going", "hi, how are you doing?"):
            reply = self.ask(text)
            self.assertEqual(reply["intent"], "smalltalk.how_are_you", text)
            self.assertIn("thanks for asking", reply["text"])

    def test_thanks_and_goodbye(self):
        for text in ("thanks", "Thank you so much!", "thx", "appreciate it"):
            self.assertEqual(self.ask(text)["intent"], "smalltalk.thanks", text)
        for text in ("bye", "Goodbye!", "see you later", "that's all"):
            reply = self.ask(text)
            self.assertEqual(reply["intent"], "smalltalk.bye", text)
            self.assertIn(PHONE, reply["text"])

    def test_who_are_you_is_honest(self):
        for text in ("who are you?", "are you a bot", "Are you AI?", "are you a real person",
                     "is this a bot?", "am I talking to a human"):
            reply = self.ask(text)
            self.assertEqual(reply["intent"], "smalltalk.who", text)
            # AVATAR: the website helper is "not a person, and not AI"; the avatar is an AI avatar
            self.assertIn("AI Agent, not a person", reply["text"])
            self.assertNotIn("not AI", reply["text"])
            self.assertIn("tel:+18322950852", [link["path"] for link in reply["links"]])

    def test_asking_for_a_human(self):
        for text in ("can I talk to a real person", "human please", "speak to someone",
                     "I want to talk to Craig"):
            reply = self.ask(text)
            self.assertEqual(reply["intent"], "smalltalk.human", text)
            self.assertIn(PHONE, reply["text"])
            self.assertIn(EMAIL, reply["text"])
            self.assertIn("/contact-us/", [link["path"] for link in reply["links"]])

    def test_a_complaint_gets_an_apology_and_the_ways_to_reach_us(self):
        for text in ("The DJ was late and it ruined our night", "I'm really disappointed",
                     "nobody called me back, terrible"):
            reply = self.ask(text)
            self.assertEqual(reply["intent"], "smalltalk.complaint", text)
            self.assertIn("sorry", reply["text"].lower())
            self.assertIn(PHONE, reply["text"])

    def test_profanity_stays_polite_and_never_repeats_the_word(self):
        reply = self.ask("fuck this shit")
        self.assertEqual(reply["intent"], "smalltalk.profanity")
        self.assertNotIn("fuck", reply["text"].lower())
        self.assertNotIn("shit", reply["text"].lower())
        self.assertTrue(reply["chips"])
        # A real question with a swear word in it still gets its answer.
        reply = self.ask("how much is a damn foam party")
        self.assertEqual(reply["intent"], "prices.foam")
        self.assertNotIn("damn", reply["text"].lower())

    def test_off_topic(self):
        for text in ("what's the weather tomorrow", "tell me a joke", "who should I vote for",
                     "help with my math homework", "should I buy bitcoin"):
            reply = self.ask(text)
            self.assertEqual(reply["intent"], "smalltalk.offtopic", text)
            self.assertIn("I can only help with Houston Party Network events and nights out",
                          reply["text"])
            self.assertTrue(reply["chips"])

    def test_an_outdoor_party_in_the_rain_is_not_off_topic(self):
        self.assertEqual(self.ask("what if it rains at my outdoor party?")["intent"],
                         "topic:indoor-outdoor")

    def test_empty_emoji_and_gibberish(self):
        for text in ("", "   ", "😀🎉", "?!"):
            reply = self.ask(text)
            self.assertEqual(reply["intent"], "smalltalk.empty", repr(text))
            self.assertTrue(reply["chips"])
        reply = self.ask("asdfghjkl")
        self.assertEqual(reply["intent"], "smalltalk.gibberish")
        self.assertFalse(reply["confident"])

    def test_a_very_long_message_is_answered_from_its_start(self):
        reply = self.ask("Is the foam safe for kids? " + "blah " * 600)
        self.assertEqual(reply["intent"], "topic:foam-safe")

    def test_acknowledgements(self):
        for text in ("ok", "cool", "sounds good"):
            self.assertEqual(self.ask(text)["intent"], "smalltalk.ack", text)


class ExceptionalCaseTests(HelperTestCase):
    def test_a_card_number_is_never_repeated(self):
        reply = self.ask("my card is 4111 1111 1111 1111 exp 12/28")
        self.assertEqual(reply["intent"], "smalltalk.card")
        self.assertNotIn("4111", reply["text"])
        # AVATAR: said for voice as well as typing ("don't type card numbers" on the website)
        self.assertIn("don't share card numbers", reply["text"])

    def test_availability_for_a_date(self):
        for text in ("Are you available on June 14, 2027?", "is my date available",
                     "are you free on 6/14 for a wedding", "do you have availability next Saturday"):
            reply = self.ask(text)
            self.assertEqual(reply["intent"], "smalltalk.availability", text)
            self.assertIn("we'll confirm your date", reply["text"])
        wedding = self.ask("Are you available for our wedding on May 3?")
        self.assertIn("/contact-us/?event=wedding", [link["path"] for link in wedding["links"]])

    def test_money_and_terms_nobody_has_written_are_never_made_up(self):
        for text, intent in (("do you offer any discounts?", "smalltalk.policy_discount"),
                             ("how much is the deposit", "smalltalk.policy_deposit"),
                             ("what's your cancellation policy", "smalltalk.policy_cancel"),
                             ("can I get a refund", "smalltalk.policy_cancel")):
            reply = self.ask(text)
            self.assertEqual(reply["intent"], intent, text)
            self.assertIn("the team will confirm", reply["text"])
            self.assertFalse(reply["confident"])     # so Craig sees it in the CMS
            self.assertNotIn("$", reply["text"])

    def test_insurance_and_power_come_from_their_topics(self):
        self.assertEqual(self.ask("do you have liability insurance?")["intent"], "topic:insurance")
        self.assertEqual(self.ask("do you need a power outlet?")["intent"], "topic:venue-needs")
        self.assertEqual(self.ask("is there a contract")["intent"], "topic:contract")

    def test_booking_directly_goes_to_the_planner(self):
        reply = self.ask("can I book right now?")
        self.assertEqual(reply["intent"], "topic:book-directly")
        self.assertIn("/plan-my-event/", [link["path"] for link in reply["links"]])

    def test_every_reply_ends_with_somewhere_to_go(self):
        for text in ("hi", "who are you", "the dj was rude", "asdf", "llamas?", "weather",
                     "how much is a wedding", "", "4111111111111111"):
            reply = self.ask(text)
            self.assertTrue(reply["links"] or reply["chips"], text)

    def test_fallback_offers_the_closest_topics_and_contact(self):
        reply = self.ask("can the foam be pink for a gender reveal?")
        if reply["intent"] == "fallback":
            self.assertEqual(len(reply["chips"]), 3)
        self.assertEqual(self.ask("do you rent llamas")["intent"], "fallback")
        reply = self.ask("do you rent llamas for petting zoos")
        self.assertEqual(reply["intent"], "fallback")
        self.assertEqual(reply["kind"], "fallback")
        self.assertFalse(reply["confident"])
        self.assertIn("I'm not sure about that one yet", reply["text"])
        self.assertEqual(len(reply["chips"]), 3)
        paths = [link["path"] for link in reply["links"]]
        self.assertIn("/contact-us/", paths)
        self.assertTrue(any(p.startswith("tel:") for p in paths))
        self.assertTrue(any(p.startswith("mailto:") for p in paths))

# ---------------------------------------------------------------- from HPN test_live.py

class PriceTests(HelperTestCase):
    def catalog(self):
        # AVATAR: the website's /api/app/v1/plan/catalog/, as scripts/hpn_refresh.py saved it
        items = json.loads((KNOWLEDGE / "catalogue.json").read_text(encoding="utf-8"))["items"]
        return {"packages": [i for i in items if i["kind"] == "package"],
                "add_ons": [i for i in items if i["kind"] == "addon"]}

    def expected(self, event_type):
        return [(p["price_display"], p["hourly_display"] or p["tagline"] or p["summary"])
                for p in self.catalog()["packages"]
                if event_type in p["suits_event_types"]]

    def cards(self, reply):
        return [(c["subtitle"], c["detail"]) for c in reply["cards"]]

    def test_each_family_is_quoted_exactly_as_the_catalogue_api_says(self):
        for question, event_type, intent in (
                ("How much is a wedding?", "wedding", "prices.wedding"),
                ("what do your party packages cost", "party", "prices.party"),
                ("birthday party prices", "party", "prices.party"),
                ("foam party price", "foam", "prices.foam"),
                ("how much for a corporate event", "corporate", "prices.corporate"),
                ("quote for a school dance", "school", "prices.school")):
            reply = self.ask(question)
            self.assertEqual(reply["intent"], intent, question)
            self.assertEqual(reply["kind"], "live")
            self.assertEqual(self.cards(reply), self.expected(event_type), question)
            self.assertTrue(all(c["subtitle"].startswith("Starting at") for c in reply["cards"]))

    def test_corporate_and_school_get_the_party_packages(self):
        party = self.cards(self.ask("party prices"))
        self.assertEqual(self.cards(self.ask("corporate event price")), party)
        self.assertEqual(self.cards(self.ask("how much for a prom")), party)

    def test_a_price_change_in_the_cms_is_quoted_at_once(self):
        # AVATAR: a refresh with a new price (the catalogue's own display string, as the website writes it)
        for o in self.kb.offerings:
            if o.slug == "party-essentials":
                o.price_display = "Starting at $599 for 3 hours"
        self.rebuild(offerings=self.kb.offerings)
        reply = self.ask("party prices")
        self.assertIn("Starting at $599 for 3 hours", [c["subtitle"] for c in reply["cards"]])

    def test_weddings_name_craig_and_only_the_two_wedding_add_ons(self):
        reply = self.ask("wedding packages price")
        self.assertIn("Craig plans each wedding personally", reply["text"])
        self.assertIn("Dancing on a Cloud", reply["text"])
        self.assertIn("LED Party Sticks", reply["text"])
        self.assertNotIn("Photo Booth (", reply["text"])
        self.assertIn("/contact-us/?event=wedding", [link["path"] for link in reply["links"]])

    def test_outdoor_movie_night_is_never_offered(self):
        # Even switched ON by mistake, it stays out of every answer while the page is hidden.
        # AVATAR: an outdoor movie night in the catalogue by mistake (it is not in the live one)
        movie = Offering(slug="outdoor-movie-night", kind="addon", name="Outdoor Movie Night", price_usd=None,
                         price_display="Starting at $400", summary="A movie under the stars.")
        self.rebuild(offerings=self.kb.offerings + [movie])
        for question in ("what add-ons do you have", "party prices", "prices", "extra hours",
                         "foam party add-ons", "add-ons for a birthday"):
            reply = self.ask(question)
            blob = str(reply).lower()
            self.assertNotIn("outdoor movie night", blob, question)
            self.assertNotIn("/outdoor-movies/", blob, question)
        # Asked for by name, the answer is that it isn't offered: no price, no card, no link.
        reply = self.ask("how much is an outdoor movie night")
        self.assertEqual(reply["intent"], "topic:outdoor-movies")
        self.assertIn("aren't something we're offering", reply["text"])
        self.assertEqual(reply["cards"], [])
        self.assertNotIn("/outdoor-movies/", str(reply))
        self.assertNotIn("$", reply["text"])

    def test_overview_extra_hours_and_add_ons(self):
        reply = self.ask("prices")
        self.assertEqual(reply["intent"], "prices.overview")
        self.assertEqual(len(reply["cards"]), 3)
        hours = self.ask("how much is an extra hour?")
        self.assertEqual(hours["intent"], "prices.extra_hours")
        details = {c["title"]: c["detail"] for c in hours["cards"]}
        catalog = {p["name"]: p["hourly_display"] for p in self.catalog()["packages"]
                   if p["hourly_display"]}
        self.assertEqual(details, catalog)
        self.assertIn("quoted by Craig", hours["text"])
        addons = self.ask("what extras do you have?")
        self.assertEqual(addons["intent"], "prices.addons")
        self.assertEqual({c["title"]: c["subtitle"] for c in addons["cards"]},
                         {a["name"]: a["price_display"] for a in self.catalog()["add_ons"]
                          if a["slug"] not in engine.HIDDEN_OFFERINGS})

    def test_one_add_on(self):
        reply = self.ask("how much is a photo booth?")
        self.assertEqual(reply["intent"], "prices.addon")
        self.assertEqual(reply["cards"][0]["title"], "Photo Booth")
        self.assertEqual(reply["cards"][0]["subtitle"], "Starting at $275")
        karaoke = self.ask("how much is karaoke for my party")
        self.assertEqual(karaoke["cards"][0]["title"], "Karaoke")
        self.assertIn("a DJ comes with it", karaoke["text"])
        bingo = self.ask("price of jukebox bingo for a private party")
        self.assertEqual(bingo["cards"][0]["subtitle"], "Price on request")

    def test_a_package_by_name(self):
        reply = self.ask("What does Signature Wedding include?")
        self.assertEqual(reply["intent"], "prices.package")
        self.assertEqual(reply["cards"][0]["subtitle"], "Starting at $1,495")
        self.assertIn("Everything in Reception Essentials, plus:", reply["text"])
        self.assertIn("separate ceremony sound system", reply["text"])
        self.assertIn("DJ assistant/event technician", reply["text"])
        headliner = self.ask("tell me about headliner party")
        self.assertIn("Choose two:", headliner["text"])

    def test_how_much_after_a_wedding_answer_means_the_wedding(self):
        self.assertEqual(self.ask("how much?", last="topic:weddings")["intent"], "prices.wedding")


class NightTests(HelperTestCase):
    """THURSDAY_8PM: Molly's Pub Tomball has karaoke at 9 tonight."""

    def test_tonight(self):
        reply = self.ask("what's on tonight?")
        self.assertEqual(reply["intent"], "nights.tonight")
        self.assertEqual([c["title"] for c in reply["cards"]], ["Molly's Pub Tomball"])
        self.assertEqual(reply["cards"][0]["subtitle"], "Karaoke · 9 PM")
        self.assertEqual(reply["cards"][0]["detail"], "Tomball · tonight")
        self.assertIn("/night-out/", [link["path"] for link in reply["links"]])

    def test_tomorrow_and_a_weekday(self):
        reply = self.ask("anything happening tomorrow")
        self.assertEqual(reply["intent"], "nights.tomorrow")
        self.assertEqual([c["title"] for c in reply["cards"]], ["Franklin's Tower"])
        self.assertEqual(reply["cards"][0]["subtitle"], "Jukebox Bingo · 7 PM")
        reply = self.ask("karaoke on saturday")
        self.assertEqual(reply["intent"], "nights.day")
        self.assertEqual(sorted(c["title"] for c in reply["cards"]),
                         ["Molly's Pub Tomball", "Tony's Tavern"])

    def test_this_weekend(self):
        reply = self.ask("what's on this weekend?")
        self.assertEqual(reply["intent"], "nights.weekend")
        titles = [(c["title"], c["subtitle"]) for c in reply["cards"]]
        self.assertIn(("Franklin's Tower", "Jukebox Bingo · 7 PM"), titles)
        self.assertIn(("Tony's Tavern", "DJ Party · 8 PM"), titles)
        self.assertNotIn(("Mongo's", "Jukebox Bingo · 7 PM"), titles)   # a Wednesday

    def test_by_town_and_kind(self):
        reply = self.ask("bingo in Montgomery")
        self.assertEqual([c["title"] for c in reply["cards"]], ["Mongo's"])
        reply = self.ask("karaoke nights in houston")
        self.assertEqual({c["title"] for c in reply["cards"]}, {"Tony's Tavern"})
        self.assertTrue(all(c["subtitle"].startswith("Karaoke") for c in reply["cards"]))
        reply = self.ask("where can I sing karaoke in Tomball")
        self.assertEqual({c["title"] for c in reply["cards"]}, {"Molly's Pub Tomball"})

    def test_nothing_then_offers_the_next_ones(self):
        reply = self.ask("bingo tonight")
        self.assertEqual(reply["intent"], "nights.tonight")
        self.assertIn("I can't see any Jukebox Bingo nights tonight", reply["text"])
        self.assertEqual(reply["cards"][0]["title"], "Franklin's Tower")

    def test_a_follow_up_keeps_the_subject(self):
        self.assertEqual(self.ask("and tomorrow?", last="nights.tonight")["intent"],
                         "nights.tomorrow")

    def test_a_switched_off_venue_is_not_offered(self):
        # AVATAR: a venue switched off is not on the live schedule the refresh reads
        self.rebuild(slots=[s for s in self.kb.slots if s.venue.name != "Molly's Pub Tomball"])
        reply = self.ask("what's on tonight?")
        self.assertNotIn("Molly's Pub Tomball", str(reply))


class DjTests(HelperTestCase):
    # AVATAR: the website reads HPN's private DJ rota (who is on a gig now); the avatar has no access to it,
    # so it never says who is - or is not - on, and points to the Request a song page instead.
    def test_who_is_djing_points_to_the_request_page_without_naming_anyone(self):
        for question in ("who's DJing right now?", "request a song", "who is the dj"):
            reply = self.ask(question, now=None)
            self.assertEqual(reply["intent"], "djs.now", question)
            self.assertEqual(reply["cards"], [], question)
            self.assertIn("I can't see which DJs are on right now", reply["text"])
            self.assertNotIn("No DJ is on", reply["text"])
            self.assertIn("QR code at the DJ booth", reply["text"])
            self.assertIn("follows the venue schedule", reply["text"])
            self.assertIn("/request/", [link["path"] for link in reply["links"]])


class ContactAndTownTests(HelperTestCase):
    def test_contact_details_come_from_settings_like_the_app_meta(self):
        meta = self.kb.business          # AVATAR: the export's business details (the app's meta)
        reply = self.ask("what's your phone number?")
        self.assertEqual(reply["intent"], "contact")
        self.assertIn(meta["phone_display"], reply["text"])
        self.assertIn(meta["email"], reply["text"])
        paths = [link["path"] for link in reply["links"]]
        self.assertIn(f"tel:{meta['phone_e164']}", paths)
        self.assertIn(f"mailto:{meta['email']}", paths)
        self.assertEqual(self.ask("how can I contact you")["intent"], "contact")

    def test_a_number_changed_in_the_cms_follows(self):
        # AVATAR: a refreshed export with another number
        self.rebuild(business={**self.kb.business, "phone_display": "(713) 555-0100", "phone_e164": "+17135550100"})
        reply = self.ask("phone number")
        self.assertIn("(713) 555-0100", reply["text"])
        self.assertIn("tel:+17135550100", [link["path"] for link in reply["links"]])

    def test_towns_served(self):
        reply = self.ask("what areas do you serve?")
        self.assertEqual(reply["intent"], "towns")
        self.assertIn(self.kb.counts["town_list"], reply["text"])
        reply = self.ask("do you serve Conroe?")
        self.assertEqual(reply["intent"], "towns")
        self.assertIn("Yes, Conroe is one of the towns we serve", reply["text"])

# ---------------------------------------------------------------- from HPN test_knowledge_base.py

# The website's pages the helper links to (HPN: each checked live, GET 200, on 29 Oct 2026) and its Django
# pages (song requests, the portal, account deletion, sign-in).
SITE_PAGES = {
    "/plan-my-event/", "/night-out/", "/contact-us/", "/weddings/", "/parties-and-events/",
    "/foam-parties/", "/karaoke/", "/jukebox-bingo/", "/for-bars-and-venues/", "/story/",
    "/about-us/", "/terms-and-privacy/", "/planning-a-birthday/", "/planning-a-private-party/",
    "/planning-a-corporate-event/", "/planning-a-school-or-community-event/",
    "/request/", "/portal/", "/accounts/delete/", "/accounts/login/", "/accounts/signup/",
    "/accounts/password/reset/", "/accounts/data/",
}
NEVER = {"/outdoor-movies/", "/whats-on/"}


def link_exists(path):
    if path.startswith(("tel:", "mailto:")):
        return True
    bare = path.split("?")[0]
    return bare not in NEVER and bare in SITE_PAGES


class KnowledgeBaseTests(HelperTestCase):
    # AVATAR: the knowledge is HPN's export of its live database (knowledge/hpn), not the data migration

    def test_the_export_has_80_to_120_topics_each_with_questions_and_an_answer(self):
        topics = self.kb.topics
        self.assertGreaterEqual(len(topics), 80)
        self.assertLessEqual(len(topics), 120)
        self.assertEqual(len({t.slug for t in topics}), len(topics))
        for topic in topics:
            self.assertTrue(topic.answer_template.strip(), topic.slug)
            self.assertGreaterEqual(len(topic.question_lines), 3, topic.slug)

    def test_the_djs_team_topics_are_left_out(self):
        self.assertNotIn("team", {t.category for t in self.kb.topics})
        self.assertEqual(self.kb.excluded_categories, ("team",))

    def test_every_link_is_a_real_page(self):
        for topic in self.kb.topics:
            for link in topic.links:
                self.assertTrue(link_exists(link["path"]), f"{topic.slug}: {link['path']}")
        for _, path in engine.PAGES.values():
            self.assertTrue(link_exists(path), path)

    def test_no_topic_states_a_price_or_names_hidden_things(self):
        for topic in self.kb.topics:
            answer = self.kb.fill(topic.answer_template)
            self.assertNotIn("$", answer, topic.slug)
            self.assertNotIn("HPN", answer.replace("HPN's", ""), topic.slug)
            self.assertNotIn("/outdoor-movies/", str(topic.links), topic.slug)

    def test_tokens_are_filled_in(self):
        reply = self.ask("Which areas do you serve?")
        self.assertIn(self.kb.counts["town_list"], reply["text"])
        for topic in self.kb.topics:
            self.assertNotRegex(self.kb.fill(topic.answer_template), r"\{[a-z_]+\}", topic.slug)

    def test_every_suggested_question_gets_a_real_answer(self):
        chips = set(engine.STARTER_CHIPS)
        for topic in self.kb.topics:
            chips.update(topic.chips)
        for question in ("party prices", "wedding prices", "what's on tonight?", "prices",
                         "how much is a photo booth", "request a song", "who are you",
                         "do you offer discounts", "llamas please"):
            chips.update(self.ask(question)["chips"])
        for chip in sorted(chips):
            reply = self.ask(chip)
            self.assertNotIn(reply["intent"], ("fallback", "smalltalk.gibberish"), chip)

    def test_every_topic_answers_its_own_title_and_examples(self):
        for topic in self.kb.topics:
            for question in [topic.title] + topic.question_lines:
                reply = self.ask(question)
                self.assertNotEqual(reply["intent"], "fallback", f"{topic.slug}: {question}")

    def test_a_new_topic_answers_at_once(self):
        parking = Topic(slug="parking", title="Parking", category="other",
                        answer_template="Ask your venue about parking.",
                        question_lines=["Is there parking?", "Where do we park?"],
                        keyword_list=["parking", "park", "car"],
                        links=[{"label": "Contact us", "path": "/contact-us/"}], chips=[])
        self.rebuild(topics=self.kb.topics + [parking])
        self.assertEqual(self.intent("where can our guests park?"), "topic:parking")
        self.rebuild(topics=[t for t in self.kb.topics if t.slug != "parking"])
        self.assertNotEqual(self.intent("where can our guests park?"), "topic:parking")


# (what a visitor types, the intent that must answer it)
VISITORS = [
    ("Plan an event", "topic:plan-my-event"),
    ("How do I book you guys?", "topic:how-booking-works"),
    ("I sent an enquiry yesterday, what happens next?", "topic:after-enquiry"),
    ("how far in advance do we need to book", "topic:book-ahead"),
    ("can you handle 300 guests", "topic:guest-count"),
    ("where do I put my must play songs", "topic:planning-form"),
    ("how do I pay", "topic:payment"),
    ("do you take credit cards?", "topic:payment"),
    ("what does starting at mean", "topic:starting-prices"),
    ("is there a travel fee", "topic:travel"),
    ("what are the headliner party enhancements", "topic:headliner-choice"),
    ("Do you do weddings?", "topic:weddings"),
    ("can you do our ceremony and the reception?", "topic:wedding-ceremony"),
    ("how long do you play at a wedding", "topic:wedding-hours"),
    ("can we have a photo booth at our wedding", "topic:wedding-extras"),
    ("will you make the announcements", "topic:emcee"),
    ("can we choose our own music?", "topic:music-choice"),
    ("Do you do kids parties?", "topic:kids-parties"),
    ("can you set up in our backyard", "topic:party-at-home"),
    ("We don't need a DJ, just speakers", "topic:just-sound"),
    ("do you invoice companies", "topic:invoice-company"),
    ("can you keep the music clean for the kids", "topic:clean-music"),
    ("prom DJ", "topic:school-events"),
    ("are you insured", "topic:insurance"),
    ("is setup included", "topic:setup-included"),
    ("Is the foam safe for my toddler?", "topic:foam-safe"),
    ("does the foam make a mess", "topic:foam-cleanup"),
    ("what ages are foam parties good for", "topic:foam-ages"),
    ("can you do foam indoors", "topic:foam-indoors"),
    ("is there music with the foam party", "topic:foam-music"),
    ("glow foam party at night?", "topic:foam-glow"),
    ("Can we have karaoke at our event?", "topic:karaoke-private"),
    ("what songs do you have for karaoke", "topic:karaoke-songs"),
    ("what is jukebox bingo", "topic:jukebox-bingo"),
    ("what are the bingo themes", "topic:bingo-themes"),
    ("how do song requests work", "topic:song-requests-how"),
    ("can I tip the DJ with venmo", "topic:tip-dj"),
    ("where did my request go", "topic:requests-reset"),
    ("why wasn't my song played", "topic:request-not-played"),
    ("who started Houston Party Network", "topic:story"),
    ("are you hiring DJs", "topic:jobs"),
    ("I own a bar, can you run karaoke there?", "topic:bars-venues"),
    ("how do I delete my account", "topic:delete-account"),
    ("do you sell my data", "topic:privacy"),
    ("is there an app", "topic:app"),
    # AVATAR: ("how do I sign in to the team app", "topic:team-app") - the team app topics are left out
    ("do you have cold sparks", "topic:cold-sparks"),
    ("do you do outdoor movies", "topic:outdoor-movies"),
    ("How much is a wedding?", "prices.wedding"),
    ("how much is a birthday party", "prices.party"),
    ("foam party cost", "prices.foam"),
    ("what are your prices", "prices.overview"),
    ("how much is the photo booth", "prices.addon"),
    ("how much is an extra hour", "prices.extra_hours"),
    ("what's included in foam frenzy", "prices.package"),
    ("what's on tonight", "nights.tonight"),
    ("karaoke this weekend", "nights.weekend"),
    ("bingo in spring", "nights.week"),
    ("who is the dj tonight", "djs.now"),
    ("what's your email", "contact"),
    ("do you come to Magnolia", "towns"),
    ("are you available on the 14th of June", "smalltalk.availability"),
    ("hello", "greeting"),
    ("are you a robot", "smalltalk.who"),
]


class VisitorQuestionTests(HelperTestCase):
    def test_realistic_questions(self):
        self.assertGreaterEqual(len(VISITORS), 40)
        wrong = []
        for question, intent in VISITORS:
            got = self.ask(question)["intent"]
            if got != intent:
                wrong.append(f"{question!r}: wanted {intent}, got {got}")
        self.assertEqual(wrong, [])
