# Copied unchanged from the HPN project (Peony0077/HPN @ 045de53, backend/apps/helper/text.py): the
# Event helper's own rules for reading a question, so the avatar matches it exactly as the website does.
# Do not edit here: change it in HPN and copy it again (tests/test_hpn_engine.py checks the routing).
"""
Reading what a visitor typed: the text made comparable, and the ideas found in it.

Pure Python, no Django, so every rule here is quick to test and has no hidden inputs. The
helper is deterministic on purpose: the same question gets the same answer, and the reason
for an answer can be read out of this file.

normalise() makes "What's the PRICE of a Wedding?!" and "whats the price of a wedding" the
same string. concepts() then finds the ideas in it -- #price, #wedding, #tonight -- from
lists of words and phrases that mean the same thing to a visitor (price, cost, how much,
rate, quote...). The words themselves are kept too: "bar" is a #where to the schedule, but
"I own a bar" still needs the word to reach the topic for venue owners.
"""
import re
import unicodedata

# What is compared. A long message is answered from its start (the spec: first 500 chars),
# which is where people put the question.
MATCH_CHARS = 500

CONTRACTIONS = {
    "what's": "what is", "whats": "what is", "who's": "who is", "whos": "who is",
    "where's": "where is", "wheres": "where is", "when's": "when is", "how's": "how is",
    "it's": "it is", "that's": "that is", "thats": "that is", "there's": "there is",
    "here's": "here is", "let's": "let us", "i'm": "i am", "im": "i am", "i've": "i have",
    "i'd": "i would", "i'll": "i will", "you're": "you are", "youre": "you are",
    "you've": "you have", "you'll": "you will", "you'd": "you would", "we're": "we are",
    "we've": "we have", "we'll": "we will", "they're": "they are", "isn't": "is not",
    "isnt": "is not", "aren't": "are not", "arent": "are not", "don't": "do not",
    "dont": "do not", "doesn't": "does not", "doesnt": "does not", "didn't": "did not",
    "didnt": "did not", "can't": "can not", "cant": "can not", "cannot": "can not",
    "won't": "will not", "wont": "will not", "wouldn't": "would not", "couldn't": "could not",
    "shouldn't": "should not", "haven't": "have not", "hasn't": "has not", "wasn't": "was not",
    "weren't": "were not", "y'all": "you all", "yall": "you all", "gonna": "going to",
    "wanna": "want to", "gotta": "got to", "u": "you", "ur": "your", "r": "are", "pls": "please",
    "plz": "please", "thx": "thanks", "ty": "thank you", "b-day": "birthday", "bday": "birthday",
    "tonite": "tonight", "tmrw": "tomorrow", "tmr": "tomorrow", "2nite": "tonight",
}


def normalise(text):
    """Lowercase, accents and punctuation gone, contractions spelled out, one space between
    words, at most MATCH_CHARS characters of the original considered."""
    text = (text or "")[:MATCH_CHARS].lower()
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    # "café" -> "cafe", "adiós" -> "adios": accents are not meaning.
    text = "".join(ch for ch in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(ch))
    text = text.replace("ß", "ss")
    words = []
    for word in re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)*", text):
        expanded = CONTRACTIONS.get(word)
        if expanded is None and "'" in word:
            # "craig's" -> "craig", "djs'" handled by the split above.
            expanded = CONTRACTIONS.get(word.replace("'", ""), word.split("'")[0])
        words.append(expanded or word)
    text = " ".join(" ".join(words).replace("-", " ").split())
    for before, after in (("bar mitzvahs", "barmitzvahs"), ("bat mitzvahs", "batmitzvahs"),
                          ("bar mitzvah", "barmitzvah"), ("bat mitzvah", "batmitzvah")):
        text = f" {text} ".replace(f" {before} ", f" {after} ").strip()
    # The company's own name is not a town ("Houston") or a party: one token, a stopword.
    for name in ("houston party network", "h town party network", "htown party network",
                 "h town party"):
        text = f" {text} ".replace(f" {name} ", " hpn ").strip()
    return text


def has_phrase(norm, phrase):
    """Whole-word phrase match in a normalised string."""
    return f" {phrase} " in f" {norm} "


# --- ideas a visitor can mean in many words ---------------------------------------------------
#
# Each entry: concept -> words and phrases (already in normalised form). Matching is whole
# words, so "rate" does not fire inside "celebrate".
CONCEPTS = {
    "price": ["price", "prices", "pricing", "priced", "cost", "costs", "how much",
              "rate", "rates", "fee", "fees", "charge", "charges", "charging", "quote",
              "quotes", "expensive", "cheap", "cheapest", "affordable", "budget", "dollars",
              "precio", "precios", "cuanto", "cuanto cuesta", "cuesta"],
    "wedding": ["wedding", "weddings", "bride", "brides", "groom", "reception", "ceremony",
                "married", "marry", "marrying", "engaged", "boda", "bodas", "nupcial",
                # Common misspellings people really type.
                "weding", "wedings", "weddng", "wedidng", "wedding s", "bridal"],
    "foam": ["foam", "bubbles", "bubble", "espuma"],
    "karaoke": ["karaoke", "sing", "singing", "singer", "mic", "karoke", "karaoki"],
    "bingo": ["bingo", "jukebox", "music bingo"],
    "dj": ["dj", "djs", "deejay", "deejays", "disc jockey", "djing", "music", "dj party"],
    "tonight": ["tonight", "today", "this evening", "this afternoon", "right now",
                "esta noche", "hoy"],
    "tomorrow": ["tomorrow", "manana"],
    "weekend": ["weekend", "this weekend", "the weekend", "weekends"],
    "this_week": ["this week", "next few days", "next 7 days", "coming days"],
    "book": ["book", "booking", "books", "booked", "reserve", "reserving", "reservation",
             "hire", "rent", "rental", "sign up for", "reservar"],
    "request": ["request", "requests", "song request", "play my song", "play a song",
                "my song", "song", "songs", "requesting"],
    "contact": ["phone", "call", "calling", "phone number", "email", "e mail",
                "contact", "reach", "text you", "get in touch", "telephone", "address",
                "your number", "contact number", "telefono", "correo"],
    "where": ["where", "location", "locations", "located", "venue", "venues", "bar", "bars",
              "pub", "near me", "nearby", "close to", "donde"],
    "party": ["party", "parties", "birthday", "birthdays", "celebration", "anniversary",
              "retirement", "reunion", "private party", "private event", "fiesta",
              "cumpleanos", "quinceanera", "housewarming", "engagement party", "event",
              "events", "quinceaneras", "quince", "sweet 16", "sweet sixteen", "barmitzvah",
              "batmitzvah", "barmitzvahs", "batmitzvahs", "mitzvah"],
    "corporate": ["corporate", "company", "work party", "holiday party", "office party",
                  "staff party", "team building", "business", "christmas party", "employees",
                  "awards", "client event", "work event"],
    "school": ["school", "schools", "prom", "homecoming", "graduation", "graduations",
               "school dance", "fundraiser", "fundraisers", "church", "community", "daycare",
               "field day", "pta", "students", "festival"],
    "kids": ["kids", "kid", "children", "child", "toddler", "toddlers", "teens", "teenagers",
             "ninos"],
    "extra_time": ["extra hour", "extra hours", "additional hour", "additional hours",
                   "another hour", "more time", "extra time", "additional time", "overtime",
                   "longer", "each hour", "per hour", "hourly", "go over"],
    "addons": ["add on", "add ons", "addon", "addons", "extras", "extra", "enhancement",
               "enhancements", "upgrades", "upgrade"],
    "package": ["package", "packages", "paquete", "paquetes"],
    "what_on": ["what is on", "anything on", "going on", "happening", "what to do",
                "night out", "nights out", "things to do", "go out", "open mic", "schedule",
                "calendar"],
    "areas": ["serve", "service area", "areas", "travel", "come to", "towns", "cities",
              "city", "outside houston", "your area", "my area"],
    # Words that mean "can you do my date", which the helper cannot see.
    "availability": ["available", "availability", "availible", "avail", "booked up",
                     "fully booked", "free on", "open on", "still open", "check my date",
                     "check the date", "check a date", "have my date", "date open",
                     "disponible"],
    "policy_discount": ["discount", "discounts", "coupon", "promo", "promo code", "deal",
                        "deals", "military discount", "special offer", "cheaper"],
    "policy_deposit": ["deposit", "deposits", "down payment", "retainer", "upfront"],
    "policy_contract": ["contract", "contracts", "agreement", "terms of booking",
                        "sign something"],
    "policy_cancel": ["cancel", "cancellation", "cancelling", "canceling", "cancelled",
                      "refund", "refunds", "reschedule", "rescheduling", "postpone",
                      "change the date", "move the date", "rain date"],
    "policy_insurance": ["insurance", "insured", "liability", "certificate of insurance",
                         "coi"],
    "policy_power": ["power", "outlet", "outlets", "electricity", "electric", "generator",
                     "extension cord", "space", "room size", "how much space", "setup space",
                     "set up space"],
    "policy_cover": ["cover charge", "cover charges", "a cover", "no cover", "entry fee",
                     "entrance fee", "admission", "door charge", "free to play", "free entry",
                     "free to attend", "free to get in", "cost to get in", "pay to get in"],
    "policy_age": ["age limit", "age limits", "age restriction", "age requirement",
                   "minimum age", "minors", "underage", "under age", "all ages", "over 21",
                   "under 21", "be 21", "21 and over", "21 and up", "over 18", "under 18",
                   "be 18", "18 and over", "can kids come", "can children come", "bring my kids",
                   "bring kids", "bring children", "bring my children", "kids allowed",
                   "children allowed", "how old do", "how old must", "how old to"],
    "policy_permit": ["permit", "permits", "license", "licence", "licensed", "hoa"],
    "policy_conduct": ["drink", "drinking", "alcohol", "dj meal", "vendor meal", "feed the dj",
                       "need a meal", "meals for", "take breaks", "take a break", "dj breaks",
                       "dress code", "what will the dj wear", "what do djs wear",
                       "what does the dj wear", "attire", "smoke", "smoking"],
    "policy_genre": ["tejano", "cumbia", "reggaeton", "bachata", "salsa music", "spanish music",
                     "music in spanish", "latin music", "country music", "hip hop", "rap",
                     "edm", "genre", "genres", "what kind of music", "what type of music",
                     "what music do you play", "musica"],
    "policy_arrival": ["when do you arrive", "what time do you arrive", "what time will you arrive",
                       "when will you arrive", "how early", "arrival time", "arrive",
                       "load in time"],
    # Health questions the foam page does not answer (eyes, asthma, pregnancy...).
    "policy_health": ["eye", "eyes", "asthma", "asthmatic", "pregnant", "pregnancy", "rash",
                      "breathing", "epilepsy", "epileptic", "seizure", "seizures",
                      "medical condition", "slippery", "slip", "slipping"],
    # Nobody has said a client can choose their DJ.
    "policy_dj_choice": ["pick my dj", "choose my dj", "pick the dj", "choose the dj",
                         "pick our dj", "choose our dj", "specific dj", "particular dj",
                         "same dj", "who will be my dj", "who will be our dj",
                         "which dj will", "request a dj", "book a specific dj"],
    "payment": ["pay", "payment", "payments", "paying", "invoice", "invoices", "card",
                "credit card", "venmo", "zelle", "cash app", "paypal", "cash"],
}

# Concepts that are a question about money or terms the helper must not make up an answer
# to. Answered by a topic only when the topic itself is about that thing.
POLICY_CONCEPTS = ("policy_discount", "policy_deposit", "policy_contract", "policy_cancel",
                   "policy_insurance", "policy_power", "policy_cover", "policy_age",
                   "policy_permit", "policy_conduct", "policy_genre", "policy_arrival",
                   "policy_health", "policy_dj_choice")

# Ideas that show the question is about Houston Party Network's business at all; an
# off-topic word ("weather") beside one of these is a real question ("what if the weather
# is bad for my foam party?").
DOMAIN_CONCEPTS = ("price", "wedding", "foam", "karaoke", "bingo", "dj", "book", "request",
                   "contact", "party", "corporate", "school", "kids", "extra_time", "addons",
                   "package", "what_on", "areas", "availability", "payment") + POLICY_CONCEPTS


def concepts(norm):
    """The set of concepts whose words or phrases appear in a normalised string."""
    padded = f" {norm} "
    return {name for name, phrases in CONCEPTS.items()
            if any(f" {p} " in padded for p in phrases)}


# --- small talk ---------------------------------------------------------------------------------

GREETINGS = ["good morning", "good afternoon", "good evening", "good day", "buenos dias",
             "buenas tardes", "buenas noches", "hi there", "hello there", "hey there",
             "what is up", "whats up", "wassup", "sup", "hello", "hi", "hey", "heya", "hiya",
             "howdy", "hola", "yo", "greetings", "morning", "evening", "afternoon",
             "good morning to you", "hey hey"]
# Words that may follow a greeting without being a question: "hi there team", "hey guys".
GREETING_FILLER = {"there", "team", "guys", "all", "everyone", "folks", "you", "friend",
                   "hpn", "houston", "party", "network", "helper", "again", "y", "and",
                   "so", "ok", "okay", "please", "um", "uh", "well", "yall", "sir", "maam",
                   "mam", "buddy", "bot"}
HOW_ARE_YOU = ["how are you", "how are you doing", "how is it going", "how you doing",
               "how are things", "how is your day", "how have you been", "you good",
               "how do you do", "como estas", "como esta", "que tal"]
THANKS = ["thanks", "thank you", "thank you so much", "thanks a lot", "many thanks",
          "appreciate it", "appreciate you", "much appreciated", "gracias", "muchas gracias",
          "perfect thanks", "great thanks", "awesome thanks"]
BYE = ["bye", "goodbye", "good bye", "bye bye", "see you", "see ya", "see you later",
       "later", "catch you later", "have a good night", "good night", "goodnight",
       "that is all", "that is it", "nothing else", "i am done", "adios", "hasta luego",
       "ciao", "take care"]
WHO = ["who are you", "what are you", "are you a bot", "are you a robot", "are you ai",
       "are you an ai", "are you artificial intelligence", "is this ai", "is this a bot",
       "is this a real person", "is this a person", "are you real", "are you human",
       "are you a human", "are you a person", "are you a real person", "am i talking to",
       "am i chatting with", "chatgpt", "chat gpt", "are you chatgpt", "are you a computer",
       "your name", "what is your name", "who am i talking to", "are you a machine",
       "is this automated", "are you automated", "what can you do", "how do you work",
       "what do you know", "eres un robot"]
HUMAN = ["talk to a human", "speak to a human", "talk to a person", "speak to a person",
         "talk to a real person", "speak to a real person", "talk to someone",
         "speak to someone", "speak with someone", "talk with someone", "real person please",
         "a human", "human please", "an agent", "live agent", "representative",
         "customer service", "talk to craig", "speak to craig", "speak with craig",
         "talk to the owner", "speak to the owner", "talk to a manager", "speak to a manager",
         "live person", "live agent", "real human", "actual person"]
# Asking to be called: nobody reads a number typed here in time, so the reply says how to
# get a call instead of "of course".
CALLBACK = ["call me", "call me back", "someone call me", "can you call me",
            "could you call me", "give me a call", "call back", "callback", "ring me",
            "text me", "have someone call", "have craig call", "contact me",
            "reach me", "reach out to me", "email me", "llamame"]
COMPLAINT = ["complaint", "complain", "complaining", "terrible", "awful", "horrible",
             "worst", "angry", "upset", "disappointed", "disappointing", "unhappy",
             "frustrated", "frustrating", "annoyed", "rude", "never showed", "no show",
             "did not show", "did not show up", "showed up late", "was late", "ruined",
             "bad experience", "not happy", "unacceptable", "useless", "not helpful",
             "unhelpful", "you are not helping", "not helping", "waste of time",
             "no one answered", "nobody answered", "nobody called", "no one called",
             "never called", "never heard back", "not responding", "no response",
             "poor service", "bad service", "scam", "ripped off", "you suck", "this sucks"]
# Never repeated back. Matched as whole words only, so "Scunthorpe" problems cannot arise.
PROFANITY = ["fuck", "fucking", "fucked", "fucker", "fck", "fk", "shit", "shitty", "bullshit",
             "bitch", "bitches", "asshole", "ass", "bastard", "dick", "dickhead", "cunt",
             "piss", "pissed", "damn", "goddamn", "crap", "wtf", "stfu", "motherfucker",
             "idiot", "idiots", "stupid", "dumb", "moron", "retard", "retarded", "loser",
             "jerk", "suck", "sucks", "puta", "mierda", "pendejo", "cabron"]
OFF_TOPIC = ["weather", "forecast", "temperature", "is it going to rain", "joke", "jokes",
             "tell me a joke", "funny story", "politics", "political", "president",
             "election", "vote", "voting", "trump", "biden", "harris", "democrat",
             "republican", "congress", "homework", "math", "algebra", "calculus", "equation",
             "essay", "write my", "write me", "poem", "story about", "crypto", "bitcoin",
             "ethereum", "stock", "stocks", "invest", "investing", "lottery", "recipe",
             "cook", "cooking", "sports", "football score", "who won", "score of the game",
             "capital of", "meaning of life", "news", "programming", "python",
             "javascript", "coding", "medical", "doctor", "lawyer", "legal advice",
             "dating", "girlfriend", "boyfriend", "horoscope", "religion", "god",
             "movie recommendation", "netflix", "video game", "minecraft", "fortnite",
             "chatgpt can you", "solve", "how old are you", "what time is it",
             "what is the time", "what day is it", "what is the date today",
             "what is today s date"]
SPANISH = ["hola", "precio", "precios", "boda", "bodas", "fiesta", "fiestas", "gracias",
           "adios", "cuanto", "cuesta", "quiero", "necesito", "para", "buenos dias",
           "buenas tardes", "buenas noches", "por favor", "donde", "cumpleanos", "esta noche",
           "quinceanera", "espanol", "hablas", "habla", "telefono", "como estas"]

# Common words that carry no topic on their own. Kept small: "party", "book" and "song"
# are NOT here, they are the whole question.
STOPWORDS = set("""
a an the and or but if then so to of in on at for from by with about as into than too very
is are was were be been being am do does did doing have has had having can could would should
will shall may might must i me my mine we us our you your yours he she it its they them their
this that these those there here what which who whom whose when where why how all any some
no not only just also really please thanks thank hi hello hey get got want wanted need needs
like know tell let make does much many more most other such own same one two up out over
again further once each few both s t now ok okay yes yeah yep sure well im ill id ive
anything something someone somebody thing things way going go would like give help hpn
houston network question questions ask asking info information
""".split())


def content_words(norm):
    """The words of a normalised string that are not stopwords, in order, once each."""
    seen = []
    for word in norm.split():
        if word not in STOPWORDS and word not in seen:
            seen.append(word)
    return seen


def any_phrase(norm, phrases):
    return any(has_phrase(norm, p) for p in phrases)


def strip_greeting(norm):
    """(greeted, rest): whether the text opens with a greeting, and what follows it."""
    greeted = False
    rest = norm
    changed = True
    while changed and rest:
        changed = False
        for phrase in sorted(GREETINGS, key=len, reverse=True):
            if rest == phrase or rest.startswith(phrase + " "):
                rest = rest[len(phrase):].strip()
                greeted = changed = True
                break
    if greeted:
        words = rest.split()
        while words and words[0] in GREETING_FILLER:
            words.pop(0)
        rest = " ".join(words)
    return greeted, rest


def is_spanish(norm):
    return any_phrase(norm, SPANISH)


VOWELS = set("aeiouy")


def looks_like_gibberish(norm, known_words):
    """
    True when nothing in the text is a word the helper knows and it looks like key-mashing:
    "asdfgh", "jjjjjj", "qwrtzp". Ordinary unknown questions ("what about llamas") are not
    gibberish -- they get the polite fallback, which is more useful.
    """
    words = norm.split()
    if not words:
        return True
    if any(w in known_words or w in STOPWORDS for w in words):
        return False
    for word in words:
        letters = [c for c in word if c.isalpha()]
        if not letters:
            continue
        if len(set(word)) <= 2 and len(word) >= 3:
            return True                       # "jjjj", "hahaha" is fine below
        vowel_share = sum(c in VOWELS for c in letters) / len(letters)
        if len(letters) >= 4 and vowel_share < 0.2:
            return True
        if re.search(r"(asdf|qwer|zxcv|hjkl|sdfg|dfgh|fghj|wert|erty|uiop|jkl)", word):
            return True
    return False
