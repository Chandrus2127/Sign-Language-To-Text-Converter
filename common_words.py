"""
common_words.py - Smart Word Suggestions and Spellchecking Engine
Provides prefix-based autocomplete and spellcheck recommendations
using common English vocabulary with fallback to PyEnchant.
"""

try:
    import enchant
    _enchant_dict = enchant.Dict("en_US")
except Exception:
    _enchant_dict = None

# Curated vocabulary of common English words
COMMON_WORDS = [
    "ABOUT", "ABOVE", "ACROSS", "ACTION", "ACTIVITY", "ACTUALLY", "ADD", "ADDRESS",
    "AFTER", "AGAIN", "AGAINST", "AGE", "AGO", "AGREE", "AIR", "ALL", "ALLOW",
    "ALMOST", "ALONE", "ALONG", "ALREADY", "ALSO", "ALTHOUGH", "ALWAYS", "AM",
    "AMERICAN", "AMONG", "AMOUNT", "AND", "ANIMAL", "ANOTHER", "ANSWER", "ANY",
    "ANYBODY", "ANYONE", "ANYTHING", "ANYWAY", "APPEAR", "APPLE", "AREA", "ARM",
    "AROUND", "ARRIVE", "ART", "ARTICLE", "AS", "ASK", "AT", "ATTACK", "ATTENTION",
    "AUTHOR", "AUTHORITY", "AVAILABLE", "AVOID", "AWAY", "BABY", "BACK", "BAD",
    "BAG", "BALL", "BANK", "BAR", "BASE", "BASIC", "BASIS", "BE", "BEAR", "BEAT",
    "BEAUTIFUL", "BECAUSE", "BECOME", "BED", "BEFORE", "BEGIN", "BEHIND", "BELIEVE",
    "BELONG", "BELOW", "BESIDE", "BEST", "BETTER", "BETWEEN", "BEYOND", "BIG",
    "BILL", "BIRD", "BIRTH", "BIT", "BLACK", "BLOOD", "BLOW", "BLUE", "BOARD",
    "BODY", "BOOK", "BORN", "BOTH", "BOTTLE", "BOTTOM", "BOX", "BOY", "BRANCH",
    "BREAD", "BREAK", "BREATH", "BRING", "BROTHER", "BROWN", "BUILD", "BUILDING",
    "BURN", "BUSINESS", "BUSY", "BUT", "BUY", "BY", "CALL", "CAMERA", "CAMP",
    "CAN", "CANCER", "CANDIDATE", "CAPITAL", "CAR", "CARD", "CARE", "CAREER",
    "CARRY", "CASE", "CATCH", "CAUSE", "CELL", "CENTER", "CENTRAL", "CENTURY",
    "CERTAIN", "CERTAINLY", "CHAIR", "CHALLENGE", "CHANCE", "CHANGE", "CHARACTER",
    "CHARGE", "CHECK", "CHILD", "CHILDREN", "CHOICE", "CHOOSE", "CHURCH", "CITIZEN",
    "CITY", "CIVIL", "CLAIM", "CLASS", "CLEAR", "CLEARLY", "CLOSE", "CLOSER",
    "CLOTHES", "CLOUD", "CLUB", "COACH", "COLD", "COLLECTION", "COLLEGE", "COLOR",
    "COME", "COMMERCIAL", "COMMON", "COMMUNITY", "COMPANY", "COMPARE", "COMPLETE",
    "COMPLETELY", "COMPUTER", "CONCERN", "CONDITION", "CONFERENCE", "CONGRESS",
    "CONSIDER", "CONSUMER", "CONTAIN", "CONTINUE", "CONTROL", "CONVERSATION",
    "CONVERSION", "COOK", "COOL", "CORNER", "COST", "COULD", "COUNTRY", "COUPLE",
    "COURSE", "COURT", "COVER", "CREATE", "CRIME", "CULTURAL", "CULTURE", "CUP",
    "CURRENT", "CURRENTLY", "CUSTOMER", "CUT", "DAILY", "DANCE", "DANGER", "DARK",
    "DATA", "DAUGHTER", "DAY", "DEAD", "DEAF", "DEAL", "DEATH", "DECIDE", "DECISION",
    "DEEP", "DEFENSE", "DEGREE", "DEMOCRAT", "DEMOCRATIC", "DESCRIBE", "DESIGN",
    "DESPITE", "DETAIL", "DETERMINE", "DEVELOP", "DEVELOPMENT", "DEVICE", "DIFFERENCE",
    "DIFFERENT", "DIFFICULT", "DINNER", "DIRECTION", "DIRECTOR", "DISCOVER", "DISCUSS",
    "DISCUSSION", "DISEASE", "DO", "DOCTOR", "DOG", "DOOR", "DOWN", "DRAW", "DREAM",
    "DRIVE", "DROP", "DRUG", "DUMB", "DURING", "EACH", "EARLY", "EAST", "EASY",
    "EAT", "ECONOMIC", "ECONOMY", "EDGE", "EDUCATION", "EFFECT", "EFFORT", "EIGHT",
    "EITHER", "ELECTION", "ELSE", "EMPLOYEE", "END", "ENERGY", "ENJOY", "ENOUGH",
    "ENTER", "ENTIRE", "ENVIRONMENT", "ENVIRONMENTAL", "ESPECIALLY", "ESTABLISH",
    "EVEN", "EVENING", "EVENT", "EVER", "EVERY", "EVERYBODY", "EVERYONE", "EVERYTHING",
    "EVIDENCE", "EXACTLY", "EXAMPLE", "EXECUTIVE", "EXIST", "EXPECT", "EXPERIENCE",
    "EXPERT", "EXPLAIN", "EYE", "FACE", "FACT", "FACTOR", "FAIL", "FALL", "FAMILY",
    "FAR", "FAST", "FATHER", "FEAR", "FEATURE", "FEDERAL", "FEEL", "FEELING", "FEW",
    "FIELD", "FIGHT", "FIGURE", "FILL", "FILM", "FINAL", "FINALLY", "FINANCIAL",
    "FIND", "FINE", "FINGER", "FINISH", "FIRE", "FIRM", "FIRST", "FISH", "FIVE",
    "FLOOR", "FLY", "FOCUS", "FOLLOW", "FOOD", "FOOT", "FOR", "FORCE", "FOREIGN",
    "FOREVER", "FORGET", "FORM", "FORMATION", "FORWARD", "FOUND", "FOUR", "FREE",
    "FREEDOM", "FRIEND", "FROM", "FRONT", "FULL", "FUN", "FUNCTION", "FUTURE",
    "GAME", "GARDEN", "GAS", "GENERAL", "GENERATION", "GESTURE", "GET", "GIRL",
    "GIVE", "GLASS", "GO", "GOAL", "GOD", "GOOD", "GOODBYE", "GOVERNMENT", "GREAT",
    "GREEN", "GROUND", "GROUP", "GROW", "GROWTH", "GUESS", "GUEST", "GUIDE", "GUN",
    "GUY", "HAIR", "HALF", "HAND", "HANG", "HAPPEN", "HAPPY", "HARD", "HAVE", "HE",
    "HEAD", "HEALTH", "HEAR", "HEARING", "HEART", "HEAT", "HEAVY", "HELD", "HELL",
    "HELLO", "HELMET", "HELP", "HER", "HERE", "HERSELF", "HIGH", "HIM", "HIMSELF",
    "HIS", "HISTORY", "HIT", "HOLD", "HOME", "HOPE", "HOSPITAL", "HOT", "HOTEL",
    "HOUR", "HOUSE", "HOW", "HOWEVER", "HUGE", "HUMAN", "HUNDRED", "HUSBAND",
    "I", "ICE", "IDEA", "IDENTIFY", "IF", "IMAGE", "IMAGINE", "IMPACT", "IMPORTANT",
    "IMPROVE", "IN", "INCLUDE", "INCLUDING", "INCREASE", "INDEED", "INDICATE",
    "INDIVIDUAL", "INDUSTRY", "INFORMATION", "INSIDE", "INSTEAD", "INSTITUTION",
    "INTEREST", "INTERESTING", "INTERNATIONAL", "INTERVIEW", "INTO", "INVESTMENT",
    "INVOLVE", "ISSUE", "IT", "ITEM", "ITS", "ITSELF", "JOB", "JOIN", "JUST",
    "KEEP", "KEY", "KID", "KILL", "KIND", "KITCHEN", "KNOW", "KNOWLEDGE", "LAND",
    "LANGUAGE", "LARGE", "LAST", "LATE", "LATER", "LAUGH", "LAW", "LAWYER", "LAY",
    "LEAD", "LEADER", "LEADERSHIP", "LEARN", "LEAST", "LEAVE", "LEFT", "LEG",
    "LEGAL", "LESS", "LET", "LETTER", "LEVEL", "LIE", "LIFE", "LIGHT", "LIKE",
    "LIKELY", "LINE", "LIST", "LISTEN", "LITTLE", "LIVE", "LOCAL", "LONG", "LOOK",
    "LOSE", "LOSS", "LOT", "LOVE", "LOW", "MACHINE", "MAGAZINE", "MAIN", "MAINTAIN",
    "MAJOR", "MAJORITY", "MAKE", "MAN", "MANAGE", "MANAGEMENT", "MANAGER", "MANY",
    "MARKET", "MARRIAGE", "MATERIAL", "MATTER", "MAY", "MAYBE", "ME", "MEAN",
    "MEASURE", "MEDIA", "MEDICAL", "MEET", "MEETING", "MEMBER", "MEMORY", "MENTION",
    "MESSAGE", "METHOD", "MIDDLE", "MIGHT", "MILITARY", "MILLION", "MIND", "MINUTE",
    "MISS", "MISSION", "MODEL", "MODERN", "MOMENT", "MONEY", "MONTH", "MORE",
    "MORNING", "MOST", "MOTHER", "MOUTH", "MOVE", "MOVEMENT", "MOVIE", "MR", "MRS",
    "MS", "MUCH", "MUSIC", "MUST", "MY", "MYSELF", "NAME", "NATION", "NATIONAL",
    "NATURAL", "NATURE", "NEAR", "NEARLY", "NECESSARY", "NEED", "NETWORK", "NEVER",
    "NEW", "NEWS", "NEWSPAPER", "NEXT", "NICE", "NIGHT", "NINE", "NO", "NONE",
    "NOR", "NORMAL", "NORMALLY", "NORTH", "NOT", "NOTE", "NOTHING", "NOTICE",
    "NOW", "NUMBER", "OBJECT", "OCCUR", "OF", "OFF", "OFFER", "OFFICE", "OFFICER",
    "OFFICIAL", "OFTEN", "OH", "OIL", "OK", "OKAY", "OLD", "ON", "ONCE", "ONE",
    "ONLY", "ONTO", "OPEN", "OPERATION", "OPINION", "OPPORTUNITY", "OPTION", "OR",
    "ORDER", "ORGANIZATION", "OTHER", "OTHERS", "OUR", "OUT", "OUTSIDE", "OVER",
    "OWN", "OWNER", "PAGE", "PAIN", "PAINTING", "PAPER", "PARENT", "PART",
    "PARTICIPANT", "PARTICULAR", "PARTICULARLY", "PARTNER", "PARTY", "PASS",
    "PAST", "PATIENT", "PATTERN", "PAY", "PEACE", "PEOPLE", "PER", "PERFORM",
    "PERFORMANCE", "PERHAPS", "PERIOD", "PERSON", "PERSONAL", "PHONE", "PHYSICAL",
    "PICK", "PICTURE", "PIECE", "PLACE", "PLAN", "PLANT", "PLAY", "PLAYER", "PM",
    "POINT", "POLICE", "POLICY", "POLITICAL", "POLITICS", "POOR", "POPULAR",
    "POPULATION", "POSITION", "POSITIVE", "POSSIBLE", "POWER", "PRACTICE", "PREPARE",
    "PRESENT", "PRESIDENT", "PRESSURE", "PRETTY", "PREVENT", "PRICE", "PRIVATE",
    "PROBABLY", "PROBLEM", "PROCESS", "PRODUCE", "PRODUCT", "PRODUCTION", "PROFESSIONAL",
    "PROFESSOR", "PROGRAM", "PROJECT", "PROPERTY", "PROTECT", "PROVE", "PROVIDE",
    "PUBLIC", "PULL", "PURPOSE", "PUSH", "PUT", "QUALITY", "QUESTION", "QUICK",
    "QUICKLY", "QUIET", "QUITE", "RACE", "RADIO", "RAISE", "RANGE", "RATE",
    "RATHER", "REACH", "READ", "READY", "REAL", "REALITY", "REALIZE", "REALLY",
    "REASON", "RECEIVE", "RECENT", "RECENTLY", "RECOGNIZE", "RECORD", "RED",
    "REDUCE", "REFLECT", "REGION", "RELATE", "RELATIONSHIP", "RELIGIOUS", "REMAIN",
    "REMEMBER", "REMOVE", "REPORT", "REPRESENT", "REPUBLICAN", "REQUIRE", "RESEARCH",
    "RESOURCE", "RESPOND", "RESPONSE", "RESPONSIBILITY", "REST", "RESULT", "RETURN",
    "REVEAL", "RICH", "RIGHT", "RISE", "RISK", "ROAD", "ROCK", "ROLE", "ROOM",
    "RULE", "RUN", "SAFE", "SAME", "SAVE", "SAY", "SCENE", "SCHOOL", "SCIENCE",
    "SCIENTIST", "SCORE", "SEA", "SEASON", "SEAT", "SECOND", "SECTION", "SECURITY",
    "SEE", "SEEK", "SEEM", "SELL", "SEND", "SENIOR", "SENSE", "SERIES", "SERIOUS",
    "SERVE", "SERVICE", "SET", "SEVEN", "SEVERAL", "SEX", "SHAKE", "SHARE", "SHE",
    "SHOOT", "SHORT", "SHOT", "SHOULD", "SHOULDER", "SHOW", "SIDE", "SIGN",
    "SIGNAL", "SIGNIFICANT", "SIMILAR", "SIMPLE", "SIMPLY", "SINCE", "SING",
    "SINGLE", "SISTER", "SIT", "SITE", "SITUATION", "SIX", "SIZE", "SKILL",
    "SKIN", "SMALL", "SMILE", "SO", "SOCIAL", "SOCIETY", "SOLDIER", "SOME",
    "SOMEBODY", "SOMEONE", "SOMETHING", "SOMETIMES", "SON", "SONG", "SOON",
    "SORRY", "SORT", "SOUND", "SOURCE", "SOUTH", "SOUTHERN", "SPACE", "SPEAK",
    "SPECIAL", "SPECIFIC", "SPEECH", "SPEND", "SPORT", "SPRING", "STAFF",
    "STAGE", "STAND", "STANDARD", "STAR", "START", "STATE", "STATEMENT", "STATION",
    "STAY", "STEP", "STILL", "STOCK", "STOP", "STORE", "STORY", "STRATEGY",
    "STREET", "STRONG", "STRUCTURE", "STUDENT", "STUDY", "STUFF", "STYLE",
    "SUBJECT", "SUCCESS", "SUCCESSFUL", "SUCH", "SUDDENLY", "SUFFER", "SUGGEST",
    "SUGGESTION", "SUMMER", "SUPPORT", "SURE", "SURFACE", "SYSTEM", "TABLE",
    "TAKE", "TALK", "TASK", "TAX", "TEACH", "TEACHER", "TEAM", "TECHNOLOGY",
    "TELEVISION", "TELL", "TEN", "TEND", "TERM", "TEST", "TEXT", "THAN",
    "THANK", "THANKS", "THAT", "THE", "THEIR", "THEM", "THEMSELVES", "THEN",
    "THEORY", "THERE", "THESE", "THEY", "THING", "THINK", "THIRD", "THIS",
    "THOSE", "THOUGH", "THOUGHT", "THOUSAND", "THREAT", "THREE", "THROUGH",
    "THROUGHOUT", "THROW", "THUS", "TIME", "TO", "TODAY", "TOGETHER", "TONIGHT",
    "TOO", "TOP", "TOTAL", "TOUGH", "TOWARD", "TOWN", "TRADE", "TRADITIONAL",
    "TRAINING", "TRAVEL", "TREAT", "TREATMENT", "TREE", "TRIAL", "TRIP", "TRUE",
    "TRUTH", "TRY", "TURN", "TV", "TWO", "TYPE", "UNDER", "UNDERSTAND", "UNIT",
    "UNTIL", "UP", "UPON", "US", "USE", "USER", "USUAL", "USUALLY", "VALUE",
    "VARIOUS", "VERY", "VICTIM", "VIEW", "VIOLENCE", "VISIT", "VOICE", "VOTE",
    "WAIT", "WALK", "WALL", "WANT", "WAR", "WATCH", "WATER", "WAVE", "WAY",
    "WE", "WEAPON", "WEAR", "WEEK", "WEIGHT", "WELCOME", "WELL", "WEST", "WESTERN",
    "WHAT", "WHATEVER", "WHEN", "WHERE", "WHETHER", "WHICH", "WHILE", "WHITE",
    "WHO", "WHOLE", "WHOM", "WHOSE", "WHY", "WIDE", "WIFE", "WILL", "WIN",
    "WIND", "WINDOW", "WISH", "WITH", "WITHIN", "WITHOUT", "WOMAN", "WONDER",
    "WORD", "WORK", "WORKER", "WORLD", "WORRY", "WOULD", "WRITE", "WRITER",
    "WRONG", "YARD", "YEAH", "YEAR", "YES", "YET", "YOU", "YOUNG", "YOUR",
    "YOURSELF"
]

# Priority common conversational words (appear first in suggestions)
PRIORITY_WORDS = [
    "HELLO", "HELP", "HELMET", "HOW", "HAVE", "HERE", "HOME", "HEAD", "HEART",
    "THANK", "THANKS", "THAT", "THIS", "THE", "THEY", "THERE", "THESE", "THINK",
    "GOOD", "GREAT", "GIVE", "GET", "GO", "GOING", "GAME",
    "PLEASE", "PROJECT", "PEOPLE", "PERSON", "PLACE", "PLAY",
    "SORRY", "SIGN", "SIGNAL", "LANGUAGE", "SPEAK", "START", "STOP",
    "WHAT", "WHEN", "WHERE", "WHY", "WHO", "WHICH", "WITH", "WATER", "WORK",
    "YES", "NO", "NAME", "NICE", "NEED", "NEW", "NEXT", "NIGHT",
    "YOU", "YOUR", "ME", "MY", "FINE", "FRIEND", "FAMILY", "LOVE", "LIKE"
]

def get_word_suggestions(raw_word: str, max_count: int = 3) -> list:
    """
    Returns up to max_count word suggestions for the given raw_word.
    Combines prefix matching with spellcheck suggestions from enchant.
    """
    if not raw_word:
        return ["HELLO", "THANK YOU", "PLEASE"][:max_count]

    query = raw_word.strip().upper()
    if not query:
        return ["HELLO", "THANK YOU", "PLEASE"][:max_count]

    suggestions = []

    # 1. Exact priority conversational matches
    for word in PRIORITY_WORDS:
        if word.startswith(query):
            if word not in suggestions:
                suggestions.append(word)
                if len(suggestions) >= max_count:
                    return suggestions

    # 2. Exact matches from general common words
    for word in COMMON_WORDS:
        if word.startswith(query):
            if word not in suggestions:
                suggestions.append(word)
                if len(suggestions) >= max_count:
                    return suggestions

    # 3. PyEnchant spellcheck fallback
    if _enchant_dict is not None and len(suggestions) < max_count:
        try:
            ench_results = _enchant_dict.suggest(query)
            for cand in ench_results:
                cand_upper = cand.strip().upper()
                if cand_upper.isalpha() and cand_upper not in suggestions:
                    suggestions.append(cand_upper)
                    if len(suggestions) >= max_count:
                        return suggestions
        except Exception:
            pass

    # 4. Fallback fuzzy search if still under count
    if len(suggestions) < max_count:
        first_char = query[0] if query else ""
        for word in COMMON_WORDS:
            if word.startswith(first_char) and word not in suggestions:
                suggestions.append(word)
                if len(suggestions) >= max_count:
                    break

    # Pad with empty strings if not enough
    while len(suggestions) < max_count:
        suggestions.append("")

    return suggestions[:max_count]
