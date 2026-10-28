from dotenv import load_dotenv
import os

load_dotenv()

# Reddit API credentials
REDDIT_CLIENT_ID = os.getenv("ID")
REDDIT_CLIENT_SECRET = os.getenv("SECRET")
REDDIT_USER_AGENT = "NHL Sentiment Analysis Tool v1.0 by /u/Charming_Craft_1102"

# NHL Subreddits to scrape
NHL_SUBREDDITS = [
    "nhl",                   # Main NHL subreddit
    "AnaheimDucks",          # Anaheim Ducks
    "coyotes",               # Arizona Coyotes
    "BostonBruins",          # Boston Bruins
    "sabres",                # Buffalo Sabres
    "CalgaryFlames",         # Calgary Flames
    "canes",                 # Carolina Hurricanes
    "hawks",                 # Chicago Blackhawks
    "ColoradoAvalanche",     # Colorado Avalanche
    "BlueJackets",           # Columbus Blue Jackets
    "DallasStars",           # Dallas Stars
    "DetroitRedWings",       # Detroit Red Wings
    "EdmontonOilers",        # Edmonton Oilers
    "FloridaPanthers",       # Florida Panthers
    "losangeleskings",       # Los Angeles Kings
    "wildhockey",            # Minnesota Wild
    "Habs",                  # Montreal Canadiens
    "Predators",             # Nashville Predators
    "devils",                # New Jersey Devils
    "NewYorkIslanders",      # New York Islanders
    "rangers",               # New York Rangers
    "OttawaSenators",        # Ottawa Senators
    "Flyers",                # Philadelphia Flyers
    "penguins",              # Pittsburgh Penguins
    "SanJoseSharks",         # San Jose Sharks
    "SeattleKraken",         # Seattle Kraken
    "stlouisblues",          # St. Louis Blues
    "TampaBayLightning",     # Tampa Bay Lightning
    "leafs",                 # Toronto Maple Leafs
    "canucks",               # Vancouver Canucks
    "goldenknights",         # Vegas Golden Knights
    "caps",                  # Washington Capitals
    "winnipegjets",          # Winnipeg Jets
]

# Team names and alternate names mapping
TEAM_NAMES = {
    "Anaheim Ducks": ["ducks", "anaheim", "quack"],
    "Boston Bruins": ["bruins", "boston", "bears"],
    "Buffalo Sabres": ["sabres", "buffalo"],
    "Calgary Flames": ["flames", "calgary"],
    "Carolina Hurricanes": ["hurricanes", "carolina", "canes"],
    "Chicago Blackhawks": ["blackhawks", "chicago", "hawks"],
    "Colorado Avalanche": ["avalanche", "colorado", "avs"],
    "Columbus Blue Jackets": ["blue jackets", "columbus", "cbj"],
    "Dallas Stars": ["stars", "dallas"],
    "Detroit Red Wings": ["red wings", "detroit", "wings"],
    "Edmonton Oilers": ["oilers", "edmonton"],
    "Florida Panthers": ["panthers", "florida"],
    "Los Angeles Kings": ["kings", "los angeles", "la kings"],
    "Minnesota Wild": ["wild", "minnesota"],
    "Montreal Canadiens": ["canadiens", "montreal", "habs"],
    "Nashville Predators": ["predators", "nashville", "preds"],
    "New Jersey Devils": ["devils", "new jersey"],
    "New York Islanders": ["islanders", "isles"],
    "New York Rangers": ["rangers"],
    "Ottawa Senators": ["senators", "ottawa", "sens"],
    "Philadelphia Flyers": ["flyers", "philadelphia", "philly"],
    "Pittsburgh Penguins": ["penguins", "pittsburgh", "pens"],
    "San Jose Sharks": ["sharks", "san jose"],
    "Seattle Kraken": ["kraken", "seattle"],
    "St. Louis Blues": ["blues", "st. louis", "stl"],
    "Tampa Bay Lightning": ["lightning", "tampa bay", "bolts"],
    "Toronto Maple Leafs": ["maple leafs", "toronto", "leafs"],
    "Utah Hockey Club": ["utah hockey club", "utah", "utah hockey"],
    "Vancouver Canucks": ["canucks", "vancouver"],
    "Vegas Golden Knights": ["golden knights", "vegas", "knights", "vgk"],
    "Washington Capitals": ["capitals", "washington", "caps"],
    "Winnipeg Jets": ["jets", "winnipeg"],
}

# Player abnormality detector keywords
INJURY_KEYWORDS = [
    "injury", "injured", "hurt", "pain", "strain", "sprain", "broken",
    "fracture", "concussion", "surgery", "hospital", "rehab", "rehabilitation",
    "protocol", "recovery", "sidelined", "out for", "miss time", "missing",
    "week-to-week", "day-to-day", "IR", "LTIR", "injured reserve"
]

RETURN_KEYWORDS = [
    "return", "returned", "returning", "comeback", "back from", "recovered",
    "recovering", "healing", "healed", "back on ice", "practicing", "practice",
    "activated", "cleared", "back in lineup", "back in action"
]

# Sentiment analysis settings
SENTIMENT_THRESHOLD_POSITIVE = 0.2
SENTIMENT_THRESHOLD_NEGATIVE = -0.2

# Scraping settings
POSTS_LIMIT = 1000  # Number of posts to scrape per subreddit
COMMENT_LIMIT = 100  # Number of comments to scrape per post
TIME_FILTER = "year"  # Options: day, week, month, year, all 