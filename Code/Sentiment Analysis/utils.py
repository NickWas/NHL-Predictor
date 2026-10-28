import re
import os
import json
import pandas as pd
from datetime import datetime
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import matplotlib.pyplot as plt
import config
import logging

logger = logging.getLogger(__name__)

def download_nltk_data():
    """Download required NLTK data if not already downloaded."""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')

def preprocess_text(text):
    """
    Preprocess text for analysis by:
    - Converting to lowercase
    - Removing special characters
    - Removing URLs
    - Tokenizing
    - Removing stopwords
    """
    if not text or not isinstance(text, str):
        return []
    
    text = text.lower()
    
    # remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # remove special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    tokens = word_tokenize(text)
    
    # remove stopwords for efficiency
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    return filtered_tokens

def detect_teams_in_text(text):
    """Detect NHL teams mentioned in text."""
    if not text or not isinstance(text, str):
        return []
    
    text = text.lower()
    mentioned_teams = []
    
    for team, aliases in config.TEAM_NAMES.items():
        if any(alias in text for alias in aliases) or team.lower() in text:
            mentioned_teams.append(team)
    
    return mentioned_teams

def detect_player_abnormalities(text):
    """
    Detect player abnormalities in text.
    Returns a tuple of (player_name, status, context) if found, None otherwise.
    """
    if not text or not isinstance(text, str):
        return None
    
    text = text.lower()
    
    # reduce false positives w/ common player names (not all of them but for the sake of sanity)
    common_player_patterns = [
        r'\b(mcdavid|crosby|ovechkin|matthews|mackinnon|kucherov|makar|vasilevskiy|fox|hedman|draisaitl|pastrnak|marchand|panarin|huberdeau|gaudreau|stamkos|rantanen|tkachuk|pettersson|stone|tavares|point|marner|kaprizov|hughes|hellebuyck|shesterkin|robertson|eichel|connor|miller)\b',
        # last names
        r'\b(smith|jones|johnson|karlsson|backstrom|toews|price|fleury|malkin|pietrangelo|subban|kane|josi|barkov|aho|svechnikov|guentzel|laine|hellebuyck|andersen|markstrom|bobrovsky|kuemper|saros|rask|murray|bishop|schneider|crawford|gibson|quick|holtby|luongo|lundqvist)\b',
        # first names
        r'\b(sidney|connor|alex|auston|nathan|nikita|cale|andrei|adam|victor|leon|david|brad|artemi|johnny|steven|mikko|matthew|elias|mark|john|brayden|mitchell|kirill|quinn|igor|jason|jack|kyle|patrick|brady)\s+\w{3,15}\b'
    ]
    
    # autogen'd list of common
    non_player_words = {
        'scoring', 'win', 'start', 'player', 'time', 'team', 'season', 'game', 'hockey', 
        'temporarily', 'quick', 'finally', 'shortly', 'eventually', 'soon', 'definitely', 
        'before', 'after', 'during', 'while', 'until', 'since', 'because', 'although',
        'without', 'despite', 'through', 'throughout', 'within', 'beyond',
        
        'nhl', 'ahl', 'ref', 'refs', 'games', 'minutes', 'roster', 'goal', 'goals', 'shots',
        'lineup', 'skate', 'skates', 'puck', 'bench', 'gear', 'defense', 'offense',
        'forward', 'goalie', 'ice', 'rink', 'stick', 'league', 'playoff', 'playoffs',
        'season', 'seasons', 'schedule', 'trade', 'draft', 'pick', 'camp', 'practice',
        
        'play', 'playing', 'skate', 'skating', 'shoot', 'shooting', 'score', 'scoring',
        'pass', 'passing', 'block', 'blocking', 'hit', 'hitting', 'check', 'checking',
        'miss', 'missing', 'make', 'making', 'come', 'coming', 'go', 'going', 'get', 'getting',
        'be', 'being', 'do', 'doing', 'see', 'seeing', 'look', 'looking', 'feel', 'feeling',
        
        'quick', 'slow', 'good', 'bad', 'better', 'worse', 'best', 'worst', 'strong', 'weak',
        'fast', 'hard', 'soft', 'long', 'short', 'early', 'late', 'high', 'low',
        'big', 'small', 'more', 'less', 'most', 'least', 'many', 'few',
        
        'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
        'first', 'second', 'third', 'fourth', 'fifth', 'last', 'next',
        
        'a', 'an', 'the', 'this', 'that', 'these', 'those', 'they', 'them', 'their',
        'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'we', 'us', 'our', 'ours',
        'you', 'your', 'yours', 'i', 'me', 'my', 'mine',
        
        'to', 'from', 'in', 'on', 'at', 'by', 'with', 'about', 'against', 'between',
        'into', 'during', 'before', 'after', 'above', 'below', 'over', 'under', 'for'

        'point', 'quick'
    }
    
    # additional context patterns (test functionality)
    player_context_patterns = [
        r'\bplayer\b',
        r'\bforward\b',
        r'\bdefense(man)?\b',
        r'\bgoalie\b',
        r'\bgoaltender\b',
        r'\bcenter\b',
        r'\bwinger\b',
        r'\bcaptain\b',
        r'\brookie\b',
        r'\bgame\b',
        r'\bteam\b',
        r'\bline\b',
        r'\broster\b',
        r'\bseason\b',
        r'\bnhl\b'
    ]

    # text must contain necessary context
    has_hockey_context = any(re.search(pattern, text) for pattern in player_context_patterns)
    if not has_hockey_context:
        return None
    
    # check for injury keywords
    for keyword in config.INJURY_KEYWORDS:
        if keyword in text:
            # find name near keyword
            keyword_pos = text.find(keyword)
            context_before = text[max(0, keyword_pos - 50):keyword_pos] # 50 char context window
            context_after = text[keyword_pos:min(keyword_pos + 50, len(text))]
            full_context = context_before + context_after
            
            # find all patterns among already defined player patterns
            for pattern in common_player_patterns:
                player_matches = re.findall(pattern, full_context)
                if player_matches:
                    player_name = player_matches[0]
                    if isinstance(player_name, tuple):  # Multiple capture groups
                        player_name = player_name[0]
                    return (player_name, "injury", full_context)
            
            # if no patterns matched, try the words before the keyword
            words = text.split()
            if keyword in words:
                keyword_index = words.index(keyword)
                # check 3 words before the keyword
                for i in range(1, min(4, keyword_index + 1)):
                    potential_player = words[keyword_index - i]
                    
                    # skip common words defined above (surprisingly effective)
                    if (potential_player.lower() in non_player_words or
                        len(potential_player) < 3 or
                        potential_player in stopwords.words('english') or
                        potential_player.isdigit() or
                        potential_player in ['team', 'player', 'game', 'nhl', 'hockey', 'league', 'coach'] or
                        any(char.isdigit() for char in potential_player) or
                        potential_player.endswith('.') or
                        potential_player.endswith(',') or
                        potential_player.endswith('?') or
                        potential_player.endswith('!')
                    ):
                        continue
                    
                    # only accept likely player names (proper nouns). DEFINITELY NOT PERFECT.
                    if len(potential_player) > 3 and not potential_player.endswith('ing'):
                        # check if it looks like a name (capitalized in original text)
                        original_word_index = text.find(potential_player)
                        if original_word_index >= 0:
                            original_word = text[original_word_index:original_word_index + len(potential_player)]
                            # if the word was capitalized in the original text, it's more likely to be a proper noun
                            # but also this is reddit so im not sure
                            if original_word[0].isupper():
                                return (potential_player, "injury", full_context)
    
    # checking for return keywords
    for keyword in config.RETURN_KEYWORDS:
        if keyword in text:
            # find name near keyword
            keyword_pos = text.find(keyword)
            context_before = text[max(0, keyword_pos - 50):keyword_pos]
            context_after = text[keyword_pos:min(keyword_pos + 50, len(text))]
            full_context = context_before + context_after
            
            # find all patterns among already defined player patterns
            for pattern in common_player_patterns:
                player_matches = re.findall(pattern, full_context)
                if player_matches:
                    player_name = player_matches[0]
                    if isinstance(player_name, tuple):  # Multiple capture groups
                        player_name = player_name[0]
                    return (player_name, "return", full_context)
            
            # if no patterns matched, try the words before the keyword
            words = text.split()
            if keyword in words:
                keyword_index = words.index(keyword)
                # check 3 words before the keyword
                for i in range(1, min(4, keyword_index + 1)):
                    potential_player = words[keyword_index - i]
                    
                    # same deal as above
                    if (potential_player.lower() in non_player_words or
                        len(potential_player) < 3 or
                        potential_player in stopwords.words('english') or
                        potential_player.isdigit() or
                        potential_player in ['team', 'player', 'game', 'nhl', 'hockey', 'league', 'coach'] or
                        any(char.isdigit() for char in potential_player) or
                        potential_player.endswith('.') or
                        potential_player.endswith(',') or
                        potential_player.endswith('?') or
                        potential_player.endswith('!')
                    ):
                        continue
                    
                    # only accept likely player names (proper nouns)
                    if len(potential_player) > 3 and not potential_player.endswith('ing'):
                        # check if it looks like a name (capitalized in original text)
                        original_word_index = text.find(potential_player)
                        if original_word_index >= 0:
                            original_word = text[original_word_index:original_word_index + len(potential_player)]
                            # if the word was capitalized in the original text, it's more likely to be a proper noun
                            # but also this is reddit so im not sure
                            if original_word[0].isupper():
                                return (potential_player, "return", full_context)
    
    return None

def save_to_json(data, filename):
    """Save data to a JSON file with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename}_{timestamp}.json"
    os.makedirs('results', exist_ok=True)
    filepath = os.path.join('results', filename)
    
    # Makes data JSON serializable (ensures all data types are supported)
    def json_serialize(obj):
        # handle sets
        if isinstance(obj, set):
            return list(obj)
        # handle pandas Timestamp
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        # handle any other non-serializable types
        return str(obj)
    
    # Recursive function to process nested structures - safer version
    def process_data(item):
        try:
            if isinstance(item, dict):
                return {k: process_data(v) for k, v in item.items()}
            elif isinstance(item, (list, tuple, set)):
                return [process_data(i) for i in item]
            elif item is None:
                return None
            # Safely check if it's a datetime without using pandas functions
            elif hasattr(item, 'isoformat'):
                return item.isoformat()
            # Stick to basic types to avoid issues
            elif isinstance(item, (str, int, float, bool)):
                return item
            else:
                # Convert anything else to string
                return str(item)
        except Exception as e:
            # In case of any error, convert to string
            logger.warning(f"Error processing data for JSON: {str(e)}")
            return str(item)
    
    try:
        # using process_data b/c safer
        processed_data = process_data(data)
        
        with open(filepath, 'w') as f:
            json.dump(processed_data, f, indent=2, default=json_serialize)
        
        print(f"Data saved to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving JSON file: {str(e)}")
        # fallback - try to save a simplified version
        try:
            # try with a simple stringified version as last resort
            with open(filepath, 'w') as f:
                json.dump(str(data), f)
            logger.warning(f"Saved simplified version of data due to serialization error")
            return filepath
        except Exception as e2:
            logger.error(f"Failed to save even simplified data: {str(e2)}")
            return None

def save_to_csv(df, filename):
    """Save DataFrame to a CSV file with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename}_{timestamp}.csv"
    os.makedirs('results', exist_ok=True)
    filepath = os.path.join('results', filename)
    
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")
    return filepath


def plot_team_sentiment(team_sentiment, output_dir='results', output_file=None):
    """Plot team sentiment scores.
    
    Args:
        team_sentiment (dict): Dictionary mapping team names to sentiment scores
        output_dir (str): Directory to save the plot
        output_file (str): Optional file path to save the plot. If provided, overrides output_dir.
    
    Returns:
        str: Path to the saved plot file
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(list(team_sentiment.items()), columns=['Team', 'Sentiment'])
    df = df.sort_values('Sentiment', ascending=False)
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(df['Team'], df['Sentiment'])
    
    for i, bar in enumerate(bars):
        if df['Sentiment'].iloc[i] > 0:
            bar.set_color('green')
        else:
            bar.set_color('red')
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xticks(rotation=90)
    plt.xlabel('NHL Teams')
    plt.ylabel('Sentiment Score')
    plt.title('NHL Team Sentiment Analysis')
    plt.tight_layout()
    
    if output_file:
        filepath = output_file
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"team_sentiment_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
    
    plt.savefig(filepath)
    plt.close()
    
    print(f"Sentiment plot saved to {filepath}")
    return filepath

def load_scraped_ids(filename='scraped_post_ids.json'):
    """
    Load already scraped post IDs from file.
    
    Args:
        filename (str): Filename to load IDs from
        
    Returns:
        set: Set of post IDs that have already been scraped
    """
    filepath = os.path.join('results', filename)
    
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                scraped_ids = set(json.load(f))
            print(f"Loaded {len(scraped_ids)} already scraped post IDs")
            return scraped_ids
        except Exception as e:
            print(f"Error loading scraped IDs: {e}")
            return set()
    else:
        print("No existing scraped IDs file found. Starting fresh.")
        return set()

def save_scraped_ids(scraped_ids, filename='scraped_post_ids.json'):
    """
    Save scraped post IDs to file.
    
    Args:
        scraped_ids (set): Set of post IDs that have been scraped
        filename (str): Filename to save IDs to
    """
    os.makedirs('results', exist_ok=True)
    filepath = os.path.join('results', filename)
    
    try:
        with open(filepath, 'w') as f:
            json.dump(list(scraped_ids), f)
        print(f"Saved {len(scraped_ids)} scraped post IDs")
    except Exception as e:
        print(f"Error saving scraped IDs: {e}") 