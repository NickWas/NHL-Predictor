import pandas as pd
import re
import logging
import utils
import config
from collections import defaultdict

# super scuffed. doesn't fully work.

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("log/player_detector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PlayerDetector:
    """
    A class to detect player abnormalities in NHL-related Reddit content.
    """
    
    def __init__(self):
        """
        Initialize the player detector.
        """
        logger.info("Player detector initialized")
    
    def detect_abnormalities(self, df, save_to_file=True):
        """
        Detect player abnormalities in the provided DataFrame.
        
        Args:
            df (DataFrame): DataFrame containing Reddit data
            
        Returns:
            DataFrame: DataFrame containing detected abnormalities
        """
        if df.empty:
            logger.warning("Empty DataFrame provided, no abnormalities to detect")
            return pd.DataFrame()
        
        logger.info("Detecting player abnormalities")
        
        abnormalities = []
        
        posts_mask = df['is_post'] == True
        for idx, row in df[posts_mask].iterrows():
            title = str(row['title'] or '') if pd.notna(row['title']) else ''
            selftext = str(row['selftext'] or '') if pd.notna(row['selftext']) else ''
            combined_text = title + ' ' + selftext
            
            detected = utils.detect_player_abnormalities(combined_text)
            if detected:
                player, status, context = detected
                
                abnormalities.append({
                    'player': player,
                    'status': status,
                    'context': context,
                    'subreddit': row['subreddit'],
                    'post_id': row['id'],
                    'post_title': row['title'],
                    'created_utc': row['created_utc'],
                    'is_comment': False
                })
        
        comments_mask = df['is_post'] == False
        for idx, row in df[comments_mask].iterrows():
            comment_text = str(row['comment_body'] or '') if pd.notna(row['comment_body']) else ''
            
            detected = utils.detect_player_abnormalities(comment_text)
            if detected:
                player, status, context = detected
                
                abnormalities.append({
                    'player': player,
                    'status': status,
                    'context': context,
                    'subreddit': row['subreddit'],
                    'post_id': row['parent_id'],
                    'comment_id': row['id'],
                    'created_utc': row['created_utc'],
                    'is_comment': True
                })
        
        abnormalities_df = pd.DataFrame(abnormalities)
        
        if not abnormalities_df.empty:
            if save_to_file:
                utils.save_to_csv(abnormalities_df, "player_abnormalities")
            
            logger.info(f"Detected {len(abnormalities_df)} player abnormalities")
        else:
            logger.info("No player abnormalities detected")
        
        return abnormalities_df
    
    def get_abnormality_summary(self, abnormalities_df, save_to_file=True):
        """
        Generate a summary of detected abnormalities.
        
        Args:
            abnormalities_df (DataFrame): DataFrame containing detected abnormalities
            
        Returns:
            dict: Dictionary containing summary information
        """
        if abnormalities_df.empty:
            return {
                'total': 0,
                'injuries': 0,
                'returns': 0,
                'players': set(),
                'recent_injuries': [],
                'recent_returns': []
            }
        
        status_counts = abnormalities_df['status'].value_counts().to_dict()
        injuries_count = status_counts.get('injury', 0)
        returns_count = status_counts.get('return', 0)
        
        unique_players = set(abnormalities_df['player'].unique())
        
        abnormalities_df['created_utc'] = pd.to_datetime(abnormalities_df['created_utc'])
        recent_df = abnormalities_df.sort_values('created_utc', ascending=False)
        
        clean_df = recent_df.copy()
        
        clean_df = clean_df.fillna({
            'post_title': '',  # replace NaN post_title with empty string to avoid error
            'comment_body': '',
            'subreddit': '',
            'post_id': '',
            'comment_id': ''
        })
        
        # convert all columns to appropriate types
        for col in clean_df.columns:
            if pd.api.types.is_datetime64_any_dtype(clean_df[col]):
                clean_df[col] = clean_df[col].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # recent injuries and returns (up to 5 each)
        recent_injuries = clean_df[clean_df['status'] == 'injury'].head(5).to_dict('records')
        recent_returns = clean_df[clean_df['status'] == 'return'].head(5).to_dict('records')
        
        summary = {
            'total': len(abnormalities_df),
            'injuries': injuries_count,
            'returns': returns_count,
            'players': unique_players,
            'recent_injuries': recent_injuries,
            'recent_returns': recent_returns
        }
        if save_to_file:
            utils.save_to_json(summary, "abnormality_summary")
        
        return summary
    
    def group_by_player(self, abnormalities_df, save_to_file=True):
        """
        Group abnormalities by player.
        
        Args:
            abnormalities_df (DataFrame): DataFrame containing detected abnormalities
            
        Returns:
            dict: Dictionary mapping players to their abnormalities
        """
        if abnormalities_df.empty:
            return {}
        
        player_abnormalities = defaultdict(list)
        
        for _, row in abnormalities_df.iterrows():
            player = row['player']
            player_abnormalities[player].append(row.to_dict())
        
        for player in player_abnormalities:
            player_abnormalities[player] = sorted(
                player_abnormalities[player],
                key=lambda x: x['created_utc'],
                reverse=True
            )
        if save_to_file:
            utils.save_to_json(dict(player_abnormalities), "player_abnormalities")
        
        return dict(player_abnormalities)

# test
if __name__ == "__main__":
    from reddit_scraper import RedditScraper
    
    scraper = RedditScraper()
    detector = PlayerDetector()
    
    data = scraper.scrape_nhl_subreddits()
    
    abnormalities = detector.detect_abnormalities(data)
    
    summary = detector.get_abnormality_summary(abnormalities)
    
    player_abnormalities = detector.group_by_player(abnormalities)
    
    print(f"Total abnormalities detected: {summary['total']}")
    print(f"Injuries: {summary['injuries']}")
    print(f"Returns: {summary['returns']}")
    print(f"Unique players: {len(summary['players'])}") 