import praw
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import time
import config
import utils
import logging
import os

# Set up logging
# Create log directory if it doesn't exist
os.makedirs('log', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("log/reddit_scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RedditScraper:
    """
    A class to handle scraping of NHL-related content from Reddit.
    """
    
    def __init__(self, client_id=None, client_secret=None, user_agent=None):
        """
        Initialize the Reddit scraper with API credentials.
        
        Args:
            client_id (str): Reddit API client ID
            client_secret (str): Reddit API client secret
            user_agent (str): Reddit API user agent
        """
        # default params defined in config, override if needed
        self.client_id = client_id or config.REDDIT_CLIENT_ID
        self.client_secret = client_secret or config.REDDIT_CLIENT_SECRET
        self.user_agent = user_agent or config.REDDIT_USER_AGENT
        
        # reddit api client
        self.reddit = praw.Reddit(
            client_id=self.client_id,
            client_secret=self.client_secret,
            user_agent=self.user_agent
        )
        
        # prev scraped ids for no dupes
        self.scraped_ids = utils.load_scraped_ids()
        
        logger.info("Reddit API client initialized")
        
    def scrape_subreddit(self, subreddit_name, posts_limit=None, time_filter=None, comment_limit=None, skip_existing=True):
        """
        Scrape posts and comments from a subreddit.
        
        Args:
            subreddit_name (str): Name of the subreddit to scrape
            posts_limit (int): Maximum number of posts to scrape
            time_filter (str): Time filter for posts (day, week, month, year, all)
            comment_limit (int): Maximum number of comments to scrape per post
            skip_existing (bool): Whether to skip posts that have already been scraped
            
        Returns:
            DataFrame: DataFrame containing scraped data
        """
        # TODO: remove redundancy
        posts_limit = posts_limit or config.POSTS_LIMIT
        time_filter = time_filter or config.TIME_FILTER
        comment_limit = comment_limit or config.COMMENT_LIMIT
        
        logger.info(f"Scraping r/{subreddit_name} - Limit: {posts_limit} posts, Time filter: {time_filter}")
        
        subreddit = self.reddit.subreddit(subreddit_name)
        
        data = []
        
        # Count for new posts
        new_posts_count = 0
        skipped_posts_count = 0
        
        try:
            # Get top posts from the subreddit
            posts = subreddit.top(time_filter=time_filter, limit=posts_limit)
            
            for post in tqdm(posts, desc=f"Scraping r/{subreddit_name}", total=posts_limit):
                # skip if this post has already been scraped and skip_existing is True
                if skip_existing and post.id in self.scraped_ids:
                    skipped_posts_count += 1
                    continue
                
                post_data = {
                    'id': post.id,
                    'subreddit': subreddit_name,
                    'title': post.title,
                    'selftext': post.selftext,
                    'score': post.score,
                    'upvote_ratio': post.upvote_ratio,
                    'url': post.url,
                    'created_utc': datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                    'num_comments': post.num_comments,
                    'is_post': True,
                    'parent_id': None,
                    'comment_body': None,
                    'comment_score': None
                }
                
                # add post to data
                data.append(post_data)
                
                # mark as scraped
                self.scraped_ids.add(post.id)
                new_posts_count += 1
                
                # get comments
                post.comments.replace_more(limit=0)  # Skip "load more comments" links
                
                comment_count = 0
                for comment in post.comments.list():
                    if comment_count >= comment_limit:
                        break
                    
                    # skip deleted or removed comments
                    if comment.body in ['[removed]', '[deleted]']:
                        continue
                    
                    comment_data = {
                        'id': comment.id,
                        'subreddit': subreddit_name,
                        'title': None,
                        'selftext': None,
                        'score': None,
                        'upvote_ratio': None,
                        'url': None,
                        'created_utc': datetime.fromtimestamp(comment.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                        'num_comments': None,
                        'is_post': False,
                        'parent_id': post.id,
                        'comment_body': comment.body,
                        'comment_score': comment.score
                    }
                    
                    data.append(comment_data)
                    comment_count += 1
                
                time.sleep(1)
                
            logger.info(f"Subreddit r/{subreddit_name}: {new_posts_count} new posts, {skipped_posts_count} skipped posts")
            
        except Exception as e:
            logger.error(f"Error scraping r/{subreddit_name}: {str(e)}")
        
        df = pd.DataFrame(data)
        logger.info(f"Scraped {len(df)} items from r/{subreddit_name}")
        
        return df
    
    def scrape_nhl_subreddits(self, subreddits=None, time_filter=config.TIME_FILTER, skip_existing=True, save_to_csv=True):
        """
        Scrape multiple NHL-related subreddits.
        
        Args:
            subreddits (list): List of subreddit names to scrape
            time_filter (str): Time filter for posts
            skip_existing (bool): Whether to skip posts that have already been scraped
            save_to_csv (bool): Whether to save the scraped data to CSV
            
        Returns:
            DataFrame: DataFrame containing all scraped data
        """
        subreddits = subreddits or config.NHL_SUBREDDITS
        
        all_data = []
        
        for subreddit in subreddits:
            try:
                df = self.scrape_subreddit(subreddit, time_filter=time_filter, skip_existing=skip_existing)
                all_data.append(df)
                
                # avoid rate limits
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing subreddit r/{subreddit}: {str(e)}")
        
        # ensure no future dupes
        utils.save_scraped_ids(self.scraped_ids)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"Total scraped items: {len(combined_df)}")
            
            # Save to CSV if requested
            if save_to_csv:
                utils.save_to_csv(combined_df, "reddit_raw_data")
                logger.info("Data saved to reddit_raw_data CSV file with timestamp")
            
            return combined_df
        else:
            logger.warning("No data was scraped")
            return pd.DataFrame()
    
    def search_for_team(self, team_name, subreddits=None, limit=100):
        """
        Search for posts about a specific team across NHL subreddits.
        
        Args:
            team_name (str): Name of the team to search for
            subreddits (list): List of subreddit names to search in
            limit (int): Maximum number of posts to retrieve per subreddit
            
        Returns:
            DataFrame: DataFrame containing search results
        """
        subreddits = subreddits or config.NHL_SUBREDDITS
        
        search_results = []
        
        for subreddit in subreddits:
            try:
                logger.info(f"Searching for '{team_name}' in r/{subreddit}")
                
                sr = self.reddit.subreddit(subreddit)
                
                for post in sr.search(team_name, limit=limit):
                    search_results.append({
                        'subreddit': subreddit,
                        'title': post.title,
                        'selftext': post.selftext,
                        'score': post.score,
                        'created_utc': datetime.fromtimestamp(post.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                        'url': post.url,
                        'id': post.id,
                        'num_comments': post.num_comments
                    })
                
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error searching for '{team_name}' in r/{subreddit}: {str(e)}")
        
        if search_results:
            results_df = pd.DataFrame(search_results)
            logger.info(f"Found {len(results_df)} posts containing '{team_name}'")
            return results_df
        else:
            logger.warning(f"No posts found for '{team_name}'")
            return pd.DataFrame()

#  test
if __name__ == "__main__":

    scraper = RedditScraper()
    data = scraper.scrape_nhl_subreddits()
    team_data = scraper.search_for_team("Maple Leafs") 