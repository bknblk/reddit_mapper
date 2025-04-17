"""
Code for pulling data into dataset. Not used in program since database is already populated.
"""
import main as orig
import pymongo
import praw
import os
import json

def insert_into_mongo(subreddits:list[str], post_per_sub:int=50) -> None:
    """
    Insert into mongo collection. No usages in program.
    :param subreddits: list of subreddit names to collect from
    :param post_per_sub: How many headlines per subreddit to scrape
    :return None
    """
    mongo_uri = os.environ["MONGO_SECRET"]
    client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=15000)

    reddit = praw.Reddit(
        client_id='lyy6mwO8koaoLkZSkNhwmg',
        client_secret= os.environ['REDDIT_CLIENT_SECRET'],
        user_agent='testscript',
    )
    try:
        db = client['db5']
        collection = db['posts']
        inp = []
        for sub in subreddits:
            for post in reddit.subreddit(sub).top(limit=post_per_sub):
                print(f"Title: {post.title}, subreddit: {sub}")
                inp.append({'title': post.title, 'subreddit': sub})
        collection.insert_many(inp)
    except OSError as e:
        print(e)

def main():
    """
    main file
    :return: None
    """
    if not os.path.exists('subreddits.json'):
        with open('subreddits.json', 'w') as outfile:
            json.dump(orig.getsubreddits(4), outfile)