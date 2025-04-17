"""
This program is a streamlit program to display similar subreddits
Author: Roman Zalewski
"""
from requests import get
import re
import os
import pymongo
import praw
from sentence_transformers import SentenceTransformer
import json
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import random


def getsubreddits(amount:int=1, filter = False) -> list[str]:
    """
    Extract subreddit names through web scraping. Many subreddits use the word "porn" to describe something
    satisfying such as "r/foodporn" being nice looking food. Hence, the filter boolean for more "academic" results.
    :param amount: amount of subreddits to scrape, by 250 (1=250, 2=500, etc...)
    :param filter: Filter out the word porn
    :return:
    """
    res = []
    text = ""
    for i in range(amount):
        text = text + get(f"https://www.reddit.com/best/communities/{i+1}/").text
    for itm in re.findall(r'(?<=id="\/r\/)[^\/]+', text):
        print(itm)
        res.append(itm)
    if filter:
        res = [sub for sub in res if "porn" not in sub.lower()]
    with open('subreddits.json', 'w') as outfile:
        json.dump(res, outfile)
    return res


@st.cache_resource
def pull_from_mongo(subreddits: list[str]) -> tuple[dict, list]:
    """
    Pull from mongo collection
    :param subreddits: list of subreddits
    :return: The sentence transforms for each headline as a dictionary,
    then a list with the 2d vectors used for plotting transformed with
    TSNE. Indices match.
    """
    mongo_uri = os.environ["MONGO_SECRET"]
    client = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=15000)
    db = client['db5']
    collection = db['posts']
    transforms = {}
    model = SentenceTransformer('all-MiniLM-L12-v2')
    for sub in subreddits:
        temp = []
        for res in collection.find({'subreddit': sub}):
            temp.append(model.encode(res['title'], normalize_embeddings=True))
            print(f"appended {res['title']} from {res['subreddit']}")
        mean_vector = []
        for i in range(len(temp[0])):
            total = []
            for vec in temp:
                total.append(vec[i])
            mean_vector.append(float(sum(total) / len(temp[0])))
        transforms[sub] = mean_vector
    tsne = TSNE(n_components=2, random_state=42, perplexity=25)
    data_array = np.array(list(transforms.values()))
    data_2d = tsne.fit_transform(data_array).tolist()
    return transforms, data_2d


def compare(transforms: dict, vec1:str, vec2:str) -> int:
    """
    Simple Dot product
    :param transforms: Data to draw from
    :param vec1: Subreddit 1
    :param vec2: Subreddit 2
    :return: int showing similarity
    """
    return sum([transforms[vec1][i] * transforms[vec2][i] for i in range(len(transforms[vec1]))])

@st.cache_data
def visualize(transforms: dict, vec_2d: list, *args) -> plt.figure:
    """
    Visualizes data using matplotlib
    :param transforms: Data to draw from
    :param vec_2d: vectors to plot
    :param args: subreddits to plot
    :return: a matplotlib figure
    """
    labels = transforms.keys()
    fig, ax = plt.subplots()
    for i in range(len(args)):
        vector = vec_2d[list(labels).index(args[i])]
        ax.scatter(vector[0], vector[1])
        ax.annotate(args[i], xy=(vector[0], vector[1]))
        xs = [v[0] for v in vec_2d]
        ys = [v[1] for v in vec_2d]
    ax.set_xlim(ax.get_xlim()[0] - 5, ax.get_xlim()[1] + 5)
    ax.set_ylim(ax.get_ylim()[0] - 5, ax.get_ylim()[1] + 5)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig

@st.cache_data
def cluster(transforms: dict, data_2d: list) -> plt.figure:
    """
    Clusters points using sklearn
    :param transforms:
    :param data_2d:
    :return: figure showing the clusters
    """
    labels = transforms.keys()
    c = KMeans(n_clusters=5)
    cols = ['r', 'g', 'b', 'y','orange']
    out_data = c.fit_predict(list(transforms.values()))
    fig, ax = plt.subplots()
    for i in range(len(out_data)):
        ax.scatter(data_2d[i][0], data_2d[i][1], c=cols[out_data[i]])
        if i%7 == 0:
            ax.annotate(list(labels)[i], xy=(data_2d[i][0], data_2d[i][1]))
    fig.savefig("clustering.png")
    return fig

def ui(transforms:dict, vec_2d: list, subreddits: list, sampledefault:int = 10) -> None:
    """
    Uses streamlit to display a UI
    :param subreddits: subreddits to be drawn from
    :param transforms: Data to draw
    :param vec_2d: 2d vectors based on transforms
    :param sampledefault: Number of points to plot by default before user input
    :return: None
    """
    st.title("Subreddit Similarity")
    with st.container():
        defaults = random.sample(subreddits, sampledefault)
        options = st.sidebar.multiselect('Subreddits', subreddits)
        print(defaults)
        print(options)
        if len(options) == 0:
            plot = visualize(transforms, vec_2d, *defaults)
        else:
            plot = visualize(transforms, vec_2d, *options)
        st.pyplot(plot, dpi=600)
    st.divider()
    with st.container():
        st.header("Compare two subreddits directly")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Subreddit 1")
            choice1 = st.selectbox("Subreddit 1", options=subreddits)
        with col2:
            st.subheader("Subreddit 2")
            choice2 = st.selectbox("Subreddit 2", options=subreddits)
        sim = compare(transforms, choice1, choice2)
        st.markdown(f"<h2 style='text-align: center'>{round(sim*100,4)} similarity rate!</h2>", unsafe_allow_html=True)
    st.divider()
    with st.container():
        st.header("Clustering the top 100 subreddits")
        st.pyplot(cluster(
            {k: v for i, (k, v) in enumerate(transforms.items()) if i < 100},
            vec_2d[:100]
        ))

def main() -> None:
    """
    Main function
    :return: None
    """
    if not os.path.exists('subreddits.json'):
        subreddits = getsubreddits(4, filter=True)
        with open('subreddits.json', 'w') as outfile:
            json.dump(subreddits, outfile)
    else:
        with open("subreddits.json", "r") as json_file:
            subreddits = json.load(json_file)
    if not os.path.exists('transforms.json'):
        transforms, vec_2d = pull_from_mongo()
        with open('transforms.json', 'w') as outfile:
            json.dump(transforms, outfile)
    else:
        with open("transforms.json", "r") as json_file:
            transforms, vec_2d = json.load(json_file)
    cluster(transforms, vec_2d)
    ui(transforms, vec_2d, subreddits, sampledefault=10)


if __name__ == '__main__':
    main()