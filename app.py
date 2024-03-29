import streamlit as st
import json
import tekore as tk
from dotenv import load_dotenv
import os
import pandas as pd
import plotly.express as px
import seaborn as sns
from matplotlib import pyplot as plt
from time import sleep
from sklearn.preprocessing import RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import umap
from datetime import timedelta
from statistics import median


load_dotenv()

st.set_page_config(
    page_title="Clusterfy",
    menu_items={
        "About": "You can find the source code on GitHub at [Excidion/clusterfy](https://github.com/Excidion/clusterfy).",
        "Report a Bug": "https://github.com/Excidion/clusterfy/issues",
    }
)


@st.cache_resource
def login():
    cred = tk.RefreshingCredentials(
        client_id=os.getenv("client_id"),
        client_secret=os.getenv("client_secret"),
        redirect_uri="http://localhost:/callback",
    )
    token = cred.request_client_token()
    spotify = tk.Spotify(token=token)
    return spotify


@st.cache_data
def load_profile(user_id):
    spotify = login()
    user = spotify.user(user_id)
    return user


@st.cache_data(ttl=timedelta(hours=1))
def load_playlists(user_id):
    spotify = login()
    playlists = spotify.playlists(user_id)
    songs = []
    for playlist in playlists.items:
        playlist = spotify.playlist(playlist_id=playlist.id)
        for track in spotify.all_items(playlist.tracks):
            track = track.track
            if track.id is None:
                continue
            audio_features = spotify.track_audio_features(track.id)
            song = {
                "playlist": playlist.name,
                "song_interpret": ", ".join([artist.name for artist in track.artists]),
                "song_title": track.name,
                "acousticness": audio_features.acousticness,
                "danceability": audio_features.danceability,
                "duration": audio_features.duration_ms / 1e3,
                "energy": audio_features.energy,
                "instrumentalness": audio_features.instrumentalness,
                "key": audio_features.key,
                "liveness": audio_features.liveness,
                "loudness": audio_features.loudness,
                "mode": audio_features.mode,
                "speechiness": audio_features.speechiness,
                "tempo": audio_features.tempo,
                "time_signature": audio_features.time_signature,
                "valence": audio_features.valence,
            }
            songs.append(song)
            sleep(0.1)
    songs = pd.DataFrame.from_records(songs)
    return songs


@st.cache_data
def plot_songs(songs, dimensions=2):
    typical_playlist_size = songs.groupby("playlist").size().agg(median)
    pipe = Pipeline(
        [
            ("encoder", ColumnTransformer([("cat", OneHotEncoder(), ["time_signature", "mode", "key"])], remainder="passthrough")),
            ("scaler", RobustScaler()),
            ("umap", umap.UMAP(n_components=dimensions, n_neighbors=round(typical_playlist_size/2), min_dist=0.75, random_state=42)),
        ]
    )
    embedding = pipe.fit_transform(
        X = songs.drop(["playlist", "song_title", "song_interpret"], axis=1),
        y = LabelEncoder().fit_transform(songs["playlist"]),
    )
    songs["x"] = embedding[:, 0]
    songs["y"] = embedding[:, 1]
    if dimensions == 3:
        songs["z"] = embedding[:, 2]
    songs["title"] = songs["song_interpret"] + " - " + songs["song_title"]
    not_in_hover = ["x", "y", "z", "playlist", "title","song_interpret", "song_title"]
    hover_data = {col: (col not in not_in_hover) for col in songs.columns}
    plot_args = dict(
        color="playlist", 
        hover_name="title",
        labels={"x":"", "y":"",},
        hover_data=hover_data,
    )
    if dimensions == 2:
        fig = px.scatter(songs, x="x", y="y", **plot_args)
    elif dimensions == 3:
        plot_args["labels"]["z"] = ""
        fig = px.scatter_3d(songs, x="x", y="y", z="z", **plot_args)
    else:
        raise ValueError("Parameter 'dimensions' has to be either 2 or 3.")
    fig.update_xaxes(tickvals=[], zeroline=False)
    fig.update_yaxes(tickvals=[], zeroline=False)
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            title=None,
        ),
        dragmode = "pan" if dimensions == 2 else None,
    )
    return fig


def plot_distribution(songs, audio_features, x, hue):
    audio_feature = audio_features.get(x)
    if "value_map" in audio_feature.keys():
        sns.countplot(songs, x=x, hue=hue)
        value_map = audio_feature.get("value_map")
        plt.gca().set_xticklabels([value_map.get(str(label), label) for label in plt.gca().get_xticks()])
        plt.ylabel("Number of songs")
    elif "unit" in audio_feature.keys():
        unit = audio_feature.get("unit")
        match unit:
            case "%":
                clip = (0, 1)
            case "s":
                clip = (0, None) # no durations under 0 seconds
            case _:
                clip = None
        sns.kdeplot(songs, x=x, hue=hue, common_norm=False, clip=clip)
        if unit == "%":
            plt.gca().set_xticklabels([f'{x:.0%}' for x in plt.gca().get_xticks()]) 
        else:
            plt.xlabel(f"{x} [{unit}]")
        plt.yticks([])
        plt.ylabel("Share of songs")

    if hue is not None:
        sns.move_legend(
            plt.gca(), 
            "lower center",
            bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False,
        )
    style_pyplot()
    return plt.gcf()

def style_pyplot():
    plt.style.use("dark_background")
    fig = plt.gcf()
    fig.patch.set_facecolor('b')
    fig.patch.set_alpha(0)
    ax = plt.gca()
    ax.patch.set_facecolor('b')
    ax.patch.set_alpha(0)


with open("audio_features.json", "r") as infile:
    audio_features = json.load(infile)


st.title("Clusterfy")
A, B = st.columns(2)

with A:
    user_id = st.text_input("User ID")
    if st.button("Load profile", help="For every 10 songs this will take about 1 second."):
        st.session_state["user_id"] = user_id

user_id = st.session_state.get("user_id")
if user_id in ["", None]:
    pass
else:
    user = load_profile(user_id)
    with B:
        caption = f"Hello {user.display_name}!"
        if len(user.images) != 0:
            st.image(user.images[0].url, caption=caption)
        else:
            st.caption(caption)

    songs = load_playlists(user_id)
    if len(songs) == 0:
        st.warning("No public playlists.")
    else:
        # overall stats
        a, b = st.columns(2)
        with a:
            st.metric("Playlists", songs.playlist.nunique())
        with b:
            st.metric("Songs", len(songs))

        # content
        dist, clust = st.tabs(["Dimensions", "Map of Songs"])
        with dist:
            st.write(
                "For each song in every playlist spotify calculates a set of values.",
                "The following plots show how the values in each dimension are distributed.",
            )
            x = st.selectbox(
                "Dimension", 
                songs.columns.difference(["playlist", "song_title", "song_interpret"]),
            )
            audio_feature = audio_features.get(x)
            st.info(audio_feature.get("description"))
            hue = "playlist" if st.checkbox("Split by playlist") else None
            st.pyplot(plot_distribution(songs, audio_features, x, hue))
            plt.close()
        with clust:
            st.write(
                "This page shows the result of a machine learning algorithm.",
                "The algorithm tries to identify groups of songs based on their values which you can see in the `Dimensions` tab.",
                "The results are then plotted in this map.",
            )
            config = {
                "selectZoom": False,
                'scrollZoom': True,
                "displayModeBar": False,
            }
            dims = st.radio("Map Type", [2, 3], horizontal=True, format_func=lambda x: f"{x}D")
            st.plotly_chart(plot_songs(songs, dims), use_container_width=True, config=config)
