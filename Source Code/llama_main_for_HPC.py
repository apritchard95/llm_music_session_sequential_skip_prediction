# -*- coding: utf-8 -*-
"""To actually run this on HPC, amend file paths to HPC storage locations and run using argparse commands --seed and --model_name
"""

import pandas as pd
import numpy as np
import random
import logging
from datetime import datetime
from textwrap import dedent
import argparse

import os
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def set_environment(seed, llm_name):

    # Set the seed and model for the run
    seed = seed
    model = llm_name

    # Define device to use, ideally GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Set Hugging Face token here
    os.environ['HUGGINGFACE_HUB_TOKEN'] = ''
    token = os.environ['HUGGINGFACE_HUB_TOKEN']

    return seed, model, device, token

def load_data(nrows=1500):
    # csv file paths
    listening_log_path = '/content//drive//MyDrive//University of Liverpool//MSc Project//Test Set/'
    track_features_path = '/content//drive//MyDrive//University of Liverpool//MSc Project//Track Features/'

    track_features_csvs = glob.glob(track_features_path + '*.csv')
    testset_prehist_hists = glob.glob(listening_log_path + 'log_pre*.csv')

    # Create listen log dataframes
    listen_log_df = pd.read_csv(testset_prehist_hists[0], nrows=nrows)

    # Create an empty list to hold the track features dataframes
    track_features_dfs = []

    # Loop through each file path and read the CSV file
    for csv_file in track_features_csvs:
        print(f"Loading {csv_file}")
        df = pd.read_csv(csv_file)
        track_features_dfs.append(df)

    # Concatenate all dataframes into a single dataframe
    track_features_df = pd.concat(track_features_dfs, ignore_index=True)

    return track_features_df, listen_log_df, testset_prehist_hists

"""**Load LLMs**"""

def initialise_model(model, token):
    # Instantiate Llama 3.0 model and tokenizer

    if model == 'llama3_0':
        model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

        tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        tokenizer.padding_side = "left"  # Set padding to the left side
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token # Set padding token

        llm = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto", # Automap to available device
            pad_token_id=tokenizer.eos_token_id, # Set the padding token ID
            token=token
        )

    elif model == 'llama3_1':
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

        tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        tokenizer.padding_side = "left"  # Set padding to the left side
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token # Set padding token

        llm = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto", # Automap to available device
            pad_token_id=tokenizer.eos_token_id, # Set the padding token ID
            token=token
        )

    return llm, tokenizer

# Function for model inference - get model response for a given prompt
def get_model_response(prompt, model, tokenizer, max_attempts=3):
    valid_responses = {'True', 'False'}
    attempts = 0

    # Loop until a valid response is returned
    while attempts < max_attempts:
        inputs = tokenizer(prompt, return_tensors='pt', padding=True)
        input_ids = inputs.input_ids.to(model.device)
        outputs = model.generate(
            input_ids,
            attention_mask=inputs['attention_mask'].to(model.device),
            max_new_tokens=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Trim the response to contain only the output
        response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()

        # Check if the response is valid
        if response in valid_responses:
            return response
        else:
            attempts += 1

    # If a valid response is not obtained within max_attempts, return 'Invalid'
    print(f"Warning: Unable to get a valid response after {max_attempts} attempts for this prompt. Model returned: {response}")
    return 'Invalid'

"""**Prompt Templates**"""

def exp_one_0shot_prompt(current_tracks, track_id, track_features_dict):

    # Get the details needed for current track
    feat = get_track_features(track_id, track_features_dict, experiment='exp1')

    return f"""
    Predict whether the next song in the sequence will be skipped by the user, who is listening to music through a streaming service.

    The user has listened to the following tracks in chronological order. Tracks that were skipped are followed by the word 'Skipped', and tracks that were listened to in full are followed by the words 'Not Skipped'.
    {current_tracks}

    Will the user skip the next track in the sequence? Here is the next track:
    {feat}

    Please limit your response to only one word - 'True' if you think the song will be skipped, and 'False' if you think the user will listen to the song. Do not respond with any other words.

    Answer:
    """

def exp_one_oneshot_prompt(current_tracks, num_tracks, track_id, track_features_dict, oneshot_sample_set, seed):

    # Get the details needed for current track
    feat = get_track_features(track_id, track_features_dict, experiment='exp1')

    # Generate one_shot examples and ground truths
    one_shot_examples, one_shot_current_tracks, one_shot_ground_truths = generate_oneshot_examples(oneshot_sample_set, num_tracks, track_features_dict, experiment='exp1')

    return f"""
    Predict whether the next song in the sequence will be skipped by the user, who is listening to music through a streaming service.

    The user has listened to the following tracks in chronological order. Tracks that were skipped are followed by the word 'Skipped', and tracks that were listened to in full are followed by the words 'Not Skipped'.

    Example:
    {one_shot_examples[0]}

    Will the user skip the next track in the sequence? Here is the next track:
    {one_shot_current_tracks[0]}.

    Please limit your response to only one word - 'True' if you think the song will be skipped, and 'False' if you think the user will listen to the song. Do not respond with any other words.

    Answer: {one_shot_ground_truths[0]}

    {current_tracks}

    Will the user skip the next track in the sequence? Here is the next track:
    {feat}

    Please limit your response to only one word - 'True' if you think the song will be skipped, and 'False' if you think the user will listen to the song. Do not respond with any other words.

    Answer:
    """

def exp_two_0shot_prompt(current_tracks, track_id, track_features_dict):

    # Get the details needed for current track
    feat = get_track_features(track_id, track_features_dict, experiment='exp2')

    return f"""
    Predict whether the next song in the sequence will be skipped by the user, who is listening to music through a streaming service. Here are some details about the information you will be given to help you make your decision:
    US Popularity Score: The popularity of a track is a value between 90 and 100, with 100 being the most popular. The popularity is based on the total number of plays the track has had across all users on the streaming service, and how recent those plays are.

    The user has listened to the following tracks in chronological order. Tracks that were skipped are followed by the word 'Skipped', and tracks that were listened to in full are followed by the words 'Not Skipped'.
    {current_tracks}

    Will the user skip the next track in the sequence? Here is the next track:
    {feat}

    Please limit your response to only one word - 'True' if you think the song will be skipped, and 'False' if you think the user will listen to the song. Do not respond with any other words.

    Answer:
    """

def exp_two_oneshot_prompt(current_tracks, num_tracks, track_id, track_features_dict, oneshot_sample_set, seed):

    # Get the details needed for current track
    feat = get_track_features(track_id, track_features_dict, experiment='exp2')

    # Generate one_shot examples and ground truths
    one_shot_examples, one_shot_current_tracks, one_shot_ground_truths = generate_oneshot_examples(oneshot_sample_set, num_tracks, track_features_dict, experiment='exp2')

    return f"""
    Predict whether the next song in the sequence will be skipped by the user, who is listening to music through a streaming service. Here are some details about the information you will be given to help you make your decision:
    US Popularity Score: The popularity of a track is a value between 90 and 100, with 100 being the most popular. The popularity is based on the total number of plays the track has had across all users on the streaming service, and how recent those plays are.

    The user has listened to the following tracks in chronological order. Tracks that were skipped are followed by the word 'Skipped', and tracks that were listened to in full are followed by the words 'Not Skipped'.

    Example:
    {one_shot_examples[0]}

    Will the user skip the next track in the sequence? Here is the next track:
    {one_shot_current_tracks[0]}.

    Please limit your response to only one word - 'True' if you think the song will be skipped, and 'False' if you think the user will listen to the song. Do not respond with any other words.

    Answer: {one_shot_ground_truths[0]}

    {current_tracks}

    Will the user skip the next track in the sequence? Here is the next track:
    {feat}

    Please limit your response to only one word - 'True' if you think the song will be skipped, and 'False' if you think the user will listen to the song. Do not respond with any other words.

    Answer:
    """

def feature_description():
    return dedent("""acousticness: A measure from 0.0 to 1.0 of whether the track is acoustic, with 1.0 indicating high confidence.
    beat_strength: A measure from 0.0 to 1.0 of the beat's prominence, with 1.0 being a strong, distinct beat.
    bounciness: A measure from 0.0 to 1.0 of rhythmic energy, with higher values indicating a livelier rhythm.
    danceability: A measure from 0.0 to 1.0 of how suitable the track is for dancing.
    dyn_range: A measure from 0.0 to 53.0 of the range in loudness, with higher values indicating greater variation.
    energy: A measure from 0.0 to 1.0 of intensity and activity, with higher values indicating more energetic tracks.
    flatness: A measure from 0.0 to ~1.17 of spectral flatness, indicating how noise-like a sound is.
    instrumentalness: A measure from 0.0 to 1.0 predicting the likelihood of a track being instrumental.
    liveness: A measure from 0.0 to 1.0 detecting live performance elements, with higher values suggesting a live recording.
    loudness: The average loudness of a track in decibels (dB), typically ranging between -60 and 0 dB.
    mechanism: A measure from 0.0 to 1.0 of percussive elements, with higher values indicating more mechanical sounds.
    mode: Indicates whether the track is in a major or minor key.
    organism: A measure from 0.0 to 1.0 of organic elements, with higher values suggesting more natural sounds.
    speechiness: A measure from 0.0 to 1.0 detecting spoken words, with higher values indicating more speech-like content.
    tempo: The track’s speed in beats per minute (BPM).
    valence: A measure from 0.0 to 1.0 of the track’s positiveness, with higher values indicating a more positive mood.""")

def session_feat_description():
    return dedent("""Here are some additional details about the session context:
    session_length: This is the number of songs that the user would eventually listen to in total over the course of the session. The songs you are presented with my only be a subsection of this.
    session_postion: This is the position of the song in the session.
    context_switch: A value of True means that the user switched their listening context type when beginning this song, and a value of False means that the user did not switch their listening context type.
    context_type: This is the context pertaining to the type of playlist, collection, or medium that the user is listening to their music through. 'user_collection' means that the user is listening to a playlist they created themself, 'radio' means it came up on the streaming service radio, 'editorial_playlist' indicates that the user is listening to a playlist created by the streaming app editors, and 'catalog' means that the user found this song by searching and browsing the catalog.
    start_reason: This indicates what action caused the song to begin playing on the user's device. 'appload' means that the track was opened by clicking a link from an outside source which led to the app being opened, 'fwdbtn' means that the forward button was pressed on the previous song, 'clickrow' means that the song was specifically selected by clicking or using a touchscreen, 'backbtn' means that the back button was pressed on the previous song, and 'trackdone' means that the previous song was fully played and finished.
    is_premium: A value of True means that a user is a paying subscriber of the streaming service, and a value of False means that the user is not a paying subscriber.
    is_shuffle: A value of True means that a user is playing music in Shuffle mode, and a value of False means that a user is not playing music in Shuffle mode.
    pause_before_play: A value of True means that a user had their music paused before playing the song, and a value of False means that a user did not have their music paused before playing the song.
    seek_forward: A value of True means that a user skipped forward in the song during playback, and a value of False means that a user did not skip forward while listening to the song.
    seek_backward: A value of True means that a user skipped backward in the song during playback, and a value of False means that a user did not skip backward while listening to the song.""")

def exp_three_0shot_prompt(current_tracks, track_id, track_features_dict):

    # Get the details needed for current track
    feat = get_track_features(track_id, track_features_dict, experiment='exp3')

    return f"""
    Predict whether the next song in the sequence will be skipped by the user, who is listening to music through a streaming service. Here are some details about the information you will be given to help you make your decision:
    US Popularity Score: The popularity of a track is a value between 90 and 100, with 100 being the most popular. The popularity is based on the total number of plays the track has had across all users on the streaming service, and how recent those plays are.
    {feature_description()}

    The user has listened to the following tracks in chronological order. Tracks that were skipped are followed by the word 'Skipped', and tracks that were listened to in full are followed by the words 'Not Skipped'.
    {current_tracks}

    Will the user skip the next track in the sequence? Here is the next track:
    {feat}

    Please limit your response to only one word - 'True' if you think the song will be skipped, and 'False' if you think the user will listen to the song. Do not respond with any other words.

    Answer:
    """

def exp_three_oneshot_prompt(current_tracks, num_tracks, track_id, track_features_dict, oneshot_sample_set, seed):

    # Get the details needed for current track
    feat = get_track_features(track_id, track_features_dict, experiment='exp3')

    # Generate one_shot examples and ground truths
    one_shot_examples, one_shot_current_tracks, one_shot_ground_truths = generate_oneshot_examples(oneshot_sample_set, num_tracks, track_features_dict, experiment='exp3')

    return f"""
    Predict whether the next song in the sequence will be skipped by the user, who is listening to music through a streaming service. Here are some details about the information you will be given to help you make your decision:
    US Popularity Score: The popularity of a track is a value between 90 and 100, with 100 being the most popular. The popularity is based on the total number of plays the track has had across all users on the streaming service, and how recent those plays are.
    {feature_description()}

    The user has listened to the following tracks in chronological order. Tracks that were skipped are followed by the word 'Skipped', and tracks that were listened to in full are followed by the words 'Not Skipped'.

    Example:

    {one_shot_examples[0]}

    Will the user skip the next track in the sequence? Here is the next track:
    {one_shot_current_tracks[0]}.

    Please limit your response to only one word - 'True' if you think the song will be skipped, and 'False' if you think the user will listen to the song. Do not respond with any other words.

    Answer: {one_shot_ground_truths[0]}

    {current_tracks}

    Will the user skip the next track in the sequence? Here is the next track:
    {feat}

    Please limit your response to only one word - 'True' if you think the song will be skipped, and 'False' if you think the user will listen to the song. Do not respond with any other words.

    Answer:
    """

def exp_four_0shot_prompt(current_tracks, track_id, session_context, track_features_dict):

    # Get the details needed for current track
    feat = get_track_features(track_id, track_features_dict, experiment='exp4')

    return f"""
    Predict whether the next song in the sequence will be skipped by the user, who is listening to music through a streaming service. Here are some details about the information you will be given to help you make your decision:
    US Popularity Score: The popularity of a track is a value between 90 and 100, with 100 being the most popular. The popularity is based on the total number of plays the track has had across all users on the streaming service, and how recent those plays are.
    {feature_description()}

    {session_feat_description()}

    The user has listened to the following tracks in chronological order. Tracks that were skipped are followed by the word 'Skipped', and tracks that were listened to in full are followed by the words 'Not Skipped'.
    {current_tracks}

    Will the user skip the next track in the sequence? Here is the next track:

    {feat}
    {session_context}

    Please limit your response to only one word - 'True' if you think the song will be skipped, and 'False' if you think the user will listen to the song. Do not respond with any other words.

    Answer:
    """

def exp_four_oneshot_prompt(current_tracks, num_tracks, track_id, session_context, track_features_dict, oneshot_sample_set, seed):

    # Get the details needed for current track
    feat = get_track_features(track_id, track_features_dict, experiment='exp4')

    # Generate one_shot examples and ground truths
    one_shot_examples, one_shot_current_tracks, one_shot_ground_truths = generate_oneshot_examples(oneshot_sample_set, num_tracks, track_features_dict, experiment='exp4')

    return f"""
    Predict whether the next song in the sequence will be skipped by the user, who is listening to music through a streaming service. Here are some details about the information you will be given to help you make your decision:
    US Popularity Score: The popularity of a track is a value between 90 and 100, with 100 being the most popular. The popularity is based on the total number of plays the track has had across all users on the streaming service, and how recent those plays are.
    {feature_description()}

    {session_feat_description()}

    The user has listened to the following tracks in chronological order. Tracks that were skipped are followed by the word 'Skipped', and tracks that were listened to in full are followed by the words 'Not Skipped'.

    Example:

    {one_shot_examples[0]}

    Will the user skip the next track in the sequence? Here is the next track:
    {one_shot_current_tracks[0]}.

    Please limit your response to only one word - 'True' if you think the song will be skipped, and 'False' if you think the user will listen to the song. Do not respond with any other words.

    Answer: {one_shot_ground_truths[0]}

    {current_tracks}

    Will the user skip the next track in the sequence? Here is the next track:

    {feat}
    {session_context}

    Please limit your response to only one word - 'True' if you think the song will be skipped, and 'False' if you think the user will listen to the song. Do not respond with any other words.

    Answer:
    """

def exp_five_0shot_prompt(current_tracks, track_id, session_context, track_features_dict):

    # Get the details needed for current track
    feat = get_track_features(track_id, track_features_dict, experiment='exp5')

    return f"""
    Predict whether the next song in the sequence will be skipped by the user, who is listening to music through a streaming service. Here are some details about the information you will be given to help you make your decision:
    {feature_description()}

    {session_feat_description()}

    The user has listened to the following tracks in chronological order. Tracks that were skipped are followed by the word 'Skipped', and tracks that were listened to in full are followed by the words 'Not Skipped'.
    {current_tracks}

    Will the user skip the next track in the sequence? Here is the next track:

    Unknown Track
    {feat}
    {session_context}

    Please limit your response to only one word - 'True' if you think the song will be skipped, and 'False' if you think the user will listen to the song. Do not respond with any other words.

    Answer:
    """

def exp_five_oneshot_prompt(current_tracks, num_tracks, track_id, session_context, track_features_dict, oneshot_sample_set, seed):

    # Get the details needed for current track
    feat = get_track_features(track_id, track_features_dict, experiment='exp5')

    # Generate one_shot examples and ground truths
    one_shot_examples, one_shot_current_tracks, one_shot_ground_truths = generate_oneshot_examples(oneshot_sample_set, num_tracks, track_features_dict, experiment='exp5')

    return f"""
    Predict whether the next song in the sequence will be skipped by the user, who is listening to music through a streaming service. Here are some details about the information you will be given to help you make your decision:
    {feature_description()}

    {session_feat_description()}

    The user has listened to the following tracks in chronological order. Tracks that were skipped are followed by the word 'Skipped', and tracks that were listened to in full are followed by the words 'Not Skipped'.

    Example:

    {one_shot_examples[0]}

    Will the user skip the next track in the sequence? Here is the next track:
    {one_shot_current_tracks[0]}.

    Please limit your response to only one word - 'True' if you think the song will be skipped, and 'False' if you think the user will listen to the song. Do not respond with any other words.

    Answer: {one_shot_ground_truths[0]}

    {current_tracks}

    Will the user skip the next track in the sequence? Here is the next track:

    Unknown Track
    {feat}
    {session_context}

    Please limit your response to only one word - 'True' if you think the song will be skipped, and 'False' if you think the user will listen to the song. Do not respond with any other words.

    Answer:
    """

"""**Functions for Generating Prompts**"""

# Create a dictionary from the track features data, using track_id as the keys
def create_track_features_dict(track_features_df):

    track_features_dict = track_features_df.set_index('track_id').to_dict(orient='index')
    return track_features_dict

# Create a function to extract a subset of sessions for use in building One-shot examples for prompts
def create_oneshot_session_set(listen_log_df, N=15):

    # Group by session_id and filter for sessions with exactly 10 songs
    session_lengths = listen_log_df.groupby('session_id').size()
    valid_session_ids = session_lengths[session_lengths == 10].index

    # Select the first N valid session IDs
    sample_set_ids = valid_session_ids[:N]

    # Filter the DataFrame to include only those sessions
    sample_set = listen_log_df[listen_log_df['session_id'].isin(sample_set_ids)]

    # Filter the original DataFrame to exclude the extracted sessions
    log_df = listen_log_df[~listen_log_df['session_id'].isin(sample_set_ids)]
    log_df = log_df.reset_index(drop=True)

    return sample_set, log_df

# Function for generating One-shot examples for experiments
def generate_oneshot_examples(sample_set, num_tracks_in_prompt, track_features_dict, num_examples=1, experiment='exp1'):
    one_shot_examples = []
    one_shot_current_tracks = []
    one_shot_ground_truths = []

    # Get unique session IDs from the sample set
    session_ids = sample_set['session_id'].unique()

    # Randomly select session IDs for the One-shot examples
    selected_sessions = random.sample(list(session_ids), num_examples)

    # Iterate over the randomly selected sessions to build the One-shot examples
    for session_id in selected_sessions:
        session_data = sample_set[sample_set['session_id'] == session_id]

        # Generate logs matching the length of the current prompt
        session_log = build_session_log(session_data, track_features_dict, experiment=experiment)

        # Ensure the session_log has enough tracks
        if len(session_log) >= num_tracks_in_prompt:
            one_shot_example = ', '.join(session_log[:num_tracks_in_prompt])

            one_shot_current_track = session_log[num_tracks_in_prompt]
            # Check for " - Skipped" or " - Not Skipped" and split appropriately
            if " - Skipped" in one_shot_current_track:
                one_shot_current_track = one_shot_current_track.split(' - Skipped')[0]
            elif " - Not Skipped" in one_shot_current_track:
                one_shot_current_track = one_shot_current_track.split(' - Not Skipped')[0]

            # Get the ground truth for the current track
            current_ground_truth = session_data.iloc[num_tracks_in_prompt]['skip_2']
            current_ground_truth = 'True' if current_ground_truth == True else 'False'

            # Append the One-shot example, current track, and ground truth to the lists
            one_shot_examples.append(one_shot_example)
            one_shot_current_tracks.append(one_shot_current_track)
            one_shot_ground_truths.append(current_ground_truth)
        else:
            print(f"Session {session_id} does not have enough tracks.")

    return one_shot_examples, one_shot_current_tracks, one_shot_ground_truths

# Function to get the year from a date string
def get_year_from_date(date_str):
    # If the date is already in the format of just a year
    if len(date_str) == 4 and date_str.isdigit():
        return int(date_str)  # Return the year as an integer

    # If the date is in the format 'YYYY-MM'
    elif len(date_str) == 7 and date_str[:4].isdigit() and date_str[5:7].isdigit():
        date_obj = datetime.strptime(date_str, '%Y-%m')
        return date_obj.year

    # If the date is in the format 'YYYY-MM-DD'
    elif len(date_str) == 10 and date_str[:4].isdigit() and date_str[5:7].isdigit() and date_str[8:10].isdigit():
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        return date_obj.year

# Function for getting the track features needed
def get_track_features(track_id, track_features_dict, experiment='exp1'):

    features = {}
    track_details = track_features_dict.get(track_id, {})

    # Get track features from the track features dict
    if experiment != 'exp5':
        features.update({
            'track_name': track_details.get('track_name'),
            'track_artist': track_details.get('artist'),
            'track_album': track_details.get('album_name')
        })

    # Additional features for different experiments
    if experiment != 'exp1' and experiment != 'exp5':
        features.update({
            'track_duration': int(track_details.get('duration')),
            'track_popularity': round(track_details.get('us_popularity_estimate'), 4),
            'release_year': get_year_from_date(track_details.get('release_date_estimate'))
            })

    if experiment == 'exp3' or experiment == 'exp4' or experiment == 'exp5':
        features.update({
        'acousticness': round(track_details.get('acousticness', 0), 2),
        'beat_strength': round(track_details.get('beat_strength', 0), 2),
        'bounciness': round(track_details.get('bounciness', 0), 2),
        'danceability': round(track_details.get('danceability', 0), 2),
        'dyn_range': round(track_details.get('dyn_range_mean', 0), 2),
        'energy': round(track_details.get('energy', 0), 2),
        'flatness': round(track_details.get('flatness', 0), 2),
        'instrumentalness': round(track_details.get('instrumentalness', 0), 2),
        'liveness': round(track_details.get('liveness', 0), 2),
        'loudness': round(track_details.get('loudness', 0), 2),
        'mechanism': round(track_details.get('mechanism', 0), 2),
        'mode': track_details.get('mode'),
        'organism': round(track_details.get('organism', 0), 2),
        'speechiness': round(track_details.get('speechiness', 0), 2),
        'tempo': round(track_details.get('tempo', 0), 2),
        'valence': round(track_details.get('valence', 0), 2)
        })

    return features

# Function for getting session context details
def get_session_context(row):

    session_context = {
        'session_length': row['session_length'] // 2,  # Floor division by 2 as only using the first half of sessions
        'session_position': row['session_position'],
        'context_type': row['context_type'],
        'context_switch': True if row['context_switch'] == 1 else False,
        'start_reason': row['hist_user_behavior_reason_start'],
        'is_premium': row['premium'],
        'is_shuffle': row['hist_user_behavior_is_shuffle'],
        'pause_before_play': True if row['no_pause_before_play'] == 0 else False,
        'seek_forward': True if row['hist_user_behavior_n_seekfwd'] == 1 else False,
        'seek_backward': True if row['hist_user_behavior_n_seekback'] == 1 else False
    }

    return session_context

# Function for building a session log appropriate for the given experiment
def build_session_log(session_data, track_features_dict, experiment='exp1'):

    # Build the prompt for each track in the session
    tracks = []
    for i, row in session_data.iterrows():
        track_id = row['track_id_clean']
        session_position = row['session_position']
        ground_truth = row['skip_2']

        # Additional experiment 4 and 5 session details
        if experiment == 'exp4' or experiment == 'exp5':
            sess_features = get_session_context(row)

        # Get more features from the track features dict
        feat = get_track_features(track_id, track_features_dict, experiment=experiment)

        # Build how the tracks appear in prompt
        if experiment == 'exp1':
          track_info = f"{feat} - {'Skipped' if ground_truth else 'Not Skipped'}\n"
          tracks.append(track_info)
        elif experiment == 'exp2':
          track_info = dedent(f"""
          {feat} - {'Skipped' if ground_truth else 'Not Skipped'}""")
          tracks.append(track_info)
        elif experiment == 'exp3':
          track_info = dedent(f"""
          {feat} - {'Skipped' if ground_truth else 'Not Skipped'}""")
          tracks.append(track_info)
        elif experiment == 'exp4':
          track_info = dedent(f"""
          {feat}
          {sess_features}
           - {'Skipped' if ground_truth else 'Not Skipped'}""")
          tracks.append(track_info)
        elif experiment == 'exp5':
          track_info = dedent(f"""
          Unknown Track
          {feat}
          {sess_features}
           - {'Skipped' if ground_truth else 'Not Skipped'}""")
          tracks.append(track_info)

    return tracks

# Function for producing each prompt by iterating over session data and adding tracks from listening log
def produce_prompts(session_id, session_data, tracks, track_features_dict, oneshot_sample_set, seed, experiment='exp1', shots='zero_shot'):

    prompts = []

    # Ensure that the length of the prompt and examples given are correct and increase by 1 each time
    for i in range(len(tracks) - 1):  # No prompt for the first track in the sequence
        # Get session details for tracks listened to so far and next track
        current_tracks = ', '.join(tracks[:i+1])
        num_tracks = len(tracks[:i+1]) # Record which position in the session we are up to for dynamic One-shot example generation

        next_track_id = session_data.iloc[i+1]['track_id_clean']
        session_position = session_data.iloc[i+1]['session_position']
        next_ground_truth = session_data.iloc[i+1]['skip_2']

        # Get next track feature details from the track features dictionary - Only the details from experiment one necessary for the prompt info
        feat = get_track_features(next_track_id, track_features_dict, experiment='exp1')

        # Generate correct prompt based on experiment name and number of shots
        if experiment=='exp1':
          if shots=='zero_shot':
            prompt = exp_one_0shot_prompt(current_tracks, next_track_id, track_features_dict)
          elif shots=='one_shot':
            prompt = exp_one_oneshot_prompt(current_tracks, num_tracks, next_track_id, track_features_dict, oneshot_sample_set, seed)
        elif experiment=='exp2':
          if shots=='zero_shot':
            prompt = exp_two_0shot_prompt(current_tracks, next_track_id, track_features_dict)
          elif shots=='one_shot':
            prompt = exp_two_oneshot_prompt(current_tracks, num_tracks, next_track_id, track_features_dict, oneshot_sample_set, seed)
        elif experiment=='exp3':
          if shots=='zero_shot':
            prompt = exp_three_0shot_prompt(current_tracks, next_track_id, track_features_dict)
          elif shots=='one_shot':
            prompt = exp_three_oneshot_prompt(current_tracks, num_tracks, next_track_id, track_features_dict, oneshot_sample_set, seed)
        elif experiment=='exp4':
          sess_features = get_session_context(session_data.iloc[i+1]) # Get additional session context features for the next row
          if shots=='zero_shot':
            prompt = exp_four_0shot_prompt(current_tracks, next_track_id, sess_features, track_features_dict)
          elif shots=='one_shot':
            prompt = exp_four_oneshot_prompt(current_tracks, num_tracks, next_track_id, sess_features, track_features_dict, oneshot_sample_set, seed)
        elif experiment == 'exp5':
          sess_features = get_session_context(session_data.iloc[i+1]) # Get additional session context features for the next row
          if shots=='zero_shot':
            prompt = exp_five_0shot_prompt(current_tracks, next_track_id, sess_features, track_features_dict)
          elif shots=='one_shot':
            prompt = exp_five_oneshot_prompt(current_tracks, num_tracks, next_track_id, sess_features, track_features_dict, oneshot_sample_set, seed)

        full_prompt_info = ({
                'session_id': session_id,
                'track_name': feat['track_name'],
                'artist_name': feat['track_artist'],
                'album_name': feat['track_album'],
                'session_position': session_position,
                'ground_truth': next_ground_truth,
                'prompt': prompt.strip()
            })

        prompts.append(full_prompt_info)

    return prompts

# Function for generating the prompts to be used in each experiment
def generate_prompts(listen_log_df, track_features_dict, oneshot_sample_set, seed, experiment='exp1', shots='zero_shot'):

    prompts = []

    # Get unique session IDs
    session_ids = listen_log_df['session_id'].unique()

    for session_id in session_ids:
        # Filter data for the current session
        session_data = listen_log_df[listen_log_df['session_id'] == session_id]

        # Build the prompt for each track in the session
        tracks = build_session_log(session_data, track_features_dict, experiment)

        # Produce each prompt by iterating over session data and adding tracks from listening log, and add these to the prompts list
        prompts.extend(produce_prompts(session_id, session_data, tracks, track_features_dict, oneshot_sample_set, seed, experiment, shots))

    return prompts

# Function for running each experiment
def run_experiment(prompts, model, tokenizer, experiment='exp1', shots='zero_shot'):
    # Store model responses
    responses = []

    # Function for parsing and storing model responses - Make this an outside function
    def append_response(output):
        # Account for catastrophic failure
        if output == 'Invalid':
            skip_prediction = 'Invalid'
        else:
            skip_prediction = True if output == 'True' else False

        # Parse the model's output
        responses.append({
              'session_id': session_id,
              'track_name': track_name,
              'artist_name': artist_name,
              'album_name': album_name,
              'session_position': session_position,
              'skip_prediction': skip_prediction,
              'ground_truth': ground_truth
          })

    # Obtain track data for the responses from the prompts
    for prompt_info in prompts:
        session_id = prompt_info['session_id']
        track_name = prompt_info['track_name']
        artist_name = prompt_info['artist_name']
        album_name = prompt_info['album_name']
        ground_truth = prompt_info['ground_truth']
        session_position = prompt_info['session_position']
        prompt = prompt_info['prompt']

        response = get_model_response(prompt, model, tokenizer)

        # Parse and store the model's output
        append_response(response)

    # Convert the responses to a DataFrame
    responses_df = pd.DataFrame(responses)

    return responses_df

"""**Create and Implement Dummy Classifier**"""

# Function for training a dummy classifier - Finding majority label for each session position in the data
def train_dummy_classifier(session_df):
    majority_labels = {}

    for position in session_df['session_position'].unique():
        labels = session_df[session_df['session_position'] == position]['skip_2']
        majority_label = labels.mode().iloc[0]
        majority_labels[position] = majority_label

    return majority_labels

def predict_dummy_classifier(session_df, majority_labels):
    predictions = []

    for _, row in session_df.iterrows():
        position = row['session_position']
        predictions.append(majority_labels[position])

    return predictions

"""**Save Results for Evaluation**"""

# Function to save results to the correct directory based on experiment, seed and LLM
def save_results(experiment, model, seed, results_df):
    base_path = f'/content/drive/My Drive/ProjectResults2/{model}/Seed_{seed}/{experiment}/'

    # Create directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)

    # Save the results for this experiment
    results_df.to_csv(os.path.join(base_path, f'{experiment}_results.csv'), index=False)

# Function for generating dummy classifier results
def dummy_results(listen_log_df, majority_labels):

    # Generate predictions using the dummy classifier
    listen_log_df['skip_prediction'] = predict_dummy_classifier(listen_log_df, majority_labels)
    # Filter out the first track in each session for evaluation purposes
    dummy_results_df = listen_log_df[listen_log_df['session_position'] > 1]

    return dummy_results_df

"""**Modularisation**"""

# Create data structures

def initialize_data(track_features_df, testset_separate, listen_log_df):
    # Initialise track feature dictionary
    track_features_dict = create_track_features_dict(track_features_df)
    # Extract oneshot sample set from listening log
    oneshot_sample_set, listen_log_df = create_oneshot_session_set(listen_log_df)
    # Train the dummy classifier on seperate data that is not being used for evaluation
    dummy_train_data = pd.read_csv(testset_separate[2], nrows=10000)
    majority_labels = train_dummy_classifier(dummy_train_data)

    return track_features_dict, oneshot_sample_set, listen_log_df, majority_labels

def generate_all_prompts(listen_log_df, track_features_dict, seed, oneshot_sample_set):

    # Generate prompts for all experiments and shot types
    prompts = {}
    experiments = ['exp1', 'exp2', 'exp3', 'exp4', 'exp5']
    shot_types = ['zero_shot', 'one_shot']

    for exp in experiments:
        for shot in shot_types:
            key = f"{exp}_{shot}"
            prompts[key] = generate_prompts(listen_log_df, track_features_dict, oneshot_sample_set, seed, experiment=exp, shots=shot)
            print(f"Generated {len(prompts[key])} prompts for experiment {exp} - {shot}.")

    return prompts

def run_and_save_experiments(seed, prompts, model, llm, tokenizer, listen_log_df, majority_labels):
    results = {}

    for key, prompt_set in prompts.items():
        exp, shot = key.split('_', 1)
        results[key] = run_experiment(prompt_set, llm, tokenizer, exp, shot)
        save_results(key, model, seed=seed, results_df=results[key])
        print(f"Completed experiment {key}.")

    # Generate and save dummy classifier results if necessary
    if seed == 1 and model == 'llama3_0':
        dummy_results_df = dummy_results(listen_log_df, majority_labels)
        save_results('dummy_classifier', 'dummy', seed=1, results_df=dummy_results_df)

    return results

# Function to set up logging
def setup_logging(seed, model_name):
    log_dir = "gs://skip-prediction-bucket//logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create a unique log filename based on seed, model_name, and timestamp
    log_filename = os.path.join(log_dir, f"log_{model_name}_seed_{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    # Set up the logging configuration
    logging.basicConfig(
        filename=log_filename,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info(f"Logging started for model: {model_name} with seed: {seed}")

def main():

    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="LLM-based Music Skip Prediction")

    # Add argument for the seed
    parser.add_argument("--seed", type=int, required=True, help="Random seed for reproducibility")

    # Add argument for the model name
    parser.add_argument("--model_name", type=str, required=True, help="Name of the LLM model (llama3_0 or llama3_1)")

    # Parse the arguments
    args = parser.parse_args()

    # Set up logging
    setup_logging(args.seed, args.model_name)

    # Log the environment settings
    logging.info(f"Using model: {args.model_name} with seed: {args.seed}")

    # Set environment
    seed, model, device, token = set_environment(args.seed, args.model_name)

    # Import the data
    track_features_df, listen_log_df, testset_separate = load_data(1500)

    # Log data loading
    logging.info(f"Data loaded for model {model}, and seed {seed}")

    # Initialise the models
    llm, tokenizer = initialise_model(model, token)

    # Log model initialization
    logging.info(f"Model {model} initialized on device {device}")

    # Initialise and load data
    track_features_dict, oneshot_sample_set, listen_log_df, majority_labels = initialize_data(track_features_df, testset_separate, listen_log_df)

    # Log completion of data structuring
    logging.info(f"Data structures initialised")

    # Generate prompts
    prompts = generate_all_prompts(listen_log_df, track_features_dict, seed, oneshot_sample_set)

    # Log prompt generation
    logging.info(f"Prompts generated for all experiment for model {model}, seed {seed}")

    # Run experiments and save results
    run_and_save_experiments(seed, prompts, model, llm, tokenizer, listen_log_df, majority_labels)

    # Log completion of experiments
    logging.info(f"Experiments completed for model {model}, seed {seed}")

"""**Script Run Order**"""

if __name__ == "__main__":
    main()