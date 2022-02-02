import pandas as pd
import numpy as np
import os
import sys
import argparse
import time
import json
import pickle
from tqdm import tqdm
from IPython.display import display

def build_causality_matrix(train_annotations):
    '''Build a matrix where each row and column are combinations of verb and nouns, 
    and each cell corresponds to the number of time that column comes right after row.'''

    # Get all unique combinations of verb and noun
    unique_nv_combos = get_unique_nv_combos(train_annotations)

    # Make dict of dict representing causality matrix
    causality_matrix = {}
    for verb_noun in unique_nv_combos:
        causality_matrix[verb_noun] = {}
        for other_verb_noun in unique_nv_combos:
            causality_matrix[verb_noun][other_verb_noun] = 0

    print("Causality matrix initialized")

    # Fill matrix by iterating over train_annotations and counting the number of times each combination appears after another
    for index, row in tqdm(train_annotations.iterrows()):
        if index+1 >= len(train_annotations): break
        verb_noun = row['verb_nouns']
        current_ann_id = int(row['narration_id'].split('_')[-1])
        next_ann_id = int(train_annotations.iloc[index+1]['narration_id'].split('_')[-1])
        print(current_ann_id, next_ann_id)
        if current_ann_id+1 != next_ann_id: continue
        next_verb_noun = train_annotations.iloc[index+1]['verb_nouns']

        if verb_noun == next_verb_noun:
            # Same elements back to back, print index, verb_noun and current_video_id
            print("index", index)
            print("verb_noun, next_verb_noun", verb_noun, next_verb_noun)
            print("current_ann_id", current_ann_id)
            
        causality_matrix[verb_noun][next_verb_noun] += 1

    return causality_matrix


def build_causality_matrix_weighted(train_annotations, weight_fcn, causality_matrix = None):
    '''Build a matrix where each row and column are combinations of verb and nouns, 
    and each cell corresponds to a sum of all verb_nouns that come after, weighted by the weight_fcn.'''

    train_annotations = prepare_EK_df(train_annotations)

    # Get all unique combinations of verb and noun
    unique_nv_combos = get_unique_nv_combos(train_annotations)

    # Make dict of dict representing causality matrix
    if causality_matrix is None:
        causality_matrix = init_causality_matrix(unique_nv_combos)

    print("Causality matrix initialized")

    # Iterate over all unique video_ids and count the number of times each combination appears after another
    for vid, g in tqdm(train_annotations.groupby("video_id")):
        print("looking at group %s with len %d" %(vid,len(g)))
        for i, (idx, row) in enumerate(g.iterrows()):
            # if i+1 >= len(g): print("breaking");break
            verb_noun = row['verb_nouns']
            time1 = row['stop_frame']
            for j in range(i+1, len(g)):
                if row['stop_frame'] > g.iloc[j]['start_frame']+30: # The +30 adds 30 frames of leeway in case an action that does happen afterwards is incorrectly annotated as happening right before the end of the previous one 
                    # print("i, j:", i, j)
                    # print("stop frame: %d, next start_frame: %d" % (row['stop_frame'], g.iloc[j]['start_frame']))
                    # print("current verb_noun: %s, next verb_noun: %s" % (row['verb_nouns'], g.iloc[j]['verb_nouns']))
                    # print("stop frame greater than next start frame")
                    continue
                next_verb_noun = g.iloc[j]['verb_nouns']
                time2 = g.iloc[j]['start_frame']
                causality_matrix[verb_noun][next_verb_noun] += weight_fcn(max(0,time2-time1))

    return causality_matrix

def init_causality_matrix(unique_nv_combos):
    causality_matrix = {}
    for verb_noun in unique_nv_combos:
        causality_matrix[verb_noun] = {}
        for other_verb_noun in unique_nv_combos:
            causality_matrix[verb_noun][other_verb_noun] = 0

    return causality_matrix

def matrix_dict_to_array(cm):
    '''Transforms causality matrix from dictionary to numpy array'''
    array = np.zeros((len(cm), len(cm)))
    for i,vn in enumerate(cm.keys()):
        for j,other_vn in enumerate(cm.keys()):
            array[i][j] = cm[vn][other_vn]
    return array

def add_verb_noun_column(df):
    df["verb_nouns"] = df["verb"] + " " + df["all_nouns"]
    df["verb_nouns"] = df["verb_nouns"].str.replace("[","")
    df["verb_nouns"] = df["verb_nouns"].str.replace("]","")
    df["verb_nouns"] = df["verb_nouns"].str.replace("'","")
    df["verb_nouns"] = df["verb_nouns"].str.replace(",","")
    return df

def add_narration_number(df):
    df["narration_number"] = df["narration_id"].str.split("_").str[-1].astype(int)
    return df

def get_unique_nv_combos(df):
    unique_nv_combos = df["verb_nouns"].unique()

    print("len(unique_nv_combos)",len(unique_nv_combos))
    return unique_nv_combos

def prepare_EK_df(df):

    df = add_verb_noun_column(df)
    # df = add_narration_number(df)

    return df.sort_values(by=['video_id','start_frame'])


if __name__ == "__main__":


    # Parse arguments
    parser = argparse.ArgumentParser(description="Build causality matrix")
    parser.add_argument("--train_annotations", type=str, default='../epic-kitchens-100-annotations/EPIC_100_train.csv', help="Path to train annotations")
    parser.add_argument("--output_path", type=str, default='./caus_mat.pkl', help="Path to output causality matrix")
    args = parser.parse_args()

    # Load train annotations
    train_annotations = pd.read_csv(args.train_annotations)

    # Build causality matrix
    causality_matrix = build_causality_matrix(train_annotations)

    # Save causality matrix
    with open(args.output_path, 'wb') as f:
        pickle.dump(causality_matrix, f)