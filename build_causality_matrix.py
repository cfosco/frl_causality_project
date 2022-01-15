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
        verb_noun = row['verb_nouns']
        verb_noun_after = train_annotations.iloc(index+1)['verb_nouns']
        causality_matrix[verb_noun][verb_noun_after] += 1

    return causality_matrix

def add_verb_noun_column(df):
    df["verb_nouns"] = df["verb"] + " " + df["all_nouns"]
    df["verb_nouns"] = df["verb_nouns"].str.replace("[","")
    df["verb_nouns"] = df["verb_nouns"].str.replace("]","")
    df["verb_nouns"] = df["verb_nouns"].str.replace("'","")
    df["verb_nouns"] = df["verb_nouns"].str.replace(",","")
    return df

def get_unique_nv_combos(df):
    df = add_verb_noun_column(df)
    unique_nv_combos = df["verb_nouns"].unique()

    print("len(unique_nv_combos)",len(unique_nv_combos))
    return unique_nv_combos


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import os
    import sys
    import argparse
    import time
    import json
    import pickle
    from tqdm import tqdm

    # Parse arguments
    parser = argparse.ArgumentParser(description="Build causality matrix")
    parser.add_argument("--train_annotations", type=str, default='./epic-kitchens-100-annotations/EPIC_100_train.csv', help="Path to train annotations")
    parser.add_argument("--output_path", type=str, default='./caus_mat.pkl', help="Path to output causality matrix")
    args = parser.parse_args()

    # Load train annotations
    train_annotations = pd.read_csv(args.train_annotations)

    # Build causality matrix
    causality_matrix = build_causality_matrix(train_annotations)

    # Save causality matrix
    with open(args.output_path, 'wb') as f:
        pickle.dump(causality_matrix, f)