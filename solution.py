import os
import numpy as np
import pandas as pd
import glob
import pickle
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#This function reads from pickle files, converts it into dataframe to predict
#probability of of a shot boundary being a scene boundary
def make_predictions(data_dir, ans_dir):

    #Concatinating file path with all the .pkl files
    filenames = glob.glob(os.path.join(data_dir, "tt*.pkl"))
    
    for fn in filenames:
    
        x = pickle.load(open(fn, "rb")) #Loading pickle files one by one by iterating over each

        #Declaring dictionary for every key in pickle file
        place_dict = dict()
        cast_dict = dict()
        action_dict = dict()
        audio_dict = dict()
        gt_dict = dict()
        pr_dict = dict()
        shot_to_end_frame_dict = dict()

        #Adding the values of the keys to dictionary. Converting them to Numpy 2D Arrays
        place_dict[x["imdb_id"]] = x["place"].numpy()
        cast_dict[x["imdb_id"]] = x["cast"].numpy()
        action_dict[x["imdb_id"]] = x["action"].numpy()
        audio_dict[x["imdb_id"]] = x["audio"].numpy()
        gt_dict[x["imdb_id"]] = x["scene_transition_boundary_ground_truth"].numpy()
        pr_dict[x["imdb_id"]] = x["scene_transition_boundary_prediction"].numpy()
        shot_to_end_frame_dict[x["imdb_id"]] = x["shot_end_frame"].numpy()

        #Converting dictionaries into different DataFrames
        df1 = pd.DataFrame.from_dict(place_dict[x["imdb_id"]])
        df2 = pd.DataFrame.from_dict(cast_dict[x["imdb_id"]])
        df3 = pd.DataFrame.from_dict(action_dict[x["imdb_id"]])
        df4 = pd.DataFrame.from_dict(audio_dict[x["imdb_id"]])

        #Concatinating different dataframes created above into one dataframe
        df = pd.concat([df1, df2, df3, df4], axis=1)

        df["scene_transition_boundary_ground_truth"] = pd.DataFrame.from_dict(gt_dict[x["imdb_id"]])

        #Finally Dataframe is created. Now we divide the dataframe into Features and Predictors
        X = df[:len(place_dict[x["imdb_id"]])-1].drop(['scene_transition_boundary_ground_truth'],axis=1)
        Y = df[:len(place_dict[x["imdb_id"]])-1]["scene_transition_boundary_ground_truth"].astype(int)

        #Spliting the dataset into Training and Testing
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05, random_state=42)

        #Declaring classifier as Logistic Regression
        clf = LogisticRegression(random_state=0,max_iter=2500).fit(X_train, Y_train)

        #Printing accuracies for each pickle file
        print("Accuracy for IMDB " + x["imdb_id"] + ": " +str(clf.score(X_test, Y_test)))

        #Predicting probalities
        pr = clf.predict_proba(X)

        #Selecting the second element as it tells the probability of the scene transition boundary
        my_pr = []
        for i in range(len(pr)):
            my_pr.append(pr[i][1])

        
        #Converting the Numpy into Tensor
        x['scene_transition_boundary_prediction'] = torch.FloatTensor(my_pr) 
        
        ans_file = ans_dir + "/" + x["imdb_id"] + ".pkl"

        #Saving the results in pickle file
        pickle.dump( x, open( ans_file, "wb" ) )

if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1]
    ans_dir = sys.argv[2]

    make_predictions(data_dir, ans_dir)