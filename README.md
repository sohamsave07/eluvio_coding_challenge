# Eluvio Coding Challenge
## Eluvio DS/ML Challenge 2021

Scene Segmentation Using the MovieScenes Dataset

### Requirements:
Run in command line: 
`pip install -r requirements.txt`

### Data:

Download the data from [here](https://drive.google.com/file/d/1oZSOkd4lFmbY205VKQ9aPv1Hz3T_-N6e/view?usp=sharing) and unzip the it in `data_dir` folder, within same directory. The data is stored in `.pkl` files in form dictionary. The data contains:

1. Movie Level: It is IMDB ID of the movie.
2. Shot Level: It has 4 features; place, cast, action and audio. Each feature is 2 dimensional tensor where first feature is number of shots in the movie and second feature is feature is vector, in this case 2048, 512, 512, and 512, respectively.
3. Scene Level: It has ground truth, preliminary predictions and end frame index for each shot.

### Running the Solution:

Predictions are made for the ground truth. Probability is calculated of a shot boundary being a scene boundary. To run `solutions.py`:

1. Make a folder `my_predictions` or whatever you like to name it, without any spaces in it.
2. Run: `python solution.py data_dir my_predictions` in command line.
3. Predicitions will be generated and stored in `my_predictions` folder as pickle files.

### Evaluating the predictions:

To evaluate my prediction, I used `evaluate_sceneseg.py` provided by Eluvio, which can be found in [this repository](https://github.com/eluv-io/elv-ml-challenge). For evaluating the predictions stored in `my_predictions`, run: `python evaluate_sceneseg.py my_predictions`.
The below screenshot is evaluated results I got on my predictions.
![Evaluation Results](https://user-images.githubusercontent.com/42371264/109766011-d9004200-7bb2-11eb-83f4-19754a31fe4b.png)
