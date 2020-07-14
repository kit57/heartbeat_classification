
# Heartbeat Classification

Using a dataset containing heartbeat sounds with 5 different sounds: artifact,murmur, extrastole, extrahls and normal. 
With this data we're going to perform a Heartbeat classification with CNN and LSTM neural networks to compare their performance.


## Install dependencies

To install all the necesary dependencies, run in the terminal:

``` pip install - r requirements ```

## Data exploration

You can check Data_exploration Jupyter Notebook to explore the data I will be using in this project. In this chart we can see the
 distribution of the classes in the dataset.

<img src="images/category_chart.png" alt="Display Category Chart">

<br>

Also, you can find different plots where you can check for example the onset strength (check <a href='https://librosa.github.io/librosa/'>librosa</a> python library).
Onset strength computes a spectral flux onset strength envelope. Onset strength at time t is determined by: mean_f max(0, S[f, t] - ref_S[f, t - lag]) where ref_S is S after local max filtering along the frequency axis [1]. By default, if a time series y is provided, S will be the log-power Mel spectrogram.


<img src="images/onset_strength.png" alt="Display Onset Strength">

## Models

You can find the scrips in <i>scrips</i> folder where LSTM, CNN and Random Forest model were created. If you want to run CNN model, you first need
to convert the audio to an image file with an script called "beatsoundtoimage.py" which you can find in utils folder. After converting the audio to images
you need to have the next structure to train the CNN model:
``` 
images
    train
        normal
        murmur
        extrahls
        artifact
        extrastole

    test
        unlabelled
``` 

To do so, you can run "relocatefilesinfolder.py" placed in utils folder.

## Performance comparison

 I have trained an CNN and a LSTM neural network. You can go to LSMT_model jupyter notebook to see all the process.