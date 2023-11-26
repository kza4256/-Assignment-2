Assignment-2

This code is a machine learning project, for image classification using a convolutional neural network (CNN) with TensorFlow and Keras. Let me break down the main components and explain the output plots.
full blog is found here: https://toomajust.wixsite.com/khitam-aqel1

    Import Libraries and Setup:

    a. Make sure to install: i. !pip install -q keras-core --upgrade
    ii. !pip install -q keras-nlp --upgrade
    iii. !pip install --upgrade -q wandb git+https://github.com/soumik12345/wandb-addons
    b. Importing necessary libraries: NumPy, Pandas, import keras_nlp, import keras_core as keras, import keras_core, backend as K and TensorFlow.
    c. It sets up the data directory and loads the dataset using TensorFlow's TPU (Tensor Processing Unit) strategy if available.

    Data Preprocessing:
    a. load and data creation, combination and split.
    b. Import necessary libraries mentioned in 1..
    c. The preprocessor converts input strings into a dictionary of preprocessed tensors by tokenizing the input and converting it into a sequence of token IDs. This simplifies the data and makes it easier for models to
    understand. Padding is used to make all sequences the same length, boosting computational efficiency.

    Model Definition: Import tensorflow.keras import layers, models
    Use a Naive Bayes Classifier to predict whether the essay is written by a human or an LLM (presumably a language model).

    Training:
    The model is trained using the training dataset and validated on the validation dataset. The training history is stored.
    The training is done for 5 epochs, and the accuracy and loss are printed for each epoch.

    Hyperparameter Tuning:
    The code then explores different learning rates (0.001, 0.01, and 0.1) and trains the model for each learning rate.
    The training and validation accuracy and loss for each learning rate are printed.

    Plotting: import matplotlib.pyplot as plt
    Finally, the code uses Matplotlib to plot the training and validation accuracy over epochs for the original training and the learning rate exploration.

