# Spam_Email_Detector

## Introduction
For this project I am going to work with a dataset that has contents of emails and are labeled either spam or not spam. Using TensorFlow, a machine learning platform, I plan on developing an accurate classification algorithm to determine whether or not an email is spam or not. Using TensorFlow's ability to process and transform textual data, we will be able to train the model and come up with an accurate classification of the emails. The model will learn the typical contents in a spam email to correctly identify a spam email versus a regular email.

## Objective
   Create a machine learning algorithm that will accurately predict if emails are spam or not.

## Dataset
Email Spam Classfication Dataset (https://www.kaggle.com/datasets/purusinghvi/email-spam-classification-dataset)

The data set has two columns the first one being label where it classifies whether the email is spam or not.
The second column is text where it has the contents of the emails

Label Column
Not Spam = 0
Spam = 1

## Methodology:
  ### Libraries:
      - Tensorflow
      - Sklearn
      - Pandas 
      - Numpy
      - Seaborn
      - Matplotlib
      
  ### Steps:
      1. Import the csv file using the pandas library
      2. More spam than non spam so only took a sample of the dataset (1000 spam and non spam emails)
      3. Use Sklearn's Train_Test_Split algorithm to split the data set into taking some of the sampled dataset to train the model and some to test the model on accuracy. I set it to take 20% of the sampled dataset to test the model and 80% to train the model.
      4. Tokenize the text using tensorflow keras
      5. After you need to take the text and make it into a sequence which takes the text data and makes it into a sequence of integers. This is because neural networks process numerical data.
      6. Pad the sequences which makes sure the sequences have the same length
      7. Create a embedding layer which converts the word index from the tokenizer tp a dense vector.
      8. Flatten layer transforms the 2D array into a 1D array
      9. Configure the model for training using the optimization algorithm "adam". Loss keeps track of the loss and accuracy tells the model to monitor the accuracy of the model.
      10. Train the model on the training data
         - Epochs are the number of times the training dataset is passed through the model
         - Validation_data is what evaluates the models accuracy.
      11. Evalute is giving the model unseen data to make a prediction on the new data and test the true accuracy of the model

## Results:

The Spam Email Detector model went through training that allows it to accurately predict and classify emails as spam vs not spam.

Here are some of the results of the accuracy of the model:



### <ins>Fit</ins>

![alt text](https://github.com/noonanj5atwit/Spam_Email_Detector/blob/main/graphs/Email-Spam-Model-Fit.png?raw=true)


















The image above shows the process of training the model. The accuracy does get to 1 but this is specifically for this data set.

### <ins>Evaluate</ins>

How will the accuracy of the model change if given another dataset?

The evaluate method allows the the model to test it's training so it shows the model unseen data to prove that it's not just memorizing the answers that it used when training.

![alt text](https://github.com/noonanj5atwit/Spam_Email_Detector/blob/main/graphs/Email-Spam-Model-Evaluate.png?raw=true)

The model got an accuracy of 0.9425 (94.25%) from the unseen data. Training it on unseen data reveals the true accuracy of the model.

## Discussion:
The model is really accurate even when it is shown unseen data. The reason why the accuracy was so high when training the model is because it is just memorizing whether the email is spam or not.

### <ins>Model Training</ins>
The only thing that is trickey when training the model is that it see's the data over and over again. That is why the accuracy when training the model was so high and unrealistic. The model was just memorizing the patterns and the answers so it would end up getting 100% accuracy on data it's seen. 

One thing I struggled with at first was the time it took each epoch to train and this was because I was not taking a sample of the data from the dataset. It was going through most of the dataset's emails and took way too long. I fixed this by taking a sample of the dataset (1000 spam/1000 not spam) and used that to significantly decrease the time of each epoch. It now takes 30 - 40 seconds as shown in the picture in results.

For the future, I would like to possibly figure out a way to also scan links or files sent in emails because even if the email passes the spam detection it will get flagged as spam if the email contains malicous content. I think this is a perfect fit for this model since most mail systems scan the emails and files for malicous content. 
