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

![alt text](https://github.com/noonanj5atwit/Spam_Email_Detector/blob/main/graphs/Email-Spam-Model-Fit.png?raw=true)


















The image above shows the process of training the model. The accuracy does get to 1 but this is specifically for this data set.

How will the accuracy of the model change if given another dataset?

The evaluate method allows the the model to test it's training so it shows the model unseen data to prove that it's not just memorizing the answers that it used when training.

![alt text](https://github.com/noonanj5atwit/Spam_Email_Detector/blob/main/graphs/Email-Spam-Model-Evaluate.png?raw=true)

## Discussion:
      
      

      
3 Questions to Answer:
1. What patterns or words contribute in the decision of a spam email vs a normal email?
2. How will the model deal with speacial characters and how would that influence the classification of the email?
3. How will the accuracy of the model change if given another dataset?
