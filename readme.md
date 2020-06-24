
**Training the ML model directly from firebase storage:**

Setting up FireBase Storage

1.  Go to [https://firebase.google.com/](https://firebase.google.com/)
    
2.  Click on Get Started
    
3.  Click Add Project
    
4.  Enter a Project Name and Create a Project
    
5.  On the left hand tab click on Storage
    
6.  Click Get Started and Create a Storage bucket
    
7.  In Project Overview select either iOS, Android, or Web depending on your apps platform
    
8.  Depending on which platform you selected, copy the configuration (highlighted)
    

  
![](https://lh5.googleusercontent.com/mQ60iILMazfRZkvwSJ1GRckn15a4Qh4lyElR4OcpclFsTKoc-ETL8SJyF-NZD8uUu5C8XnV50F-vDWdVOuEauy42ZA9-HlqDaj01_X0PPMECGOvozcwY5anbLNjRl-ohKmp-Hrh_)  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  

9.  Replace the config variable in files extract_embeddings.py and recognize_video.py with the config you received from part 8
    
10.  Click on Project settings
    

  
![](https://lh5.googleusercontent.com/SdUNH957yyksV0wX3Hyz6OmwojPS72blLmYxtCurXKYgYwvAqIH64ZsA2IsR7xmFDHQFcgjbrB3nDB8sLfDAVMcWaYR46NKoEB5cd8cCiQmbcEG_y4JP-IDEVHvhHiFlNhjq4HUu)  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  

11.  Click on service accounts
    
12.  Select python and generate new private key
    

  
![](https://lh5.googleusercontent.com/u8xhUjHDrbm6P4zUapN5c0B2bPxVFwBvUOAt8rK699mMC-wsyTjV0nUXz30xIZQHcmwG-hMuMJYcaOdvc5_MRI9u5biYglm0kjBN0SblDVeuhy3qfUJ8Tc9kTM2QGb1RBwYiyndi)  
  
  
  
  
  
  
  
  
  
  
  
  
  

13.  Put the downloaded .json file into the main directory where the code is (by default it should be opencv-face-recognition)
    
14.  Inside the config variable for both python files, add “serviceAccount” as the key, and the .json file name as the value.
    

  
![](https://lh4.googleusercontent.com/GLJeX4FzIiVNrZr4Q9QozYQ4LNKKkoLb1I8OONj9tqympqPsoD5OUJOZYJUt-Mk599zAr4wOhXbt5glZ48U7gY2B9akZ8nxa9n-w7oGn2Lc_jEPs3ZqGi3mjhukWc3202QKT6AG8)  
  
  
  
  
  
  
  
  
  
  
  

15.  Now your ML model is connected to firebases’ database and storage
    

  

Storage Format:

Inside your storage, the format of files should be as follows:

-A list of directories where each directory name will be the name of a person which will be used as the label in the ML model

-Inside each directory, will be the a list of images of the person whose name is on the directory

  

*Disclaimer*: There must be an Unknown directory in storage, that can be filled with images of random individuals ![](https://lh5.googleusercontent.com/tiw6Ua447bOEluQ_9pR2R0EzzpXanWtkJEpXtyLl9Pb-VTgl7KMxbviQM3db4tGeREZy2FxZwf_9_1PHDpntd2YhJka2XbakfWj3AcTbU80-AllrlBGufXVoWEKB6rfEL7bSLRFr)

Setting up the Database

1.  Similar to how you set up your storage, on the left hand tab of your firebase console, select Database and click create database (Create a Cloud Firestore DB and not a Realtime Database).
    
2.  Inside the collection “users”, make sure you create document with the name ‘Unknown’. You can set the entry and exit notifications to whatever you would like.
    
3.  In the file recognize_video.py, on like 170, replace the current .json file path with the one you have downloaded (similar to what you did for storage)
    

  
![](https://lh3.googleusercontent.com/3R9313fpdvhoQTbW061UfEfqxgqBsdpC9ixdeiuqVR3euxgx-j66iflnLhKEnKKTgGMsaPb1mOFfbtAz0JmmwoJR1_l4A8EePBCoVntZesB_Yfe6ArVU9-kNSogxmalyPwHoQvfA)  
  
  
  
  

Running the code

1.  In extract_embeddings.py make sure lines 90-157 (#reading from google cloud storage) are not commented. Make sure lines 163-203 (#reading from local directory) are commented.
    
2.  If you are using an IDE you can run the file run.py, or if you are using terminal, you can run python3 run.py
    

  
  
  

  
  
  
  
  

Training the ML model locally:

  

*Disclaimer*: If you were to train the ML model from images locally, during the facial recognition phase there will be an error, due to the fact that the python script will search for your name in the DB. However, if you did not want to use the DB, and still wanted to use the facial recognition model for your own purposes, you can follow the steps below.

  

1.  In extract_embeddings.py make sure lines 90-157 (#reading from google cloud storage) are commented. Make sure lines 163-203 (#reading from local directory) are not commented.
    
2.  In the file recognize_video.py, make sure line 321 (sendtoDB(name_list, 0)) is commented, and make sure lines 313 and 314 (the two following lines of code) are not commented. This will print the name of the user that the facial recognition model recognizes in the console.
