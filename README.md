To run project double click on ‘run.bat’ file to get below output
![image](https://github.com/user-attachments/assets/90fb8403-e301-448e-9a9c-1b97a8b7ec0e)

 
In above screen click on ‘Upload Flight Landing Dataset’ button to upload dataset and get below output
  
In above screen selecting and uploading entire dataset folder with 3 files and then click on ‘Select Folder’ button to load dataset and get below output
![image](https://github.com/user-attachments/assets/5a5b4512-9dff-40aa-a006-ba367ca014b7)

 
In above screen dataset loaded and we can see some records from PILOT and ACTUATOR dataset and you can scroll down above screen text area to view Physical dataset values and in graph x-axis represents type of landing and y-axis represents counts of landing found in dataset. Now close above graph and then click on ‘Preprocess Dataset’ button to normalize, shuffle and split dataset into train and test and get below output
 ![image](https://github.com/user-attachments/assets/3bf3ac3e-5d92-4287-b403-29f8e28d883c)

In above screen we can see total records in dataset and then we can see total features in all dataset and we can see total dataset in pilot and other dataset and then showing training and testing records size. Now train and test data is ready and now click on ‘Run SVM Algorithm’ button to train SVM and get below output
 ![image](https://github.com/user-attachments/assets/f1620306-9cc3-41d6-b30f-4d3b3aeaf605)

In above screen with SVM we got sensitivity as 0.82 and Specificity as 0.55 and in box plot x-axis represents metric names and y-axis represents values. Now close above graph and then click on ‘Run Logistic Regression Algorithm’ button to train logistic regression and get below output ![image](https://github.com/user-attachments/assets/dde65332-df5e-439c-878a-f5e6403f5baa)

 
In above screen with logistic regression we got 0.60% sensitivity values and now click on ‘Run AP2TD Algorithm’ button to train LSTM on ‘Physical Features’ and get below output
 ![image](https://github.com/user-attachments/assets/d2e64fca-1f1f-4f79-ae77-19273d9c3d70)

In above screen with AP2TD physical features we got LSTM sensitivity as 0.92 and specificity as 0.95 and now click on ‘Run AP2DH Algorithm’ to train LSTM on Actuator features and get below output
 ![image](https://github.com/user-attachments/assets/70fc82ae-6301-4bdd-afe1-4822281cb3a8)

In above screen with AP2DH LSTM got 0.99% sensitivity and 0.98 specificity and now click on ‘Run DH2TD Algorithm’ button to train LSTM on PILOT features and get below output
 ![image](https://github.com/user-attachments/assets/7e64d31a-d9ca-42ed-97d5-2260c4ae56f7)

In above screen with DH2TD we got LSTM sensitivity as 0.93 and specificity as 0.92 and now click on ‘Comparison Graph’ button to get below comparison graph
 
In above graph x-axis represents algorithm names and y-axis represents sensitivity and specificity values. Blue bar represents sensitivity and orange bar represents Specificity. In above graph we can see propose AP2TD, AP2DH and DH2TD got high sensitivity and specificity values compare to existing LSTM and logistic Regression.
 
In above screen in last we can see sensitivity and specificity values for HYBRID LSTM by combining all 3 models. For hybrid LSTM we got sensitivity as 0.95 and specificity as 0.96%. This values are closer to value given in base paper
![image](https://github.com/user-attachments/assets/e8a1f085-3689-440d-8e05-71506ac71b0c)

