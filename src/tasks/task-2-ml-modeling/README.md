W2.2 Prepare/ Explore ML approaches  
W3.1 Explore the pre-trained network and ML models  
W3.2 CNN training and validation  
W3.3 Performing transfer learning  
W4.1 System testing and accuracy reporting  


# Model Finetuning and Results:

* Goal:  Implementation of Binary Image Classification with CNNs using the Crack/Groove dataset and Rut/Subsidence dataset. 
* Classes :  Crack/Groove and Rut/Subsidence 
* Used a balanced dataset of 2250:2250 for Crack/Groove and Used a balanced dataset of 199:199 for Rut/Subsidence.  
* Test Data has a total of 450 samples for Crack/Groove with 225 for each class and a total of 40 samples for Rut/Subsidence with 20 for each class.
* Crack/Groove Dataset: [link](https://drive.google.com/drive/folders/1pY0Yaevl3AL0s_g8QVT5Vj7DFo6Fl4qq)
* Rut/Subsidence Dataset: [link](https://drive.google.com/drive/folders/1NYd1OtEC2FSRMoY6QbrVQ7bP6JaQmBcf)
* Total number of models tried for both Crack/Groove and Rut/Subsidence:
<img src = https://github.com/OmdenaAI/uae-chapter-road-inspection/blob/main/src/tasks/task-2-ml-modeling/Assets/Dataset(Cracks%20%2BGroove).jpg>
<img src = https://github.com/OmdenaAI/uae-chapter-road-inspection/blob/main/src/tasks/task-2-ml-modeling/Assets/Dataset(Rut%2BSubsidence).jpg>

![Screenshot 2022-11-28 192847](https://user-images.githubusercontent.com/81407869/204431760-7c664701-d898-4a94-8dc3-040ad9a7cb70.jpg)

# Crack and Groove Models:

## ResNet152V2 - 97.56% Accuracy
* Notebook : [link](https://github.com/OmdenaAI/uae-chapter-road-inspection/blob/main/src/tasks/task-2-ml-modeling/crack_groove.ipynb)  
* Model Weight: [link](https://https://drive.google.com/drive/folders/136XhbrlrGz42r7j_AX_7kGoTJl4j6kvj)
<img src = https://github.com/OmdenaAI/uae-chapter-road-inspection/blob/main/src/tasks/task-2-ml-modeling/Assets/ResNet152V2(Crack%2BGroove%20Dataset)-Architecture.jpg>
<img src = https://github.com/OmdenaAI/uae-chapter-road-inspection/blob/main/src/tasks/task-2-ml-modeling/Assets/ResNet152V2(Crack%2BGroove%20Dataset).jpg>

Fine Tuning:   
   * Number of epochs trained -10 
   * Batch size – 102  
   * Train Validation split: 80:20  
   * Optimizer- adam
   * Dataset was randomly shuffled  
   * Framework : Tensorflow


## EfficientNet - 98.23% Accuracy
* Notebook : [link](https://github.com/OmdenaAI/uae-chapter-road-inspection/blob/main/src/tasks/task-2-ml-modeling/EfficientNetB0_Cracks%26Grooves.ipynb) 
* Model Weight: [link](https://drive.google.com/drive/folders/1vervas1xrMARrclvcjploHMB-eUaNKVb)

![Screenshot 2022-11-28 183740](https://user-images.githubusercontent.com/81407869/204425058-588b225b-e093-4381-bfcc-78e2937dff3f.jpg)
* <img src = https://github.com/OmdenaAI/uae-chapter-road-inspection/blob/main/src/tasks/task-2-ml-modeling/Assets/EfficientNetB0(Cracks%2BGroove%20Dataset).jpg>

Fine Tuning  

   * Number of epochs trained -50(monitored val accuracy, used early stopping and model checkpoints)  
   * Batch size -32  
   * Train validation split :80-20.  
   * Optimizer -opt, Learning rate - 0.001, #stepsize :nb_validation_samples = len(validation_generator.filenames)   
   * Dataset was randomly shuffled  
   * Framework: Tensorflow

## VGG11 - 96.00% Accuracy

* Notebook : [link](https://github.com/OmdenaAI/uae-chapter-road-inspection/blob/main/src/tasks/task-2-ml-modeling/VGG_11_Crack%2BGroove.ipynb) 
* Model Weight: [link](https://drive.google.com/drive/folders/1k4TUgo1A8dlS2DU2YIWjlwuDSUHS4a8w)

![Screenshot 2022-11-28 185127](https://user-images.githubusercontent.com/81407869/204427079-6705b943-20aa-4d84-aee6-ee4c1262bbc8.jpg)
* <img src = https://github.com/OmdenaAI/uae-chapter-road-inspection/blob/main/src/tasks/task-2-ml-modeling/Assets/VGG11_BN(Crack%2BGroove%20Dataset).jpg>

Fine Tuning  
   * Number of epochs trained -50
   * Batch size -32  
   * Train validation split :80-20.  
   * Optimizer -Adam, Learning rate - 0.01, #stepsize :len(train_dataloader)   
   * Dataset was randomly shuffled  
   * Framework: Tensorflow
   
## VGG16 - 96.60% Accuracy
* Notebook : [link](https://github.com/OmdenaAI/uae-chapter-road-inspection/blob/main/src/tasks/task-2-ml-modeling/VGG16_Crack%2BGroove.ipynb) 
* Model Weight: [link](https://drive.google.com/drive/folders/1rGTHeKmX8A1kEvI02Dgv5CHZWgFQI_3B)

<img src = https://github.com/OmdenaAI/uae-chapter-road-inspection/blob/main/src/tasks/task-2-ml-modeling/Assets/VGG16(Crack%2BGroove%20Dataset)Learning_Curve.jpg>
<img src = https://github.com/OmdenaAI/uae-chapter-road-inspection/blob/main/src/tasks/task-2-ml-modeling/Assets/VGG16(Crack%2BGroove%20Dataset).jpg>

Fine Tuning  

   * Number of epochs trained -100
   * Batch size -64  
   * Train validation split :80-20.  
   * Optimizer -RMSprop, Learning rate - 1e-5, #stepsize :int(val_test_generator/batch_size)  
   * Framework: Tensorflow
   
   ## InceptionV3 - 97.78% Accuracy
* Notebook : [link](https://github.com/OmdenaAI/uae-chapter-road-inspection/blob/main/src/tasks/task-2-ml-modeling/InceptionV3_cracks_vs_grooves.ipynb) 
* Model Weight: [link](https://drive.google.com/drive/folders/1saHLHlq8PDb6NH9xfFHUrAG1casSH51c)

 ![Screenshot 2022-11-28 190735](https://user-images.githubusercontent.com/81407869/204429049-0c72470e-4643-404a-be76-179318f4a763.jpg)
  <img src = https://github.com/OmdenaAI/uae-chapter-road-inspection/blob/main/src/tasks/task-2-ml-modeling/Assets/InceptionV3(Crack%2BGroove%20Dataset).jpg>

Fine Tuning  

   * Number of epochs trained -100(monitored val accuracy and reduced learning rate, used early stopping and model checkpoints) 
   * Batch size -64  
   * Train validation split :80-20.  
   * Optimizer -SGD, Learning rate - 0.01, stepsize = len(validation_generator)
   * Framework: Tensorflow

# Rut and Subsidence Models:

## MobileNetV2 - 97.5%

* Notebook : [link](https://github.com/OmdenaAI/uae-chapter-road-inspection/blob/main/src/tasks/task-2-ml-modeling/rut_subsidence.ipynb) 
* Model Weight: [link](https://drive.google.com/drive/folders/1pCmmwMR3cVuZ-h1E-aqm9Wk5Q7OmM2xi)

 <img src = https://github.com/OmdenaAI/uae-chapter-road-inspection/blob/main/src/tasks/task-2-ml-modeling/Assets/MobileNetV2(Rut%2BSubsidence%20Dataset)-Architecture.jpg>
  <img src = https://github.com/OmdenaAI/uae-chapter-road-inspection/blob/main/src/tasks/task-2-ml-modeling/Assets/MobileNetV2(Rut%2BSubsidence%20Dataset).jpg>

Fine Tuning  

   * Number of epochs trained -10(monitored val accuracy and reduced learning rate, used early stopping and model checkpoints) 
   * Batch size -32 
   * Train validation split :80-20.  
   * Optimizer -adam,stepsize = len(val_ds)
   * Dataset was randomly shuffled  
   * Framework: Tensorflow


## EfficientNetB0 - 97.5%

* Notebook : [link](https://github.com/OmdenaAI/uae-chapter-road-inspection/blob/main/src/tasks/task-2-ml-modeling/EfficientNetB0_Rut%26Subsidence.ipynb) 
* Model Weight: [link](https://drive.google.com/drive/folders/1_xVcI1FbudRP03YEGtVTaseeZj2JCsaL)

  ![Screenshot 2022-11-28 203535](https://user-images.githubusercontent.com/81407869/204439903-8e757c6a-245a-4c15-8b67-4b9d20426b02.jpg)
  <img src = https://github.com/OmdenaAI/uae-chapter-road-inspection/blob/main/src/tasks/task-2-ml-modeling/Assets/EfficientNet(Rut%2BSubsidence%20Dataset).jpg>
  

Fine Tuning  

   * Number of epochs trained -100(monitored val accuracy, used early stopping and model checkpoints)  
   * Batch size -32  
   * Train validation split :80-20.  
   * Optimizer -opt, Learning rate - 0.001, #stepsize :nb_validation_samples = len(val_gen)   
   * Dataset was randomly shuffled  
   * Framework: Tensorflow

   
## VGG16 - 96.60% Accuracy
* Notebook : [link](https://github.com/OmdenaAI/uae-chapter-road-inspection/blob/main/src/tasks/task-2-ml-modeling/VGG16_Rut%2BSubsidence.ipynb) 
* Model Weight: [link](https://drive.google.com/drive/folders/1-HGmRloHKyJ-dtEunePeNoqi_llSq8or)

<img src = https://github.com/OmdenaAI/uae-chapter-road-inspection/blob/main/src/tasks/task-2-ml-modeling/Assets/VGG16(Rut%2BSubsidence_%20Dataset)Learning_Curve.jpg>
<img src = https://github.com/OmdenaAI/uae-chapter-road-inspection/blob/main/src/tasks/task-2-ml-modeling/Assets/VGG16(Rut%2BSubsidence%20Dataset).jpg>

Fine Tuning  

   * Number of epochs trained -100
   * Batch size -64  
   * Train validation split :80-20.  
   * Optimizer -RMSprop, Learning rate - 1e-5, #stepsize :int(val_test_generator/batch_size)  
   * Framework: Tensorflow
