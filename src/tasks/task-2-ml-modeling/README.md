W2.2 Prepare/ Explore ML approaches  
W3.1 Explore the pre-trained network and ML models  
W3.2 CNN training and validation  
W3.3 Performing transfer learning  
W4.1 System testing and accuracy reporting  


# Model Finetuning and Results:

*Goal:  Implementation of Binary Image Classification with CNNs using the Crack/Groove dataset and Rut/Subsidence dataset. 
*Classes :  Crack/Groove and Rut/Subsidence 
*Used a balanced dataset of 2250:2250 for Crack/Groove and Used a balanced dataset of 199:199 for Rut/Subsidence.  
*Test Data has a total of 450 samples for Crack/Groove with 225 for each class and a total of 40 samples for Rut/Subsidence with 20 for each class.

* Crack/Groove Dataset: [link](https://drive.google.com/drive/folders/1pY0Yaevl3AL0s_g8QVT5Vj7DFo6Fl4qq)
<img src = https://github.com/OmdenaAI/uae-chapter-road-inspection/blob/main/src/tasks/task-2-ml-modeling/Assets/Dataset(Cracks%20%2BGroove).jpg>

* Rut/Subsidence Dataset: [link](https://drive.google.com/drive/folders/1NYd1OtEC2FSRMoY6QbrVQ7bP6JaQmBcf)
<img src = https://github.com/OmdenaAI/uae-chapter-road-inspection/blob/main/src/tasks/task-2-ml-modeling/Assets/Dataset(Rut%2BSubsidence).jpg>

## ResNet152V2
Notebook : [link](https://github.com/OmdenaAI/uae-chapter-road-inspection/blob/main/src/tasks/task-2-ml-modeling/crack_groove.ipynb)  
Model Weight: [link](https://https://drive.google.com/drive/folders/136XhbrlrGz42r7j_AX_7kGoTJl4j6kvj)
<img src = https://github.com/OmdenaAI/uae-chapter-road-inspection/blob/main/src/tasks/task-2-ml-modeling/Assets/ResNet152V2(Crack%2BGroove%20Dataset)-Architecture.jpg>
<img src = https://github.com/OmdenaAI/uae-chapter-road-inspection/blob/main/src/tasks/task-2-ml-modeling/Assets/ResNet152V2(Crack%2BGroove%20Dataset).jpg>

Fine Tuning:   
   * Number of epochs trained -10 
   * Batch size – 102  
   * Train Validation split: 80:20  
   * Optimizer- adam
   * Dataset was randomly shuffled  
   * Framework : Tensorflow

