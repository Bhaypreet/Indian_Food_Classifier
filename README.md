## 🍛 Indian Food Classification using CNN and Pretrained Models
This project focuses on classifying Indian food images into 80 different categories using deep learning techniques. The model is built using Convolutional Neural  Networks (CNN) and further enhanced with pretrained models like EfficientNet, ResNet, and MobileNet for improved accuracy.

## Dataset

   **Total Images**: 4000  
    **Classes**: 80  
    **Split**: 80% Training (3200 images), 20% Validation (800 images)  
    **Image Size**: Resized to (224, 224)  
    #for Custom_CNN it was resized to (128,128) to reduce time  

The dataset is loaded using **image_dataset_from_directory from TensorFlow.**  

## Tools & Libraries Used  

    Python  
    TensorFlow / Keras  
    Pandas, NumPy  
    Matplotlib, Seaborn  


## Model Architectures  

**1. Custom CNN Architecture**  
    
    • Conv2D (32, 3x3) + BatchNorm + MaxPooling + Dropout  
    • Conv2D (64, 3x3) + BatchNorm + MaxPooling + Dropout  
    • Flatten → Dense(256) + BatchNorm + Dropout  
    • Output Layer: Dense with Softmax activation  

**2.Pretrained Models Used :-**
    
      a) EfficientNetB0   
      b) ResNet50  
      c) MobileNetV2  

These models were used with **include_top=False** and **custom classification heads** were added to adapt to the 80-class output.  


**3. **Data Preprocessing**** 

    • Normalization: Pixel values scaled to [0, 1]  
    • Batch size: 32  
    • Shuffled and batched using image_dataset_from_directory  


**4. **Training Strategy****  

  • **Loss Function**: SparseCategoricalCrossentropy  
  • **Optimizer:** Adam  
  • **Callbacks:**  
        a) EarlyStopping (with weight restore)  
        b) ReduceLROnPlateau (dynamic learning rate adjustment)  

  
**5. **Training Parameters****  
    
    • Epochs: 20  
    • Batch Size: 32  
    • Input Image Size: (224, 224, 3)  
    

**6. **Results & Model Comparison****  

   **Model	Train Accuracy**	  
  • EfficientNetB0	~90%	Highest (~60-70%)	Best generalization, least overfitting  
  • ResNet50	~98%	Medium (~55–60%)	Good, but not as strong as EfficientNet  
  • MobileNetV2	~97%	Lower (~50–60%)	Lightweight, faster, but less accurate   
  • Custom CNN	~99%	Low (~23%)	Severe overfitting, weak validation  

**EfficientNet outperformed** all models with the best validation accuracy and stability.  


**7. **Accuracy and Loss Curves****

    • Plot for accuracy and loss against ephochs  

**8. **Model Saving****  

    • Saved every model and deploy using Streamlit  








  
