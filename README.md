link to Github: https://github.com/GabCopetti/Chest-X-Rays-Binary-Classification

*This project was carried out during the Deep Learning course at Data ScienceTech Institute (DSTI), as part of the Applied MSc in Data Science & Artificial Intelligence qualification.*

***
# **Binary Classification of Chest X-rays for Prediction of Pneumonia**
***

by Gabriela Copetti


## **Summary**



<p align="center">
<i> All computational steps were performed in Google Collaboratory notebooks, making use of the T4 GPU. <br>
Code was adapted from <a href="https://www.learnpytorch.io/" target="_blank">Daniel Bourke’s PyTorch tutorial</a> [1].
</i>
</p>

<br>
<p align="justify">
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Binary image classification models were built and trained in PyTorch using the <a href="https://huggingface.co/datasets/keremberke/chest-xray-classification" target="_blank">Chest X-Rays Classification Dataset</a>, available in Hugging Face [2]. The dataset is divided into train, validation and testing subsets, with 4077, 1165 and 582 images, respectively. Chest X-rays with NORMAL features are labelled as 0, while chest X-rays with PNEUMONIA features are labelled as 1 (see Figure 1).
</p>


<p align="justify">
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Models with varied degrees of architectural complexity and different sets of hyperparameters were built to investigate how these affect model training and performance.  A simple linear model, containing a single hidden layer with 10 nodes, was able to give good predictions of pneumonia using chest X-ray images (accuracy ~93%, F1 score ~ 95%). Increase in model architectural complexity did not result in significant increase in performance. Multilayer perceptron models, using ReLU as activation function, were the most consistent, presenting similar metrics independently of batch size or class weights. F1 score was slightly higher for the Convolutional Neural Network once L2 regularization was used to prevent overfitting (~96%). However, this increase does not compensate for significantly higher computational cost of CNNs. Further tuning of the great number hyperparameters [3], as well as data augmentation techniques [4], might needed to make use of the CNN’s full potential. Nonetheless, this work demonstrates that even simple neural networks can be powerful image classifiers depending on the task at hand.
</p>
<br>

<p align="center">
  <img 
    width="500" 
    src="https://github.com/user-attachments/assets/843a2f3d-5e60-4754-bbdf-a75593d837fd"
    alt="Chest X-rays">
</p>

<p align="center">
  <i> Fig 1 - Sample chest X-ray images from dataset, with labels 0 (NORMAL) and 1 (PNEUMONIA). </i>
</p>

<br>

> *[1] D. Bourke, "Learn PyTorch for Deep Learning: Zero to Mastery," LearnPyTorch.io. Available: https://www.learnpytorch.io/. [Accessed: Aug. 24, 2024].*
>
> *[2] K. Berke, “Chest-xray-classification,” HuggingFace.co, Feb. 22, 2023. Available: https://huggingface.co/datasets/keremberke/chest-xray-classification. [Accessed: Aug. 24, 2024].*
>
> *[3] F. Thiele, A. J. Windebank, and A. M. Siddiqui, “Motivation for using data-driven algorithms in research: A review of machine learning solutions for image analysis of micrographs in neuroscience,” Journal of Neuropathology & Experimental Neurology, vol. 82, no. 7. Oxford University Press (OUP), pp. 595–610, May 27, 2023. Doi: 10.1093/jnen/nlad040.*
>
> *[4] Amazon Web Services (AWS), "What is data augmentation?" [Online]. Available: https://aws.amazon.com/what-is/data-augmentation/. [Accessed: Aug. 24, 2024].*
 
<br>


## **Files**

- ***Chest_XRays_Classification_Report.pdf*** :a report describing the experiment and the results obtained.

- ***Chest_XRays_Data_Analysis.ipynb***: Google Colab notebook in which exploratory data analysis was used to summarize main characteristics, assessing class imabalance and looking for missing data/duplicates.

- ***Chest_XRays_Training_and_Evaluation_BatchSize32.ipynb***: Google Colab notebook in which the first part of the experiment was conducted. Models were trained and evaluated, with batch size 32 in dataloaders.

- ***Chest_XRays_Training_and_Evaluation_BatchSize64.ipynb***: Google Colab notebook in which a few models were trained and evaluated using batch size 64.

- ***models*** folder: Contains files with the *state_dict* of the models used in the evaluation with the test subset.

- ***requirements.txt***: Contains the packages used and their versions. 

<br>

## **Setting up environment**

***Please open the notebooks in Google Collaboratory.*** Notebooks contain a "Setting up environment" section which allows the user to install the Python version (3.10), as well as all packages used in this project, directly on the notebook, for better reproducibility. If a different type of installation is desired, a requirements.txt file is also made available here. However, it is not guaranteed that the exact package versions will be available in different channels.
 
