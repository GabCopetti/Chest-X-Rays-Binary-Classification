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
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Simple models with different sets of hyperparameters were built to investigate how these affect model training and performance. A simple and shallow multilayer perceptron, using two linear layers and ReLU as activation function, was able to give good predictions of pneumonia using chest X-ray images (accuracy ~93%, F1 score 0.95). This class of model showed to be the most consistent, presenting similar metrics independently of applying class weight in the loss function or changing batch size. Within this simple experiment, it outperformed CNN models, which have a more sophisticated architecture. Further tuning of hyperparameters is needed to make use of the CNN’s full potential [3]. Data augmentation, i.e. generation of artificial data, could help to reduce overfitting and class imbalance [4]. Nonetheless, all models, even simple linear neural networks, were able to give descent predictions with accuracy above 90%. This demonstrates the power that even the less complex neural networks have as image classifiers.
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

- ***Chest_XRays_Data_Analysis.ipynb***: Jupyter notebook in which exploratory data analysis was used to summarize main characteristics, assessing class imabalance and looking for missing data/duplicates.

- ***Chest_XRays_Training_and_Evaluation_BatchSize32.ipynb***: Jupyter notebook in which the first part of the experiment was conducted. Models were trained and evaluated, with batch size 32 in dataloaders.

- ***Chest_XRays_Training_and_Evaluation_BatchSize64.ipynb***: A few models were trained and evaluated using batch size 64.

- ***models*** folder: Contains files with the *state_dict* of the models used in the evaluation with the test subset.

- ***requirements.txt***: Contains the packages used and their versions. 

<br>

## **Setting up environment**

***Please open the notebooks in Google Collaboratory.*** Notebooks contain a "Setting up environment" section which allows the user to install the Python version (3.10), as well as all packages used in this project, directly on the notebook, for better reproducibility. If a different type of installation is desired, a requirements.txt file is also made available here. However, it is not guaranteed that the exact package versions will be available in different channels.
 
