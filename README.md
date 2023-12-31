<div align="center">
<img src="https://www.fondation-mines-telecom.org/wp-content/uploads/2016/01/IMT_Atlantique_logo_RVB_Baseline-1.jpg" width=40%>
</div>

# ML project

<div align="center">

Project in the introduction to the theory and practice of ML course for Mathematical and Computational Engineering specialization at IMTA.

[![Python](https://img.shields.io/badge/Python-3.12.0-yellow?logo=python)](https://www.python.org/)
[![Juyter](https://img.shields.io/badge/Jupyter-grey?logo=jupyter)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange?logo=scikit-learn)](https://scikit-learn.org/stable/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red?logo=pytorch)](https://pytorch.org/)
[![Numpy](https://img.shields.io/badge/Numpy-1.26.0-blue?logo=numpy)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.1.1-violet?logo=pandas)](https://pandas.pydata.org/)
[![GitHub](https://img.shields.io/badge/GitHub-black?logo=github)](https://https://github.com/)

</div>

---

## Authors:
- Jules DECAESTECKER
- Ella FERNANDEZ
- Frédéric LIN
- Jonathan LYS


## Datasets
- [Banknote Authentication Dataset](https://archive.ics.uci.edu/dataset/267/banknote+authentication)

Data was extracted from images of genuine and forged banknote-like specimens. For digitization, an industrial camera commonly employed for print inspection was utilized. The final images measure 400x400 pixels. Grayscale pictures with a resolution of approximately 660 dpi were obtained, taking into consideration the object lens and distance to the investigated object. Features were extracted from the images using the Wavelet Transform tool.

- [Chronic Kidney Disease](https://www.kaggle.com/mansoordaku/ckdisease)

This dataset can be used to predict the chronic kidney disease and it can be collected from the hospital nearly 2 months of period.

## Notes
In order to follow good practices, we have decided to use the following structure for our project:
- `data/` : contains the datasets and is created by the ```setup.py``` script
- `.env` : contains the environment variables (paths)
- `model.py` : contains the custom models (TorchMLP)
- `preprocessing.py` : contains the preprocessing functions (normalization, split)
- `setup.py` : contains the setup functions (create the data folder and solve some format issues)
- `test.ipynb` : contains the main code for the project

## What we do :
In the test.ipynb notebook, we have implemented the following steps:

- Import the datasets
- Clean the data, perform pre-processing
  - Replace missing values by average or median values : median
  - Visualize the data
- Split the dataset
  - Split between training set and test set
  - Split the training set for cross-validation
  - Center and normalize the data. This is done after the split to avoid data leakage.
- Train and test the models with no hyperparameter tuning
- Perform PCA
- Retrain the models via GridSearchCV for hyperparameter tuning

We have decided to modify the method of implementation, for sake of simplicity, each file serves a specific purpose.
The implementation with grid search has led us to make this choice, to separate the models and the preprocessing.
One .py is for the preprocessing, another for the setup (this should be done before everything), and one for the models.
We have incorporated a unit test within the 'model' module of our code. When running python3 model.py, we generate a toy dataset for classification purposes. 

In the end, the best models are the SVC and the Torch MLP.





