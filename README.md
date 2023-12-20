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

## Datasets
- [Banknote Authentication Dataset](https://archive.ics.uci.edu/dataset/267/banknote+authentication)
- [Chronic Kidney Disease](https://www.kaggle.com/mansoordaku/ckdisease)

## What we do :
- Import the dataset
- Clean the data, perform pre-processing
  - Replace missing values by average or median values : median
  - Center and normalize the data
- Split the dataset
  - Split between training set and test set
  - Split the training set for cross-validation
- Train the model (including feature selection) : PCA
- Validate the model

We have decided to modify the method of implementation, to simplify it and have one main function for each .py file.
The implementation with grid search has led us to make this choice.
One .py is for the preprocessing, another for the setup, and one for the models.
We have incorporated a unit test within the 'model' module of our code. When running python3 model.py, we generate a toy dataset for classification purposes. We also perform feature visualization and selection.






