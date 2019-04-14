# Project-1-AI-for-Mechatronics-MCEN90048-
Qitong Yang 889222
## Introduction
This repository is about the first project **Let us be a Fashion Critic** of AI for Mechatronics (MCEN90048).
The finished tasks are as follow:
1. Train and test a classification model based on Multi-layer perceptrons (MLP)
2. Monitor the training process by utilizing tensorboard.
3. Profile the training process by collecting runtime statistics.
4. Randomly Visualize 400 images from the dataset in TensorBoard using t-SNE and PCA.

Besides, the following addtional tasks are also done:
1. Compare three gradient methods. (Adam, GradientDescent, Momentum)
2. Compare two regularization methods. (Drop out and Batch Normalization)

## Get Started
1. Download or clone the repository to a local folder.
2. Open the repository and modify the corresponding folder path in ./GeneralTools/misc_fun.py and relevant system parameters,
especially DEFAULT_IN, DEFAULT_OUT, DEFAULT_DOWNLOAD and EXCEL_FOLDER.
(Note: DEFAULT_IN folder should be set to path like ~/data/fashion, or it will automatically download the original MNIST dataset 
which results in the high accuracy of the model.)
3. Open jupyter Notebook in ./NoteBooks/FashionMNIST.ipynb and modify the system path in first cell.
4. Run cells in jupyter Notebook to test the model and see the corresponding result.

## Note
1. The analysis of the results and relevant plots are shown in jupyter Notebook.
2. The results of pretrained model are saved in ~/Results/FashionMNIST folder. Running the cells in jupyter notebook will overwrite parts of the results.
3. The relevant figures are saved in ~/Plot/images folder and excel data which are used to generate the figures are saved in ~/Plots/traning_data_excel.
