# Project-1-AI-for-Mechatronics-MCEN90048-
## Introduction
This repository is about the first project **Let us be a Fashion Critic** of AI for Mechatronics (MCEN90048).
The finished tasks are as follow:
1. Train and test a classification model based on Multi-layer perceptron (MLP)
2. Monitor the training process by utilizing tensorboard.
3. Profile the training process by collecting runtime statistics.
4. Randomly Visualize 400 images from the dataset in TensorBoard using t-SNE and PCA.

Besides, the following additional tasks are also done:
1. Compare three gradient methods. (Adam, GradientDescent, Momentum)
2. Compare two regularization methods. (Drop out and Batch Normalization)
## Prerequisites
Make sure the following packages are installed:
1. imageio 2. numpy 3. matplotlib 4. tensorflow
## Get Started
Firstly, download or clone the repository to a local folder.

**Visualization Result**
1. Open Anaconda Prompt, run the following code **tensorboard --logdir=path**. (Path is the path to the visualization checkpoint file,
for example **C:\Users\Administrator\Desktop\Project-1-AI-for-Mechatronics-MCEN90048--master\Results\FashionMNIST\DataVisual**)
2. Open http://localhost:6006 and the results are demonstrated in **PROJECTOR**.

**Monitor training process**
1. Open Anaconda Prompt, run the following code **tensorboard --logdir=path**. (Path is the path to the visualization checkpoint file,
for example, **C:\Users\Administrator\Desktop\Project-1-AI-for-Mechatronics-MCEN90048--master\Results\FashionMNIST\MLPBaseline\Adam_event_01**
2. Open http://localhost:6006 and see the results.

**Profile training process**
1. Open google chrome and direct to **chrome://tracing/**. 
2. Load the **.json** files in the corresponding path. For example, **C:\Users\Administrator\Desktop\Project-1-AI-for-Mechatronics-MCEN90048--master\Results\FashionMNIST\MLPBaseline\Adam_timelines**

**Run Jupyter Notebook**
1. Open the repository and modify the corresponding folder path in **./GeneralTools/misc_fun.py** and other relevant system parameters,
especially DEFAULT_IN, DEFAULT_OUT, DEFAULT_DOWNLOAD and EXCEL_FOLDER.
(**Note: DEFAULT_IN folder should be set to path like ~/data/fashion, or it will automatically download the original MNIST dataset 
which results in the incredible high accuracy of the model.)**
2. Open jupyter Notebook in **./NoteBooks/FashionMNIST.ipynb** and modify the system path in the first cell.
3. Run cells in jupyter Notebook to test the model and see the corresponding results.

## Note
1. The analysis of the results and relevant plots are shown in jupyter Notebook.
2. The results of pretrained model are saved in ~/Results/FashionMNIST folder.
3. Running the cells in jupyter notebook will overwrite parts of the results. If the original results are mis-overwritten, please
redownload the results from github.
4. The relevant figures are saved in ~/Plot/images folder and excel data which are used to generate the figures are saved in ~/Plots/traning_data_excel.

## Author
Qitong Yang 889222 University of Melbourne
