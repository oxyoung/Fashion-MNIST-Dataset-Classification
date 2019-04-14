import xlrd
import os
import matplotlib.pyplot as plt
from GeneralTools.misc_fun import FLAGS
FLAGS.EXCEL_FOLDER = 'C:/Users/oxyoung/Desktop/Project1/Plots/traning_data_excel/'
def plot_dif_regularization_acc():
    Excel_folder = FLAGS.EXCEL_FOLDER
    file_location1 = os.path.join(Excel_folder, "Adam.xlsx")
    file_location2 = os.path.join(Excel_folder, "batch.xlsx")
    file_location3 = os.path.join(Excel_folder, "dropout.xlsx")
    workbook1 = xlrd.open_workbook(file_location1)
    workbook2 = xlrd.open_workbook(file_location2)
    workbook3 = xlrd.open_workbook(file_location3)
    first_sheet1 = workbook1.sheet_by_index(0)
    first_sheet2 = workbook2.sheet_by_index(0)
    first_sheet3 = workbook3.sheet_by_index(0)
    x1 = [first_sheet1.cell_value(i, 1) for i in range(1, 61)]
    y1 = [first_sheet1.cell_value(i, 3) for i in range(1, 61)]
    x2 = [first_sheet2.cell_value(i, 1) for i in range(1, 61)]
    y2 = [first_sheet2.cell_value(i, 3) for i in range(1, 61)]
    x3 = [first_sheet3.cell_value(i, 1) for i in range(1, 61)]
    y3 = [first_sheet3.cell_value(i, 3) for i in range(1, 61)]
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title('Test Accuracy with Different Regularization Methods')
    l1 = plt.plot(x1, y1)
    l2 = plt.plot(x2, y2, color='red', linewidth=1.0, linestyle='--')
    l3 = plt.plot(x3, y3, linestyle=':',)
    plt.legend(labels=['None', 'Batch Normalization', 'dropout'],  loc='best')
    plt.grid()
    plt.show()

def plot_dif_regularization_loss():
    Excel_folder = FLAGS.EXCEL_FOLDER
    file_location1 = os.path.join(Excel_folder, "Adam.xlsx")
    file_location2 = os.path.join(Excel_folder, "batch.xlsx")
    file_location3 = os.path.join(Excel_folder, "dropout.xlsx")
    workbook1 = xlrd.open_workbook(file_location1)
    workbook2 = xlrd.open_workbook(file_location2)
    workbook3 = xlrd.open_workbook(file_location3)
    first_sheet1 = workbook1.sheet_by_index(0)
    first_sheet2 = workbook2.sheet_by_index(0)
    first_sheet3 = workbook3.sheet_by_index(0)
    x1 = [first_sheet1.cell_value(i, 1) for i in range(1, 61)]
    y1 = [first_sheet1.cell_value(i, 9) for i in range(1, 61)]
    x2 = [first_sheet2.cell_value(i, 1) for i in range(1, 61)]
    y2 = [first_sheet2.cell_value(i, 9) for i in range(1, 61)]
    x3 = [first_sheet3.cell_value(i, 1) for i in range(1, 61)]
    y3 = [first_sheet3.cell_value(i, 9) for i in range(1, 61)]
    plt.xlabel('epochs')
    plt.ylabel('Cross Entropy')
    plt.title('The Loss of Test Data with Different Regularization Methods')
    l1 = plt.plot(x1, y1)
    l2 = plt.plot(x2, y2, color='red', linewidth=1.0, linestyle='--')
    l3 = plt.plot(x3, y3, linestyle=':',)
    plt.legend(labels=['None', 'Batch Normalization', 'dropout'],  loc='best')
    plt.grid()
    plt.show()

def plot_dif_optimizer():
    Excel_folder = FLAGS.EXCEL_FOLDER
    file_location1 = os.path.join(Excel_folder, "GD.xlsx")
    file_location2 = os.path.join(Excel_folder, "Adam.xlsx")
    file_location3 = os.path.join(Excel_folder, "Momentum.xlsx")
    workbook1 = xlrd.open_workbook(file_location1)
    workbook2 = xlrd.open_workbook(file_location2)
    workbook3 = xlrd.open_workbook(file_location3)
    first_sheet1 = workbook1.sheet_by_index(0)
    first_sheet2 = workbook2.sheet_by_index(0)
    first_sheet3 = workbook3.sheet_by_index(0)
    x1 = [first_sheet1.cell_value(i, 1) for i in range(1, 61)]
    y1 = [first_sheet1.cell_value(i, 3) for i in range(1, 61)]
    x2 = [first_sheet2.cell_value(i, 1) for i in range(1, 61)]
    y2 = [first_sheet2.cell_value(i, 3) for i in range(1, 61)]
    x3 = [first_sheet3.cell_value(i, 1) for i in range(1, 61)]
    y3 = [first_sheet3.cell_value(i, 3) for i in range(1, 61)]
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title('Test Accuracy with Different Optimizer')
    l1 = plt.plot(x1, y1)
    l2 = plt.plot(x2, y2, color='red', linewidth=1.0, linestyle='--')
    l3 = plt.plot(x3, y3, linestyle=':',)
    plt.legend(labels=['GradientDescent', 'Adam','Momentum'],  loc='best')
    plt.grid()
    plt.show()


if __name__ == "__main__":
    plot_dif_optimizer()
    plot_dif_regularization_loss()
    plot_dif_regularization_acc()