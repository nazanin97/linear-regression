import csv
import numpy as np
from numpy.linalg import inv


def multiply(mat1, mat2, temp):
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                temp[i][j] += float(mat1[i][k]) * float(mat2[k][j])


input = []
featureMatrix = []
output = []

with open('diabetes.csv', 'r') as csvfile:
    csvReader = csv.reader(csvfile)
    for row in csvReader:
        input.append(row)

# read data except row1 and create feature matrix
for r in input[1:]:
    row = [1]
    output.append(r[8])
    for col in r[:8]:
        row.append(col)
    featureMatrix.append(row)

featureMatrix = np.array(featureMatrix)
featureMatrixTranspose = np.transpose(featureMatrix)

output = np.array([output])
output_transpose = np.transpose(output)

temp_coefficient1 = [[0 for x in range(9)] for y in range(9)]
multiply(featureMatrixTranspose, featureMatrix, temp_coefficient1)

temp_coefficient1 = np.array(temp_coefficient1)

temp_coefficient1 = inv(temp_coefficient1)

temp_coefficient2 = [[0 for x in range(len(featureMatrixTranspose[0]))] for y in range(9)]
multiply(temp_coefficient1, featureMatrixTranspose, temp_coefficient2)

coefficient = [[0] for y in range(9)]
multiply(temp_coefficient2, output_transpose, coefficient)

coefficient = np.array(coefficient)
print("Coefficients:\n", coefficient)

output_estimation = [[0] for y in range(len(featureMatrix))]
multiply(featureMatrix, coefficient, output_estimation)

difference = 0
for i in range(len(output_estimation)):
    difference += (output_estimation[i][0] - float(output_transpose[i][0])) ** 2

print("Mean Squared Error: ", difference / 768)

precision = 0
for i in range(len(output_estimation)):
    if output_estimation[i][0] > 0:
        if int(output_transpose[i][0]) == 1:
            precision = precision + 1
    else:
        if int(output_transpose[i][0]) == 0:
            precision = precision + 1

print(precision, "out of 768 outcomes are correctly classified so precision is: ", precision / 768)
print(768 - precision, "out of 768 outcomes aren't correctly classified so error rate is: ", 100 * ((768 - precision) / 768), "%")
