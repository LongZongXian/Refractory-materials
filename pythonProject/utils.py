import random

import pandas as pd



def randomOutputData():
    data = pd.read_excel("Data/output-convert.xlsx")

    a = "2183-2082"

    ave = [170.275, 37.8, 222.9, 14.751, 3.187, 200.68, 69.5, 22.2]
    bias = [0.191, 5.374, 0.01, 0.174, 0.005,12.5, 6.1, 1.1]

    #1400,154.345,0.615,40.42,1.556,211.5,0,14.825,0.114,3.18,0.003,217.7,15.9,65.9,3.5,22.6,1.6

    print(len(data))
    for i in range(len(data)):
        if data["image_name"][i].find(a) != -1:
            print(i)

            data["Young‘modulus (GPa)"][i] = random.uniform(ave[0] - bias[0], ave[0] + bias[0])
            data["CMOR (MPa)"][i] = random.uniform(ave[1] - bias[1], ave[1] + bias[1])
            data["CCS (MPa)"][i] = random.uniform(ave[2] - bias[2], ave[2] + bias[2])
            data["Apparent porosity (%)"][i] = random.uniform(ave[3] - bias[3], ave[3] + bias[3])
            data["Bulk density (g/cm3)"][i] = random.uniform(ave[4] - bias[4], ave[4] + bias[4])
            data["fractural energy (J/m2)"][i] = random.uniform(ave[5] - bias[5], ave[5] + bias[5])
            data["characteristic length (mm)"][i] = random.uniform(ave[6] - bias[6], ave[6] + bias[6])
            data["nominal tensile strength (MPa)"][i] = random.uniform(ave[7] - bias[7], ave[7] + bias[7])

    data.to_excel("Data/output-convert.xlsx")
