import pandas as pd
import numpy as np
import time
import gc
import psutil
import os
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import random
import string

# Stage 1: Data cleaning and profiling
def import_data(file_path):
    """Import data as pandas Dataframe"""
    data_imported = pd.read_csv(file_path, sep = '\t')
    return data_imported

def data_column_rename(data):
    """Rename columns as headers are mixed in the data"""
    data_newcolname = data.rename({'T4' : 'T3', 'T3' : 'T4'}, axis=1)
    return data_newcolname

def data_missmatch (data):
    """Delete rows """
    filtered_data = data.drop(data[
                              (data['T3adjusted'].notnull()) & (data['T3'].isnull()) |
                              (data['T4adjusted'].notnull()) & (data['T4'].isnull()) |
                              (data['T3'].notnull()) & (data['T3adjusted'].isnull()) |
                              (data['T4'].notnull()) & (data['T4adjusted'].isnull())
                              ].index)
    return filtered_data

def replace_with_average(data):
    """Replace any missing values in T3 and T4 with average values for their specific level"""
    T3average_bylevel = data.groupby(['Level'])['T3'].mean()
    T4average_bylevel = data.groupby(['Level'])['T4'].mean()

    for i, row in data.iterrows():
        level = row['Level']
        if pd.isna(row['T3']):
            data.at[i, 'T3'] = T3average_bylevel.loc[level]
        if pd.isna(row['T4']):
            data.at[i, 'T4'] = T4average_bylevel.loc[level]
    return data

def predict_missingvalues(data):
    """Predict missing values for T3adjusted and T4adjusted with linear regression"""
    
    missing_values_T3 = data[data['T3adjusted'].isna()]
    missing_values_T4 = data[data['T4adjusted'].isna()]
    available_values_T3 = data.dropna(subset=['T3adjusted'])
    available_values_T4 = data.dropna(subset=['T4adjusted'])

    x_train_T3 = available_values_T3[['T3']]
    x_train_T4 = available_values_T4[['T4']]
    y_train_T3 = available_values_T3[['T3adjusted']]
    y_train_T4 = available_values_T4[['T4adjusted']]
    
    model_T3 = LinearRegression()
    model_T3.fit(x_train_T3, y_train_T3)

    model_T4 = LinearRegression()
    model_T4.fit(x_train_T4, y_train_T4)
    
    x_missing_T3 = missing_values_T3[['T3']]
    x_missing_T4 = missing_values_T4[['T4']]
    predicted_values_T3 = model_T3.predict(x_missing_T3)
    predicted_values_T4 = model_T4.predict(x_missing_T4)

    data.loc[data['T3adjusted'].isna(), 'T3adjusted'] = predicted_values_T3.reshape(-1)
    data.loc[data['T4adjusted'].isna(), 'T4adjusted'] = predicted_values_T4.reshape(-1)
    
    return data


def descriptive_statistics(data_predicted):
    """Calculate descriptive statistics including count, mean, standard deviation, min and max"""
    n = len(data_predicted['T3'])
    
    Level_count = data_predicted['Level'].count() 
    T3_count = data_predicted['T3'].count()
    T4_count = data_predicted['T4'].count()
    T3adjusted_count = data_predicted['T3adjusted'].count()
    T4adjusted_count = data_predicted['T4adjusted'].count()

    Level_mean = sum(data_predicted['Level']) / n
    T3_mean = sum(data_predicted['T3']) / n 
    T4_mean = sum(data_predicted['T4']) / n 
    T3adjusted_mean = sum(data_predicted['T3adjusted']) / n 
    T4adjusted_mean = sum(data_predicted['T4adjusted']) / n 

    Level_var = sum((item - Level_mean)**2 for item in data_predicted['Level']) / (n - 1)
    T3_var = sum((item - T3_mean)**2 for item in data_predicted['T3']) / (n - 1)
    T4_var = sum((item - T4_mean)**2 for item in data_predicted['T4']) / (n - 1)
    T3adjusted_var = sum((item - T3adjusted_mean)**2 for item in data_predicted['T3adjusted']) / (n - 1)
    T4adjusted_var = sum((item - T4adjusted_mean)**2 for item in data_predicted['T4adjusted']) / (n - 1)

    Level_std = Level_var ** 0.5
    T3_std = T3_var ** 0.5
    T4_std = T4_var ** 0.5
    T3adjusted_std = T3adjusted_var ** 0.5
    T4adjusted_std = T4adjusted_var ** 0.5

    Level_min = data_predicted['Level'].min() 
    T3_min = data_predicted['T3'].min()
    T4_min = data_predicted['T4'].min()
    T3adjusted_min = data_predicted['T3adjusted'].min()
    T4adjusted_min = data_predicted['T4adjusted'].min()

    Level_max = data_predicted['Level'].max() 
    T3_max = data_predicted['T3'].max()
    T4_max = data_predicted['T4'].max()
    T3adjusted_max = data_predicted['T3adjusted'].max()
    T4adjusted_max = data_predicted['T4adjusted'].max()

    summary_table = pd.DataFrame({
        'Level' : [Level_count, Level_mean,Level_std, Level_min, Level_max],
        'T3' : [T3_count, T3_mean, T3_std, T3_min, T3_max],
        'T4' : [T4_count, T4_mean, T4_std, T4_min, T4_max],
        'T3adjusted' : [T3adjusted_count, T3adjusted_mean, T3adjusted_std, T3adjusted_min, T3adjusted_max],
        'T4adjusted' : [T4adjusted_count, T4adjusted_mean, T4adjusted_std, T4adjusted_min, T4adjusted_max]
    }, index = ['count', 'mean', 'std', 'min', 'max'])

    return summary_table


def duplicate_check(data):
    """Identify any repeated rows or confirm that there are none"""

    rows_check = data.duplicated().any()

    if not rows_check:
        print('There are no repeated rows')
    else:
        print('Repeated rows found:')
        print(data[data.duplicated()])


# Stage 2: Time and space complexity

class MatrixMult:
    """Calculate the time and space needed for standard matrix multiplication"""
    def matrix_mult(self, size):
        """Calculate matrix multiplication"""
        intA = np.random.randint(0, 10, (size, size))
        intB = np.random.randint(0, 10, (size, size))
        return np.matmul(intA, intB)

    def timespace_test(self):
        """Perform size and space analysis for matrix multiplication"""
        process = psutil.Process(os.getpid())
        baseRam = process.memory_info().rss
        
        resTime = np.zeros(10)
        resSpace = np.zeros(10)
        
        for i in range(1,11):
            gc.collect()
            start = time.time()
            self.matrix_mult(i*500)
            end = time.time()
            ram = process.memory_info().rss
            resTime[i-1] = end - start
            resSpace[i-1] = ram - baseRam
            
        plt.figure(figsize=(10,5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1000, 11000, 1000), resTime, marker='o', linestyle='-')
        plt.xlabel('Size of Matrix')
        plt.ylabel('Time (seconds)')
        plt.title('Time vs Size of Matrix')
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1000, 11000, 1000), resSpace, marker='o', linestyle='-')
        plt.xlabel('Size of Matrix')
        plt.ylabel('Memory Usage (bytes)')
        plt.title('Memory Usage vs Size of Matrix')
        
        plt.tight_layout() 
        
        print('Calculation time:', resTime)
        print('Calculation space:', resSpace)
        plt.show()

class LoopSorting:
    """Calculate the time and space needed for loop based sorting"""
    def list_sort(self, size):
    
        rand_list=[]
        n= random.randint(0,50000)
        for i in range(n):
            rand_list.append(random.randint(3,9))
    
        min_val = rand_list[0]
    
        for i in range(0, len(rand_list)):
            for j in range(i+1, len(rand_list)):
                if rand_list[i] >= rand_list[j]:
                    rand_list[i], rand_list[j] = rand_list[j],rand_list[i]
                return rand_list

    def timespace_test(self):
        """Perform size and space analysis for loop-based sorting"""
        process = psutil.Process(os.getpid())
        baseRam = process.memory_info().rss
        
        resTime = np.zeros(10)
        resSpace = np.zeros(10)
        
        for i in range(1,11):
            gc.collect()
            start = time.time()
            self.list_sort(i*1000)
            end = time.time()
            ram = process.memory_info().rss
            resTime[i-1] = end - start
            resSpace[i-1] = ram - baseRam
            
        plt.figure(figsize=(10,5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1000, 11000, 1000), resTime, marker='o', linestyle='-')
        plt.xlabel('Size of Integer')
        plt.ylabel('Time (seconds)')
        plt.title('Time vs Size of Integer')
        
        plt.subplot(1, 2, 2)
        plt.plot(range(1000, 11000, 1000), resSpace, marker='o', linestyle='-')
        plt.xlabel('Size of Integer')
        plt.ylabel('Memory Usage (bytes)')
        plt.title('Memory Usage vs Size of Integer')
        
        plt.tight_layout() 
        
        print('Calculation time:', resTime)
        print('Calculation space:', resSpace)
        plt.show()


class SubstringFinder:
    """Calculate and compare the time and space needed for finding a substring in a spring with two different methods"""
    def find_method(self, test_str, test_substr):  
        """Find substring in text using string.find() method"""
        find_result = test_str.find(test_substr)
        return find_result

    def loop_method(self, test_str, test_substr): 
        """Find substring in text using loop method"""
        for i in range(len(test_str) - len(test_substr) + 1):
            if test_str[i:i + len(test_substr)] == test_substr:
                return i
        return -1

    def timespace_test(self):
        """Perform size and space analysis for find and loop methods"""
        process = psutil.Process(os.getpid())
        baseRam = process.memory_info().rss
        
        resTime_find = np.zeros(10)
        resSpace_find = np.zeros(10)
        
        resTime_loop = np.zeros(10)
        resSpace_loop = np.zeros(10)
        
        length = random.randint(0,100000)
        sublength = int(length / 2)
        
        for i in range(1,11):
            gc.collect()
            test_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
            test_substr = ''.join(random.choices(string.ascii_lowercase + string.digits, k = sublength))
            
            start_find = time.time()
            self.find_method(i*test_str, i*test_substr)
            end_find = time.time()
            ram_find = process.memory_info().rss
            resTime_find[i-1] = end_find - start_find
            resSpace_find[i-1] = ram_find - baseRam
            
            start_loop = time.time()
            self.loop_method(i*test_str, i*test_substr)
            end_loop = time.time()
            ram_loop = process.memory_info().rss    
            resTime_loop[i-1] = end_loop - start_loop
            resSpace_loop[i-1] = ram_loop - baseRam

        plt.figure(figsize=(10,5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1000, 11000, 1000), resTime_find, marker='o', linestyle='-')
        plt.plot(range(1000, 11000, 1000), resTime_loop, marker='x', linestyle='-')
        plt.xlabel('Size of Substring')
        plt.ylabel('Time (seconds)')
        plt.title('Time vs Size of Substring')
    
        plt.subplot(1, 2, 2)
        plt.plot(range(1000, 11000, 1000), resSpace_find, marker='o', linestyle='-')
        plt.plot(range(1000, 11000, 1000), resSpace_loop, marker='x', linestyle='-')
        plt.xlabel('Size of Substring')
        plt.ylabel('Memory Usage (bytes)')
        plt.title('Memory Usage vs Size of Substring')
        plt.tight_layout()
    
        print('Calculation time (find method):', resTime_find)
        print('Calculation space (find method):',resSpace_find)
    
        print('Calculation time (loop method):', resTime_loop)
        print('Calculation space (loop method):',resSpace_loop)
