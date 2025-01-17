# DatasetX
A code library for reading and preprocessing public battery dataset

## 1.简介（Introduction）
本代码库包含6个电池数据集的读取、预处理、双点特征提取和模型预测的python代码库。该代码旨在方便地访问电池数据，并为电池健康管理领域的分析准备数据。

This codebase contains Python code for reading, preprocessing, two-point feature extraction, and model prediction for 6 battery datasets. The code is designed to provide convenient access to battery data and prepare the data for analysis in the field of battery health management.

## 2.数据集（Datasets）
### 2.1 Dataset 1：[https://data.matr.io/1]
This dataset, used in our publication “Data-driven prediction of battery cycle life before capacity degradation”, consists of 124 commercial lithium-ion batteries cycled to failure under fast-charging conditions. These lithium-ion phosphate (LFP)/graphite cells, manufactured by A123 Systems (APR18650M1A), were cycled in horizontal cylindrical fixtures on a 48-channel Arbin LBT potentiostat in a forced convection temperature chamber set to 30°C. The cells have a nominal capacity of 1.1 Ah and a nominal voltage of 3.3 V.


### 2.2 Dataset 2: [https://iastate.figshare.com/articles/dataset/b_ISU-ILCC_Battery_Aging_Dataset_b_/22582234]
The ISU-ILCC battery aging dataset was collected jointly by the System Reliability and Safety Laboratory at Iowa State University (ISU), now the Reliability Engineering and Informatics Laboratory (REIL) at the University of Connecticut, and Iowa Lakes Community College (ILCC). The dataset is designed to study the dependency of battery capacity fade from three stress factors: charge rate, discharge rate, and depth of discharge. The dataset contains cycle aging data from 251 lithium-ion (Li-ion) polymer cells (also called lithium polymer cells) cycled under 63 unique conditions. The current release contains 238 cells; the other 12 cells have not completed the testing (data from those cells will be included in a future release).

### 2.3 Dataset 3: [https://zenodo.org/records/6645536]
Dataset of 88 commercial lithium-ion coin cells cycled under multistage constant current charging/discharging, with currents randomly changed between cycles to emulate realistic use patterns.

raw-data.zip contains the following data:

Variable Discharge: We subject 24 Powerstream LiR2032 coin cells (of nominal capacity 1C = 35mAh) to a sequence of randomly selected charge and discharge currents at room temperature for 110-120 full charge/discharge cycles. Each cycle consists of acquisition of the galvanostatic EIS spectrum, followed by a charging and discharging stage. We collect impedance measurements at 57 frequencies uniformly distributed in the log domain in the range 0.02Hz-20kHz. Charging consists of a two stage Constant Current (CC) protocol; currents are randomly selected in the ranges 70mA-140mA (2C-4C) and 35mA-105mA (1C-3C) in stages 1 and 2 respectively. If the safety threshold voltage of 4.3V is reached before the time limit then charging is stopped. During discharging, a single constant discharge current, randomly selected in the range 35mA-140mA (1C-4C), is applied, until the voltage drops to 3.0V.

Fixed Discharge: We subject an additional 16 Powerstream LiR2032 coin cells (of nominal capacity 1C = 35mAh) to the same cycling conditions as above, except now fixing the discharge current for all cells and cycles at 52.5mA (1.5C) instead of randomly changing the discharge current at each cycle.

### 2.4 Dataset 4: [https://doi.org/10.5281/zenodo.3633835]
Dataset accompanying the paper: "Identifying degradation patterns of lithium ion batteries from impedance spectroscopy using machine learning"

### 2.5 Dataset 5: [https://zenodo.org/records/6405084]
Here are the datasets for the publication named "Data-driven capacity estimation of commercial lithium-ion batteries from voltage relaxation" published in Nature Communications. Experimental cycling data for three commercial 18650 type batteries (Dataset_1:NCA battery, Dataset_2:NCM battery, and Dataset_3:NCM+NCA_battery) are given, where each csv file corresponds to one cell cycling data. The cells are named as CYX-Y_Z-#N according to their cycling conditions. X means the temperature, Y_Z represents the charge_discharge current rate, #N is the cell tag. Each csv file has 9 columns, including cycle time ('time/s'), controlled voltage and current ('control/V/mA'), battery voltage ('Ecell/V'), applied current , charge or discharge electricity ('Q discharge/mA.h' and 'Q discharge/mA.h'), controlled voltage or current ('control/V', 'control/mA' and ), and cycle number ('cycle number'). In the impedance data, one representative cell from each cycling condition is chosen for the discussion in the main text. More detailed descriptions can be found in the zip file.

### 2.6 Dataset 6: [https://mediatum.ub.tum.de/1713382]
In this study, we analyze data collected during the aging of 196 commercial lithium-ion cells with a silicon-doped graphite anode and nickel-rich NCA cathode. The cells are aged over a large range of calendar and cycle aging conditions. For different cells, these conditions are constant, alternating or randomly changing. The total test time and reached cycles are 697 days and 1500 equivalent full cycles for the calendar and cyclic aging tests respectively. A periodic check-up procedure was consistently performed at 20 °C controlled temperature, which allows for good comparability between different aging scenarios. During calendar aging, we quantify the influence of the check-up procedure, which significantly increases the aging observed after 697 days. For certain conditions, the degradation induced by the check-up even exceeds the pure calendar aging, which reveals that the lifetime of lithium-ion batteries was underestimated in previous studies and models.

## 3. 代码结构及用法(Code Structure and Usage)
本代码库包含六个文件夹，分别是Dataset1、Dataset2、Dataset3、Dataset4、Dataset5和Dataset6，每个文件夹下都包含两个.py文件，名称分别为Dataset X-Feature extraction.py和Dataset X-Train and test.py，其中Dataset X-Feature extraction.py的功能是数据的保存与提取，以及对应数据集的双点特征提取方法，Dataset X-Train and test.py的功能是对Dataset X-Feature extraction.py中提取好的双点特征数据采用XGBoost机器学习算法进行模型的训练和测试，并采用MAE、MAPE、RMSE和R2评价方法进行结果评估。

This code repository consists of six folders, namely Dataset1, Dataset2, Dataset3, Dataset4, Dataset5, and Dataset6. Each folder contains two. py files named Dataset X-Feature extraction.py and Dataset X-Train and test. py is used to save and extract data, as well as the corresponding two-point feature extraction method for the dataset. Dataset X-Train and test. py uses XGBoost machine learning algorithm to train and test the extracted two-point feature data in Dataset X-Feature extraction.py, and evaluates it using MAE, MAPE, RMSE, and R2. Method for evaluating results.


