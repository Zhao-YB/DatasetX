# DatasetX
A code library for reading and preprocessing public battery dataset

## 1.简介（Introduction）
本代码库包含6个电池数据集的读取、预处理、双点特征提取和模型预测的python代码库。该代码旨在方便地访问电池数据，并为电池健康管理领域的分析准备数据。

This codebase contains Python code for reading, preprocessing, two-point feature extraction, and model prediction for 6 battery datasets. The code is designed to provide convenient access to battery data and prepare the data for analysis in the field of battery health management.

## 2.数据集（Datasets）
### 2.1 Dataset 1：[https://data.matr.io/1]
This dataset, used in our publication “Data-driven prediction of battery cycle life before capacity degradation”, consists of 124 commercial lithium-ion batteries cycled to failure under fast-charging conditions. These lithium-ion phosphate (LFP)/graphite cells, manufactured by A123 Systems (APR18650M1A), were cycled in horizontal cylindrical fixtures on a 48-channel Arbin LBT potentiostat in a forced convection temperature chamber set to 30°C. The cells have a nominal capacity of 1.1 Ah and a nominal voltage of 3.3 V.


### 2.2 Dataset 2: [https://iastate.figshare.com/articles/dataset/_b_ISU-ILCC_Battery_Aging_Dataset_b_/22582234]
The ISU-ILCC battery aging dataset was collected jointly by the System Reliability and Safety Laboratory at Iowa State University (ISU), now the Reliability Engineering and Informatics Laboratory (REIL) at the University of Connecticut, and Iowa Lakes Community College (ILCC). The dataset is designed to study the dependency of battery capacity fade from three stress factors: charge rate, discharge rate, and depth of discharge. The dataset contains cycle aging data from 251 lithium-ion (Li-ion) polymer cells (also called lithium polymer cells) cycled under 63 unique conditions. The current release contains 238 cells; the other 12 cells have not completed the testing (data from those cells will be included in a future release).

