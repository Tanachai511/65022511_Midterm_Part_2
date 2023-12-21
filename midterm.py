from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

File_path = 'C:/study/'
File_name = 'car_data.csv'

df = pd.read_csv(File_path + File_name)
encoder = LabelEncoder()

cols = ['Gender','Purchased']
df[cols] = df[cols].apply(encoder.fit_transform)


df['Age'].fillna(method = 'pad', inplace = True)
df['AnnualSalary'].fillna(method = 'pad', inplace = True)

x = df.iloc[:,0:4]
y = df['Purchased']

model = DecisionTreeClassifier(criterion='entropy')
model.fit(x, y) 

x_pred = ['44','1','35','20000']
    
x_pred_adj = np.array(x_pred).reshape(-1, 4)

y_pred = model.predict(x_pred_adj)
print('Prediction : ', y_pred[0])
score = model.score(x,y)
print('Accuracy : ', '{:.2f}'.format(score))

feature = x.columns.tolist()
Data_class = y.tolist()

plt.figure(figsize =( 25,20))
_ = plot_tree(model,
              feature_names = feature,
              class_names = Data_class,
              label='all',
              impurity = True,
              precision = 3,
              filled = True,
              rounded = True,
              fontsize = 16)
