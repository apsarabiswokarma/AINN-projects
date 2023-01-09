
import pandas as pd

"""# Load the dataset"""

#Create the dataset
#create empty data frame
golf_df = pd.DataFrame()

#add outlook
golf_df['Outlook'] = ['rainy', 'rainy', 'overcast', 'sunny', 'sunny', 'sunny', 
                     'overcast', 'rainy', 'rainy', 'sunny', 'rainy', 'overcast',
                     'overcast', 'sunny']

#add temperature
golf_df['Temperature'] = ['hot', 'hot', 'hot', 'mild', 'cool', 'cool', 'cool',
                         'mild', 'cool', 'mild', 'mild', 'mild', 'hot', 'mild']

#add humidity
golf_df['Humidity'] = ['high', 'high', 'high', 'high', 'normal', 'normal', 'normal',
                      'high', 'normal', 'normal', 'normal', 'high', 'normal', 'high']

#add windy
golf_df['Windy'] = ['false', 'true', 'false', 'false', 'false', 'true', 'true',
                   'false', 'false', 'false', 'true', 'true', 'false', 'true']

#finally add play
golf_df['Play'] = ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 
                  'yes', 'yes', 'no']


#Print/show the new data
print(golf_df)

import math
a=5
b=9
#E(a,b)

entropy = -((a/(a+b))*math.log2(a/(a+b)))-((b/(a+b))*math.log2(b/(a+b)))
print(entropy)

"""# Import Libraries"""

## import dependencies
from sklearn import tree #For our Decision Tree
import pandas as pd # For our DataFrame
import pydotplus # To create our Decision Tree Graph
from IPython.display import Image  # To Display a image of our graph

"""# Data Preprocessing"""

# Convert categorical variable into dummy/indicator variables or (binary vairbles) essentialy 1's and 0's
# I chose the variable name one_hot_data bescause in ML one-hot is a group of bits among which the legal combinations of values are only those with a single high (1) bit and all the others low (0)
one_hot_data = pd.get_dummies(golf_df[ ['Outlook', 'Temperature', 'Humidity', 'Windy'] ])
#print the new dummy data
one_hot_data

"""# Define and train the model"""

# The decision tree classifier.
model = tree.DecisionTreeClassifier()
# Training the Decision Tree
model_train = model.fit(one_hot_data, golf_df['Play'])

"""# Visualize the model"""

# Export/Print a decision tree in DOT format.
# print(tree.export_graphviz(clf_train, None))

#Create Dot Data
dot_data = tree.export_graphviz(model_train, out_file=None, feature_names=list(one_hot_data.columns.values), 
                                class_names=['Not_Play', 'Play'], rounded=True, filled=True) #Gini decides which attribute/feature should be placed at the root node, which features will act as internal nodes or leaf nodes
#Create Graph from DOT data
graph = pydotplus.graph_from_dot_data(dot_data)

# Show graph
Image(graph.create_png())

"""# Make prediciton"""

# Test model prediction input:
# Outlook = sunny,Temperature =  hot, Humidity = normal, Windy = false
prediction = model_train.predict([[0,0,1,0,1,0,0,1,1,0]])
prediction

