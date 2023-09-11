# The goal of this activity is to use a decision tree machine learning algorithm
# to train a model to predict the customer's category.
#
# We have already used these data and this scenario with our K-Nearest Neighbors algorithm.
#
# The company assigns customers to five loyalty categories, which are the following:
# 1. Loyal customers: Customers who make up a minority of the customer base
# but who generate a large portion of sales.
# 2. Impulse customers: Customers who do not have a specific product in mind
# and purchase goods when it seems good at the time.
# 3. Discount customers: Customers who shop frequently but base buying decisions
# primarily on markdowns.
# 4. Need-based customers: Customers with the intention of buying a specific product
# based on a specific need at the time.
# 5. Wandering customers: Customers who are not sure of what they want to buy. These
# customers may be convinced to make purchases based on compelling advertisements.
#
# For this activity, the data are contained in a json file.
# The variables available are the following:
# 1. custid: This field is the unique id of each customer. This field is not required for our KNN model.
# 2. category: This field is the customer category. It can be one of five categories (loyal, impulse,
# discount, need-based, or wandering.
# 3. householdincome: This field represents the total household income for all members of the customers
# household. This field might be missing for some documents.
# 4. householdsize: This field is the total number of adults living in the customer's home.
# 5. educationlevel: This field is the total number of years of education that the customer has completed.
# 6. gender: This field represents the customers gender.

# Target Leakage: When we observe the target before we observe the features

# Do we have a potential target leakage problem if we are attempting to predict loyalty groups?
# Target leakage is when the features are observed or measured after the target that we are
# attempting to predict. From the above description of the available data elements, we cannot
# make a determination. However, most of our features seem fairly stable meaning they probably do not
# change from month-to-month. Therefore, can we assume that household income, household size, education
# level, and customer's gender were the same last month (or last period) as they are now? That seems
# reasonable for these particular features. Therefore, we probably do not have a target leakage problem
# with constructing our predictive model.

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import researchpy as rp
import os

# Turn off interactive charting. Therefore, we will have to save all visuals to a file to view them.
matplotlib.use('Agg')
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 100)
# I will also set the console output settings for my numpy arrays to facilitate viewing of the numpy arrays.
np.set_printoptions(suppress=True, threshold=5000, edgeitems=10)

# We could turn off warnings. To do so, we could execute the following two statements.
# For this example, I am going to keep the warning messages on, so I will keep these statements commented.
# import warnings
# warnings.filterwarnings('ignore')

# Task 1
# Read in the data from the json file ("customer_loyalty.json") and create a pandas Data Frame.
full_path = os.path.join(os.getcwd(), "customer_loyalty.json")
# full_path = "customer_loyalty.json"
my_df = pd.read_json(full_path)
# Let's take a quick look at our data (rows and columns)!
# We can use the head, tail or sample functions to grab a random sample of instances/examples/observations.
# Why are we doing this step and what are we looking for with this output?
# Here we are just taking a look to see if our data appears to misaligned in any way.
# For instance, did my sales data get loaded in the id column?
print(my_df.sample(5))
print(my_df.head(5))
print(my_df.tail(5))
print(my_df.info())

# Task 2
# Check for unique custid's to ensure we have no data leakage when training and testing our models.
print(my_df['custid'].is_unique)
print(len(my_df['custid'].unique()))
print(my_df[['custid']].info())
# NOTE: We see that we have 7167 non-null values in the custid value and 7167 unique values.
# Therefore, each row in the DataFrame represents a unique customer. Given that our unit of analysis is
# individual customers, we should not have any data leakage with our data moving forward.
# We could set the index to be the custid field. However, I am not going to do this task because
# having the defaults will help me later when reporting out the results.
# my_df = my_df.set_index('custid')
# print(my_df.sample(5))

# Task 3
# Check for nulls. We see that we have 8 nulls in the "householdincome" field.
# After consultation with the team, we decide to just remove those rows from the
# data frame.
print(my_df.isnull().sum())
my_df.dropna(inplace=True)
# my_df = my_df.dropna()
print(my_df.isnull().sum())
# Due to the nulls, the data type of householdincome may have been an object.
# As a result, let's change it to a numeric field using pd.to_numeric().
my_df['householdincome'] = pd.to_numeric(my_df.householdincome)
print(my_df.sample(5))
print(my_df.dtypes)

# Task 4
# Let's change our Males to 0s and our Females to 1s. We can do this because we only have two unique values
# in the gender field.
print(my_df['gender'].unique())
my_df['gender'] = my_df['gender'].map({'Male': 0, 'Female': 1})
print(my_df.sample(2))

# Task 5
# Let's print a panel of descriptive statistics for our data. Our target is the categories
# so let's group the data by categories when printing these descriptive statistics.
my_fields = ['category', 'householdincome', 'householdsize', 'educationlevel', 'gender']
print(my_df[my_fields].describe(include='all'))
full_path = os.path.join(os.getcwd(), "DescriptiveStats.xlsx")
# full_path = "DescriptiveStats.xlsx"
my_df[my_fields].describe(include='all').to_excel(full_path)

my_fields = ['householdincome', 'householdsize', 'educationlevel', 'gender']
for field in my_fields:
    print(f'\nSummary data for {field}')
    print(rp.summary_cont(my_df[field].groupby(my_df['category'])))

# Let's construct the correlation matrix and visualize those correlations between our features and targets.
# print(my_df.corr())
# full_path = os.path.join(os.getcwd(), "CorrelationMatrix.xlsx")
# full_path = "CorrelationMatrix.xlsx"
# my_df.corr().to_excel(full_path)

plt.close('all')
sns.pairplot(my_df, hue='category', diag_kind='hist', kind='scatter', palette='husl')
plt.savefig(os.path.join(os.getcwd(), 'category_scatterplot.png'))
# plt.savefig('category_scatterplot.png')
plt.close('all')

# Task 6
# Create the numpy arrays necessary to feed into the decision tree classifier
print(my_df.columns)
my_features = my_df[['householdincome', 'householdsize', 'educationlevel', 'gender']].values
print(my_features)
print(my_features.shape)
# For this problem, we have known targets (category field) so this a supervised machine learning problem.
# Our categories are currently strings, so we have to convert them to numbers to feed into the sklearn algorithm.
target_names = {"loyal": 0, "impulse": 1, "discount": 2, "need-based": 3, "wandering": 4}
my_df['category'] = my_df['category'].map(target_names)
my_targets = my_df[['category']].values.ravel()
print(my_targets)
print(my_targets.shape)

# Task 7
# Randomly split our data into training and testing data.
# Let's make sure our analyses are reproducible and keep 25% of our data for testing.
from sklearn.model_selection import train_test_split
f_train, f_test, t_train, t_test = train_test_split(my_features, my_targets, test_size=0.25, random_state=77)

# Task 8
# Now let's apply the decision tree algorithm to these data.
# To do this we will use the DecisionTreeClassifier estimator from the scikit-learn tree module.
# This estimator will construct, by default, a tree from a training data set.
# This estimator accepts a number of hyper-parameters, including:
# #   max_depth : the maximum depth of the tree. By default this is None, which means the tree is
# #               constructed until either all leaf nodes are pure, or all leaf nodes contain fewer
# #               instances than the min_samples_split hyper-parameter value.
# #   min_samples_split : the minimum number of instances required to split a node into two
# #                child nodes, by default this is two.
# #   min_samples_leaf: the minimum number of instances required to make a node terminal
# #               (i.e., a leaf node). By default this value is one.
# #   max_features: the number of features to examine when choosing the best split feature and value.
# #               By default this is None, which means all features will be explored.
# #   random_state: the seed for the random number generator used by this estimator.
# #               Setting this value ensures reproducibility.
# #   class_weight: values that can improve classification performance on
# #               unbalanced data sets. By default this value is None. This hyper-parameter
# #               is not relevant when our target is continuous.
# #   ccp_alpha: the complexity parameter used for Minimal Cost-Complexity Pruning.
from sklearn.tree import DecisionTreeClassifier
# There are two decision tree algorithms: 1) DecisionTreeRegressor for continuous targets
# 2) DecisionTreeClassifier for categorical targets.
# The following will use all the default hyper-parameters for the DecisionTreeClassifier
# machine learning algorithm. Note that we sometimes refer to a machine learning algorithm
# as an estimator, an algorithm, a classifier (for categorical targets), or a
# regressor (for continuous targets).
# dtr_estimator = DecisionTreeClassifier()
# The following will create a variable holding the classifier with custom values for
# three of the hyper-parameters.
dtr_estimator = DecisionTreeClassifier(min_samples_leaf=5, min_samples_split=5, max_depth=10)
# If we want to see all the hyper-parameters that will be used to train our decision tree model,
# we can execute the following:
print(dtr_estimator.get_params())

# Let's now use this estimator to train a decision tree model using the training data.
# Notice that we train the model with the randomly selected training data.
trained_dt_model = dtr_estimator.fit(f_train, t_train)
# Now, let's score the model against the unseen testing data.
print('Score = {:,.1%}'.format(trained_dt_model.score(f_test, t_test)))
# Let's determine the difference between training and testing to determine model variance
print('Score = {:,.1%}'.format(trained_dt_model.score(f_train, t_train)))
print(f'Difference between training and testing to determine potential overfitting '
      f'is {trained_dt_model.score(f_train, t_train) - trained_dt_model.score(f_test, t_test):,.2%} points')

# Task 9
# Let's print a few other metrics to determine performance
# Compute and display classification report
from sklearn.metrics import classification_report
target_names = ["loyal", "impulse", "discount", "need-based", "wandering"]

# Generate predictions. Notice that we are using the unseen testing data because these performance
# metrics are calculated using the randomly selected testing data.
predicted_labels = trained_dt_model.predict(f_test) # See how accurate it was
# NOTE: The "Support" column in the following output is the sample size.
print(classification_report(t_test, predicted_labels, target_names=target_names))
# Let's display the confusion matrix, which shows us how specifically the trained model is confused.
from sklearn.metrics import confusion_matrix
print(confusion_matrix(t_test, predicted_labels))

# The above output is a little hard to interpret because we don't have any labels indicating which axes
# are the predicted labels and which are the actual targets, so let's visual the confusion matrix.
# The following function is a helper function designed to display a heatmap with the confusion matrix data.


def confusion(test, predict, title, labels, categories):
    """Plot the confusion matrix to make it easier to interpret.
       This function produces a colored heatmap that displays the relationship
        between predicted labels and actual targets."""

    # Make a 2D histogram from the test and result arrays.
    # pts is essentially the output of the scikit-learn confusion_matrix method.
    pts, xe, ye = np.histogram2d(test, predict, bins=categories)

    # For simplicity we create a new DataFrame for the confusion matrix.
    pd_pts = pd.DataFrame(pts.astype(int), index=labels, columns=labels)

    # Display heatmap and add decorations.
    hm = sns.heatmap(pd_pts, annot=True, fmt="d")
    hm.axes.set_title(title, fontsize=20)
    hm.axes.set_xlabel('True Target', fontsize=18)
    hm.axes.set_ylabel('Predicted Label', fontsize=18)
    # I might sometimes return None if I want to have an if conditional associated with
    # the output of this function. For this example, it is not really needed, so I have it commented out!
    # return None


# Call confusion matrix plotting routine
plt.close('all')
confusion(t_test.flatten(), predicted_labels, f'Decision Tree Model', target_names, 5)
plt.savefig(os.path.join(os.getcwd(), 'confusion_customer_types.png'))
# plt.savefig('confusion_customer_types.png')
plt.close('all')

# Task 10
# Let's now display the relative importance of each feature on the Decision Tree model.
# Which features are the most important in our decision tree?
feature_names = ['householdincome', 'householdsize', 'educationlevel', 'gender']
for name, val in zip(feature_names, trained_dt_model.feature_importances_):
    print(f'{name} importance = {100.0*val:5.2f}%')

# Task 11
# Let's now visualize the decision tree model.
# The sklearn learn library includes an export_graphviz method that actually generates a visual tree
# representation of a constructed decision tree classifier. This representation is in the
# dot format recognized by the standard, open source graphviz library.
# NOTE: You might have to install the graphviz package. It should get installed
# when you install sklearn, but sometimes it does not!
# After generating the .dot file, go to the http://www.webgraphviz.com/ to display it online!

# NOTE: You should not have to install export_graphviz but you may have to depending on how you installed sklearn!
from sklearn.tree import export_graphviz

# Write the dot representation of the trained decision tree model to a file (.dot file)
with open(os.path.join(os.getcwd(), 'Tree_Customer_Types.dot'), 'w') as fdot:
# with open('Tree_Customer_Types.dot', 'w') as fdot:
    export_graphviz(trained_dt_model, fdot, feature_names=feature_names)

# Task 12
# Let's predict the customer type for the following customer:
# householdincome = 65890
# householdsize = 2
# educationlevel = 10
# Males are coded as 0s and Females are coded as 1s.
# gender = Male
my_data = [65890, 2, 10, 0]
my_array = np.array(my_data, ndmin=2)
print(my_array)
target_names = ["loyal", "impulse", "discount", "need-based", "wandering"]
prediction = trained_dt_model.predict(my_array)
print(prediction)
idx = int(prediction[0])
print(target_names[idx])
# Now, let's determine how confident that this model is with its prediction.
probabilities = trained_dt_model.predict_proba(my_array)
print(probabilities)

output = 'This model estimates that there is a \n'
for index, elem in np.ndenumerate(probabilities):
    output += f'{elem * 100}% chance that this customer will ' \
              f'be in the {target_names[index[1]]} category\n'
print(output)
