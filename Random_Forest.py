#Random Forest Implementation
#number of trees :n
n = 10
import csv
import math
import numpy as np

#samplefilepath = "C:/Ragavi/Machine Learning Decision Tree Project/Nursery_Data.csv"  #first data set
samplefilepath = "C:/Ragavi/Machine Learning Decision Tree Project/Car_Evaluation_Data.csv"#second data set (uncomment this to run for this dataset

#Entropy value for a target is being returned to the calling function
def entropy_for_a_target(frequency_of_tdata, length_of_data):
    ent= (-frequency_of_tdata/length_of_data)*math.log(frequency_of_tdata/length_of_data,2)
    return ent

#Function to calculate the entropy, returns the entropy
def entropy(data):
    targets = [row[0] for row in data]
    frequency_of_targets = {x: targets.count(x) for x in set(targets)}
    #after finding the frequency of targets, entropy_by_target is found by invoking entropy_for_a_target function
    entropy_by_target = {x: entropy_for_a_target(frequency, len(data)) for x, frequency in frequency_of_targets.items()}
    # the entropy value is returned
    return sum(entropy_by_target.values())

#Probability is calculated, it returns the freq/length of data
def get_probability(target_value, distribution):
    length_of_data = len(distribution)
    frequency_of_target = distribution.count(target_value)
    return frequency_of_target/length_of_data

#Function to calculate the information gain, returns the gain value
def informationgain(data, attribute_index):
    #entropy function is called
    entropy_value = entropy(data)
    attribute_values = [x[attribute_index] for x in data]
    #unique attribute values is found
    unique_attribute_values = set(attribute_values)
    gain = 0
    for attribute_value in unique_attribute_values:
        #filtered data gets calculated
        filtered_data = [x for x in data if x[attribute_index] == attribute_value]
        #Probabaility function is called
        prob = get_probability(attribute_value, attribute_values)
        entropy_for_value = entropy(filtered_data)
        gain += (prob * entropy_for_value)
        #returns the gain value
    return entropy_value - gain

# Function to choose the best attribute, it returns the maximum gain attribute
def choose_attribute(data):
    max_gain = 0
    max_gain_attribute = -1
    total_attributes = len(data[0]) - 1
    for x in range(total_attributes):
        attribute_index = x + 1
        #informationgain function is called to get the gain values of attributes
        gain = informationgain(data, attribute_index)
        if gain > max_gain:
            max_gain = gain
            max_gain_attribute = attribute_index
    return max_gain_attribute

#Function to build the decision tree, it returns a tree
def build_tree(data, title):
    total_attributes = len(data[0]) - 1
    if total_attributes == 0:
        return
    target_values = set([x[0] for x in data])
    if len(target_values) == 1:
        return target_values.pop()
    #The best attribute is choosen through the choose_attribute function
    chosen_attribute_index = choose_attribute(data)
    tree = {title[chosen_attribute_index]: {}}
    attribute_values = set([x[chosen_attribute_index] for x in data])
    for value in attribute_values:
        filtered_data = [x for x in data if x[chosen_attribute_index] == value]
        subtree = build_tree(filtered_data, title)
        tree[title[chosen_attribute_index]][value] = subtree
    return tree

#Function to get the result from decision tree, it returns a tree
def get_result_from_decision_tree(tree, row, title, unique_targets):
    if len(row) != len(title):
        print("R", row)
        print("T", title)
        print("PROBLEM")

    while True:
        root_key = list(tree.keys())[0]
        root_index = title.index(root_key)
        value = row[root_index]
        if root_key not in tree:
            return None
        if value not in tree[root_key]:
            return None
        tree = tree[root_key][value]
        if tree in list(unique_targets):
            return tree

#Function to test the decision tree and return the percentage of accuracy
def test_decision_tree(test_data, title, unique_targets, trees):
    targets = [row[0] for row in test_data]
    no_of_correct_answers = 0
    #every instance in the test data gets checked
    for index, row in enumerate(test_data):
        results = {}
        #looping the instances through for all the decision trees obtained
        for tree in trees:
            # the value from get_result_from_decision_tree is stored in result
            result = get_result_from_decision_tree(tree, row, title, unique_targets)
            count = results.get(result, 0)
            count += 1
            results[result] = count
        #actual_result gets the target value which has the majority vot
        actual_result=max(results, key=results.get)
        #if the value in actual_result equals the target value in the set it increments the no_of_correct_answers
        if actual_result == targets[index]:
              no_of_correct_answers+=1
    return no_of_correct_answers/len(test_data) * 100

#Function to read the dataset from csv file
with open(samplefilepath, 'r') as f:
  reader = csv.reader(f)
  data = list(reader)
  title = data.pop(0)
  targets = [row[0] for row in data]
  unique_targets = set(targets)
  #trees_overall for having the n number of trees
  trees_overall=[]

  for i in range(n):
     #random sampling with replacement
     training_row_indices = np.random.choice(len(data),int(len(data) * 0.80), replace=True)
     X_train = [data[i] for i in training_row_indices]
     # build_tree function is called
     tree = build_tree(X_train, title)
     #trees_overall will have all n decision trees to it
     trees_overall.append(tree)
     print(tree)

  #accuracy_of_train = test_decision_tree(X_train, title, unique_targets, trees_overall)
  #print("accuracy of train data",accuracy_of_train)

  print("\n")
  #Testing data is taken and sent to each of the decision tree
  testing_row_indices = np.random.choice(len(data), int(len(data) * 0.20), replace=True)
  X_test = [data[i] for i in testing_row_indices]
  #test_decision_tree function is called to test instances for all n decision trees
  accuracy_of_testdata = test_decision_tree(X_test, title, unique_targets, trees_overall)
  print("Accuracy of test data % ", accuracy_of_testdata)

  X_train = [data[i] for i in training_row_indices]
  training_row_indice = np.random.choice(len(data), int(len(X_train) * 0.20), replace=True)
  X_l = [data[i] for i in training_row_indice]
  accuracy_of_train=test_decision_tree(X_l,title,unique_targets,trees_overall)
  print("accuracy of train data",accuracy_of_train)


