
'''1a .
# given list'''
import statistics
ages = [19, 22, 19, 24, 20, 25, 26, 24, 25, 24]

'''1a sorting by using sort method
We are sorting the list using sort method and printing them'''
ages.sort()
print(ages)

'''1 b finding min and max value
 We are printing minimum of ages using min method and maximum of elements using max elements'''
print(min(ages))
print(max(ages))

''' 1 c adding min and max values to list
 We are trying to add minimum elements and maximum element in the list again to list'''
ages.append(min(ages))
ages.append(max(ages))
print(ages)

'''1 d finding the median age
Using statistics package we are using median method and we are finding the median of the list'''
print(statistics.median(ages))

''' 1 e finding average age
Using statistics package we are using mean method and we are finding the average of the list'''
Avg= statistics.mean(ages)
print(Avg)
''' 1 f finding range of ages
We are finding range by subtracting minimum of ages from maximum ages'''
range_age = max(ages) - min(ages)
print(range_age)




'''2.'''

'''2 a creating dog dictionary
We are creating an empty dictionary dog'''
dog={}
print(dog)

'''2 b giving names , color, breed, legs, age
We are assigning keys and values to the dictionary'''

dog={'name':'snoopy','color':'white','breed':'x','legs':2,'age':18}
print(dog)

''' 2 c student dictionary'''
'''We are creating new dictionary student and assigning the keys and values and printing'''
student=dict()
student["first_name"] = "Satya"
student["last_name"] = "Dev"
student["Gender"] = "Female"
student["Age"] = 36
student["Martal status"] = "Married"
student["Skills"] = ["python", "Java", "JS"]
student["Country"] = "India"
student["City"] = "Ct"
student["Address"] = "street 1"
print("Student's dictionary : ", student)
print()

''' 2 d  length of student dictionary'''
'''We are using str to find length of the student dictionary'''
d=str(len(student))
print("length of student is : ", d)

''' 2 e  value of student skills and datatype'''
'''We are printing the value of skills and we are finding type of the skills using type method'''
print("skills are : ", end='')
print(student["Skills"])
t=type(student["Skills"])
print("type of skills : ",t)

'''2 f modifying student skills'''
'''We are adding the 2 more skills to skill list using append'''
student["Skills"].append("s1")
student["Skills"].append("s2")
sk=student["Skills"]
print("updated student skills  are : ",sk)

'''2 g getting dictionary keys as list'''
'''We are using list method and keys method to get keys of dictionary and printing them'''
d_keys = list(dog.keys())
print("dog dictionary keys : ", d_keys)
s_keys = list(student.keys())
print("student dictionary keys : ", s_keys)
'''We are using list method and values method to get values of dictionary and printing them '''
d_values = list(dog.values())
print("dog dictionary values : ", d_values)
s_values = list(student.values())
print("student dictionary values : ", s_values)

'''3. 3 a  creating tuples brothers and sisters' and assigning values and printing them'''
si = ("s1", "s2","s3")
br = ("b1", "b2", "b3")
print("brothers : ", br)
print("sisters : ", si)

''' 3 b joining brothers and sisters tuples  by + to siblings&printing them'''
sib = br + si
print("siblings  : ", sib)
'''Finding length of siblings using Len method'''
sl=len(sib)
print("number of siblings : ", sl)

''' 3 d modifying siblings tuple by adding father name and mother name a’’’nd adding to family members using +'''
family_members = ("f", "m") + sib
print("family members : ", family_members)

# output

'''4. 4 a  given list of it companies'''
it_companies = {"Facebook", "Google", "Microsoft", "Apple", "IBM", "Oracle", "Amazon"}
A = {19, 22, 24, 20, 25, 26}
B = {19, 22, 20, 25, 26, 24, 28, 27}
age = [22, 19, 24, 25, 26, 24, 25, 24]
'''Finding length of companies using len method'''
lc=len(it_companies)
print("length of it_companies : ", lc)

''' 4 b adding twitter to it_companies using add method and printing them'''
it_companies.add("Twitter")
print("companies after adding updating :", it_companies)

''' 4 c inserting multiple it companies us’’’ing update method and printing the newly modified list'''
comp = ["NCR1", "Wipro1", "TCS1"]
it_companies.update(comp)
print("companies after adding multiple companies :", it_companies)

''' 4 d removing one company'''
'''We are removing using remove method'''
it_companies.remove("TCS1")
print("companies after removing a company : ", it_companies)

''' 4 e Difference between remove and discard method'''

''' The remove() method will raise an error if the specified item does not exist, and the discard() method will not raise error.'''

''' 4 f joining A and B using union method and printing them'''
jo = A.union(B)
print("join of A and B:", jo)

''' 4 g finding A intersection B using intersection method'''
inte = A.intersection(B)
print("Intersect of A and B is:", inte)

''' 4 h    checking whether A is subset of B using issubset and printing it using if block'''
ck = A.issubset(B)
if ck:
    print("yes A is subset of B")
else:
    print("No A is not a subset of B")

'''4 i checking whether A and B are disjoint sets'''
'''Using isdisjoint method we find whether it is disjoint or not'''
ck1 = A.isdisjoint(B)
if ck1:
    print("Yes A and B are disjoint sets")
else:
    print("No A and B are not disjoint sets")

''' 4 j joining A with B and B with A using union method'''
A_and_B = A.union(B)
B_and_A = B.union(A)
print("A join B is :", A_and_B)
print("B join A is :", B_and_A)

''' 4 k  we are finding symmetric difference between A and B by ‘’’using symmetric_difference method and printing them'''
Di = A.symmetric_difference(B)
print("symmetric difference is :", Di)

''' 4 l deleting all the sets using clear method'''
it_companies.clear()
A.clear()
B.clear()

''' 4 m converting ages list to set using set method'''
age_s = set(age)
'''Finding the length of set and list using len method'''
print("length of ages list : ", len(age))
print("length of ages set :", len(age_s))
print("length of ages list is greater than the ages set")

# output


'''5. 5 a area of circle'''
''' Given radius of circle 30 meters'''
r = 30
'''assigned variable name "_area_of_circle"'''
_area_of_circle = 3.14 * (r ** 2)
''' printing the value '''
print("area of circle :", _area_of_circle)

'''5 b circumference of circle'''
''' assigned variable name "_circum_of_circle"'''
_circum_of_circle = 2 * 3.14 * r
''' printing the value'''
print("circumference of circle :", _circum_of_circle)

''' We are taking area of circle from user input using input method'''
n=int(input())
#assigned the result to"area"
area=3.14*(n**2)
''' printing the value'''
print("Area of circle with user input :", area)

# output


'''6. a
 given string to the variable wor'''
wor = "I am a teacher and I love to inspire and teach people"
''' splitting the given string using split'''
br = wor.split()
''' using set to remove duplicates'''
set1 = set(br)
''' using len to count number of words and storing in co'''
co = len(set1)
print(co)

# output


'''7.
 tab escape
 declaring a string to the txt variable'''
'''Using /t for spaces and /n for next line'''
tt = "Name\tAge\tCountry\tCity\nAsabeneh\t250\tFinland\tHelsinki"
print(tt)

'''8.
 assigning  radius'''
radius = 10
'''Calculating area of circle formula'''
area = 3.14 * radius ** 2
'''Printing the statement according to the mentioned'''
print("The area of circle with radius {} is {} meters square".format(radius, area))

# output


'''9. '''
 #we take no of students through user input
n=int(input("Enter number of student's weight to be calculated"))
weights_in_lbs=[]
weights_in_kg=[]
# adding  the elements into the list and printing lbs in list using for loop
for i in range(n):
    weights_in_lbs.append(int(input("weight {} \n".format(i+1))))
print(weights_in_lbs)
#converting lbs to kilogram  &printing that in another list
for i in range(len(weights_in_lbs)):
    lbs=0.45359237 #1lbs= 0.45359237kg
    temp=round(weights_in_lbs[i]*lbs,2)
    weights_in_kg.append(temp)
    temp=0
print(weights_in_kg)


'''10'''



'''import numpy as np  #importing important python libraries
import matplotlib.pyplot as plt
import pandas as pd
dataframe=pd.read_csv("dataset.csv")#reading the dataset
x= dataframe['Feature'].values
y= dataframe['Class'].values
#dividing data equally into training and testing data
from sklearn.model_selection import train_test_split
features_tr, features_te, label_tr, label_te= train_test_split(x, y, random_state=0, train_size= 0.5 )
#reshaping the data feature and labes into 2D array
features_tr = np.array(features_tr).reshape(-1,1)
features_te = np.array(features_te).reshape(-1,1)
#Normalizing data
from sklearn.preprocessing import StandardScaler
normalization= StandardScaler()
features_tr= normalization.fit_transform(features_tr)
features_te= normalization.transform(features_te)
#fiting the training data into classifier model
from sklearn.neighbors import KNeighborsClassifier
model= KNeighborsClassifier(n_neighbors=3 )
model.fit(features_tr, label_tr)
#Predicting the test set result
predict_class= model.predict(features_te)
print("Predicted Test Samples Output:",predict_class)

#creating a confusion matrix
from sklearn.metrics import confusion_matrix
model_evaluation= confusion_matrix(label_te, predict_class)
print("Confusion matrix:\n",model_evaluation)
#finding model accuracy
count=sum(sum(model_evaluation))
accuracy=(model_evaluation[0,0]+model_evaluation[1,1])/count
print ('Accuracy =: ', accuracy)
# finding model sensitivity
sense = model_evaluation[0,0]/(model_evaluation[0,0]+model_evaluation[0,1])
print('Sensitivity =: ', sense )
#finding model specificity
speci = model_evaluation[1,1]/(model_evaluation[1,0]+model_evaluation[1,1])
print('Specificity =: ', speci)'''
