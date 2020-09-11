import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def welcome():
    print("***** Welcome To Salary Prediction System *****")
    print("Press Enter To Proceed : ")
    input()


def checkcsv():
    csv_files=[]
    current_directory=os.getcwd()
    content_list=os.listdir(current_directory)

    for x in content_list:
        if x.split(".")[-1]=="csv":
            csv_files.append(x)

    if len(csv_files)==0:
        return "No any csv file is found"
    else:
        return csv_files

def display_and_select_csv(csv_files):
    print("Index ..... Files")
    i=0
    for file_name in csv_files:
        print(i, "    .....", file_name)
        i=i+1

    print("Select the csv file index to create Machine Learning Model : ")
    index=int(input())
    return csv_files[index]
    
def graph(X_train, Y_train, regressionObject, X_test, Y_test, Y_predict):
    plt.scatter(X_train, Y_train, color="red", label="Training Data")
    plt.plot(X_train, regressionObject.predict(X_train), color= "blue", label="Best Fit Line")
    plt.scatter(X_test, Y_test, color="green", label="Testing Data")
    plt.scatter(X_test, Y_predict, color="black", label="Predicting Test Data")
    plt.title("Salary vs Experience")
    plt.xlabel("Years of experience")
    plt.ylabel("Corresponding Salary")
    plt.legend()
    plt.show()


def main():
    welcome()

    try:
        csv_files=checkcsv()
        if csv_files=="No any csv file is found":
            raise FileNotFoundError("No any csv file is found")

        csv_file=display_and_select_csv(csv_files)
        print(csv_file,"is selected")
        print("Reading csv file .....")
        print("Creating DataSet .....")
        dataset=pd.read_csv(csv_file)
        print("DataSet is Created !!!!!")
        X=dataset.iloc[ :, :-1].values
        Y=dataset.iloc[ : , -1].values

        print("Enter the tesing data size from 0 to 1 :")
        s=float(input())
        X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=s)

        print("Machine Learning Model creation is in progress .....")
        regressionObject=LinearRegression()
        regressionObject.fit(X_train,Y_train)
        print("Machine Learning Model Is Created !!!!!")
        print("Press Enter key to predict test data in Trained Model")
        input()

        Y_predict=regressionObject.predict(X_test)
        print("Experience ........ Salary ........ Predicted Salary")

        i=0
        while i<len(X_test):
            print(X_test[i],"..............", Y_test[i], ".........", Y_predict[i])
            i=i+1
        print("Press Enter key to see above result in graphical format")
        input()
        graph(X_train, Y_train, regressionObject, X_test, Y_test, Y_predict)
        
        r2=r2_score(Y_test,Y_predict)
        print("Our ML Model is %2.2f%% accurate" %(r2*100))

        print("Now you can use the Salary Predictor System to predict the salary of an employee")
        print("Enter the experience in years of some candidates separated by commas to Predict Their Salaries : ")

        exp=[float(e) for e in input().split(",")]

        ex=[]
        for x in exp:
            ex.append([x])
        experience=np.array(ex)
        salaries=regressionObject.predict(experience)

        plt.scatter(experience, salaries, color="blue")
        plt.xlabel("Years Of Experience")
        plt.ylabel("Corresponding Salaries")
        plt.show()

        table=pd.DataFrame({"Experience":exp, "Salaries":salaries})
        print(table)

        print("Press Enter To Exit")
        input()
        exit()

    except FileNotFoundError:
        print("No any csv file is found")
            


if __name__=="__main__":
    main()
    input()
