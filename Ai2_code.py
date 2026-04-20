#Group members : Besan Maaly 1222776 & Abeer Hussein 1220425

import os # For handling file and folder paths
import numpy as np # For numerical array operations
from PIL import Image # For loading and processing images
from sklearn.preprocessing import LabelEncoder, StandardScaler #Label encoding and feature normalization
from sklearn.model_selection import train_test_split, GridSearchCV #For splitting data and hyperparameter tuning
from sklearn.tree import DecisionTreeClassifier #For Main Decision Tree model
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score #Used to evaluation metrics
from sklearn.naive_bayes import GaussianNB #For Naive Bayes classifier
from sklearn.neural_network import MLPClassifier #Multi-layer Perceptron neural network
import matplotlib.pyplot as plt #For plotting the Decision Tree
from sklearn.tree import plot_tree #For visualizing the Decision Tree
from sklearn.feature_selection import SelectKBest, f_classif #For selecting the best k features


IMG_SIZE= (32,32)
CLASSES = ["bird", "cat", "fish"]
#______________________load_data Function______________________________

def load_data(folder_path, classes, img_size=IMG_SIZE):#Loading Data from Dataset files
    X, y = [], []
    for label in classes:
        path = os.path.join(folder_path, label)
        if not os.path.exists(path):
            print(f"Warning: Folder not found {path}")
            continue
        for file in os.listdir(path):
            try:
                img_path = os.path.join(path, file)
                img = Image.open(img_path).convert("L") # change to graystle (1 Diminssion images)
                img = img.resize(img_size)
                img_array = np.array(img).flatten()#make the imeages flatten
                X.append(img_array)
                y.append(label)
            except Exception as e:
                print(f"Error loading {file}: {e}")
    return np.array(X), np.array(y)

#_____________________run_decision_tree Function _______________________

def run_decision_tree(X_train, y_train, X_test, y_test, le):
    selector = SelectKBest(score_func=f_classif, k=200)# Select the top 200 most relevant features 
    X_train = selector.fit_transform(X_train, y_train)
    X_test = selector.transform(X_test)

    param_grid = { # Grid search parameters
        'criterion': ['gini', 'entropy'],# impurity criteria
        'max_depth': [10, 20, 30],# Tree maximum depth
        'min_samples_split': [2, 4],#minimum samples required to split a node
        'min_samples_leaf': [1, 2]#minimum samples required at a leaf node
    }
 #initialize Decision Tree
    decision_tree = DecisionTreeClassifier(random_state=42)
    #used Grid Search Cross Validation to find best hyperparameters
    grid_search = GridSearchCV(decision_tree, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    # n_jobs=-1 to use all CPU cores
    #cv=3 to use 3-fold cross-validation
    grid_search.fit(X_train, y_train)
    # to get the best model found by Grid Search
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    print("\n--- Decision Tree Results---")
    # print("Best Parameters:", grid_search.best_params_)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    plt.figure(figsize=(16, 8))
    plot_tree(best_model, filled=True, class_names=le.classes_, max_depth=2, fontsize=10) #max_depth=2 is the Limit tree depth in plot 
    plt.title("Decision Tree")
    plt.show()

#_____________________run_naive_bayes Function__________________________
def run_naive_bayes(X_train, y_train, X_test, y_test, le):
    #initialize a Gaussian Naive Bayes classifier
    model = GaussianNB()
    model.fit(X_train, y_train) #Train the model on the training data
    y_pred = model.predict(X_test)

    print("\n--- Naive Bayes Results ---")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

#______________________run_mlp_classifier Function________________________
def run_mlp_classifier(X_train, y_train, X_test, y_test, le):
    model = MLPClassifier(
        hidden_layer_sizes=(100, 50), #Two hidden layers: first with 100 neurons, second with 50 neurons
        activation='relu', #ReLU activation function
        solver='adam', # Adam optimizer
        learning_rate_init=0.001,#learning rate inital
        max_iter=2000,# Maximum number of training iterations
        early_stopping=False,         
        random_state=42,
        verbose=False              
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\n--- MLP (Neural Network) Results ---")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

#_________main Function___________________________________________
def main():
    print("Loading data...")
    X, y_raw = load_data("dataset", CLASSES, IMG_SIZE) # Load image data and labels from the dataset folder
    if X.size == 0: #check if data was loaded successfully
        print("Error: Data is empty.")
        return
    labbel_encode = LabelEncoder() #Encode string labels into numeric labels (0,1,2)
    y = labbel_encode.fit_transform(y_raw)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    #split the data into training and testing sets 80% train and 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    while True:
        print("\nChoose model to run:")
        print("1. Decision Tree")
        print("2. Naive Bayes")
        print("3. MLP Neural Network")
        print("4. exit")
        choice = input("Enter your choice (1, 2, 3 or 4 ): ")
        if choice == '1':
            run_decision_tree(X_train, y_train, X_test, y_test, labbel_encode)
        elif choice == '2':
            run_naive_bayes(X_train, y_train, X_test, y_test, labbel_encode)
        elif choice == '3':
            run_mlp_classifier(X_train, y_train, X_test, y_test, labbel_encode)
        elif choice == '4':
            print("Exiting....")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4")

if __name__ == "__main__":
    main()
