from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data  # shape (150,4)
y = iris.target # shape (150,)
print(iris.feature_names, iris.target_names)

from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state = 42)
                                                   
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state = 42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Predictions", y_pred[:5])
print("True labels", y_test[:5])

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:" , accuracy)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
confusion = confusion_matrix(y_test, y_pred)
print("confusion_matrix:",confusion, sep='\n')


import os

nested_path = "C:/Users/Holy/Documents/ITSwitch/iris-classifier/outputs/confusion"

# exist_ok=True prevents errors if the path already exists
os.makedirs(nested_path, exist_ok=True)

disp = ConfusionMatrixDisplay(confusion_matrix=confusion)
disp.plot().figure_.savefig('C:/Users/Holy/Documents/ITSwitch/iris-classifier/outputs/confusion/_matrix.png')