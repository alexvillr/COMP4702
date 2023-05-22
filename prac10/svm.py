"""
does svm things
"""
import math
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv("./prac10/datasets/ionosphere.csv")
df.dropna()
targets = df.iloc[:, -1]

train, test = train_test_split(df, test_size=0.3, random_state=42, shuffle=True)

X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]
X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]

accuracies = {("linear"): ([]), ("poly"): ([]), ("rbf"): ([]), ("sigmoid"): ([])}
matricies = {("linear"): ([]), ("poly"): ([]), ("rbf"): ([]), ("sigmoid"): ([])}

ITERS = 1000
FIVE_TESTS = ITERS // 7
for kernel, accuracies_kernel in accuracies.items():
    for i in tqdm(range(ITERS)):
        if i <= 1:
            continue
        if i % FIVE_TESTS == 0 or i == 5:
            cval = math.log10(i)
            clf = make_pipeline(
                StandardScaler(), SVC(C=cval, kernel=kernel, gamma="auto")
            )
            clf.fit(X_train, y_train)

            y_train_pred = clf.predict(X_train)
            train_acc = accuracy_score(y_train, y_train_pred)
            train_loss = 1 - train_acc

            # Make predictions on the test set and calculate the test loss
            y_test_pred = clf.predict(X_test)
            test_acc = accuracy_score(y_test, y_test_pred)
            test_loss = 1 - test_acc

            # Plot the decision regions and training/test data points
            cm = confusion_matrix(y_test_pred, y_test, labels=clf.classes_)
            accuracies_kernel.append(test_acc)
            matricies[kernel].append(cm)
        else:
            continue

for kernel, accuracies in accuracies.items():
    print(f"Kernel: {kernel}")
    for accuracy in accuracies:
        print(f"\t Test Accuracy = {accuracy}")

figure, axis = plt.subplots(len(matricies), len(matricies["linear"]))
i = 0
for kernel, matricies in matricies.items():
    for j in range(len(matricies)):
        disp = ConfusionMatrixDisplay(
            confusion_matrix=matricies[i], display_labels=clf.classes_
        ).plot(ax=axis[i, j])
        if j == 0:
            axis[i, j].set_title(f"{kernel}, C = {math.log10(5):.4f}")
        else:
            axis[i, j].set_title(f"{kernel}," "C = {math.log10((j) * five_tests):.4f}")
    i += 1

plt.show()
