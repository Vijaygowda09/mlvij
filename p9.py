from sklearn.datasets import fetch_olivetti_faces as faces
from sklearn.model_selection import train_test_split as split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score as acc, classification_report as report, confusion_matrix as matrix
import matplotlib.pyplot as plt
X, y = faces(shuffle=True, random_state=42, return_X_y=True)
X_train, X_test, y_train, y_test = split(X, y, test_size=0.3, random_state=42)
clf = GaussianNB().fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(f'Accuracy: {acc(y_test, y_pred)*100:.2f}%\n')
print('Classification Report:\n', report(y_test, y_pred, zero_division=1))
print('Confusion Matrix:\n', matrix(y_test, y_pred))
print(f'\nCross-val accuracy: {cross_val_score(clf, X, y, cv=5).mean()*100:.2f}%')
fig, ax = plt.subplots(3, 5, figsize=(10, 6))
for a, img, t, p in zip(ax.ravel(), X_test, y_test, y_pred):
    a.imshow(img.reshape(64, 64), cmap='gray')
    a.set_title(f'T:{t} P:{p}')
    a.axis('off')
plt.tight_layout(); plt.show()
