import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
data = np.random.rand(100)
train, test = data[:50], data[50:]
labels = ["Class1" if x <= 0.5 else "Class2" for x in train]
def knn(x, k):
    d = sorted([(abs(x - t), labels[i]) for i, t in enumerate(train)])
    return Counter([lbl for _, lbl in d[:k]]).most_common(1)[0][0]
for k in [1, 3, 5, 20, 30]:
    print(f"\n--- k = {k} ---")
    preds = [knn(x, k) for x in test]
    for i, (x, p) in enumerate(zip(test, preds), 51):
        print(f"x{i} (value: {x:.4f}) -> {p}")
    plt.scatter(train, [0]*50, c=["blue" if l == "Class1" else "red" for l in labels], label="Train", marker="o")
    plt.scatter(test, [1]*50, c=["blue" if p == "Class1" else "red" for p in preds], label="Test", marker="x")
    plt.title(f"k-NN Results (k={k})")
    plt.yticks([0, 1], ["Train", "Test"])
    plt.grid(True)
    plt.legend()
    plt.show()
