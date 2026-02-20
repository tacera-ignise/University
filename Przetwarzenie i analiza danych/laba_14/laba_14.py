import matplotlib.pyplot as plt

# Значения FPR и TPR
fpr = [0.00, 0.00, 0.00, 0.25, 0.25, 0.50, 0.50, 0.75, 1.00]
tpr = [0.00, 0.25, 0.50, 0.50, 0.75, 0.75, 1.00, 1.00, 1.00]

# Вычисленное значение AUC
auc = 0.6875

# Построение ROC-кривой
plt.figure()
plt.plot(fpr, tpr, marker='o', color='b', label=f'AUC = {auc:.4f}')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC-кривая')
plt.legend(loc="lower right")
plt.grid()
plt.show()
