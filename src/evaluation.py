import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, f1_score

class ModelEvaluator:
    def __init__(self, results_dir, data_dir):
        self.results_dir = results_dir
        self.data_dir = data_dir
        self.y_test = np.load(os.path.join(data_dir, 'processed_data', 'y_test.npy'))
        
        # Load predictions (labels 0, 1)
        self.y_pred_knn = np.load(os.path.join(results_dir, 'y_pred_knn.npy'))
        self.y_pred_tree = np.load(os.path.join(results_dir, 'y_pred_tree.npy'))
        
        # Load probabilities (probability 0.0 - 1.0) - Need this file for ROC plot
        # If not available, run main.py again with the fix suggested
        try:
            self.y_prob_knn = np.load(os.path.join(results_dir, 'y_prob_knn.npy'))
            self.y_prob_tree = np.load(os.path.join(results_dir, 'y_prob_tree.npy'))
        except FileNotFoundError:
            print("WARNING: Probability files (*_prob.npy) not found. Skipping ROC plot.")
            self.y_prob_knn = None
            self.y_prob_tree = None

    def plot_confusion_matrix_comparison(self):
        """Plot 2 confusion matrices side by side"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # KNN Matrix
        cm_knn = confusion_matrix(self.y_test, self.y_pred_knn)
        sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('KNN Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        axes[0].set_xticklabels(['Normal', 'Pneumonia'])
        axes[0].set_yticklabels(['Normal', 'Pneumonia'])

        # Tree Matrix
        cm_tree = confusion_matrix(self.y_test, self.y_pred_tree)
        sns.heatmap(cm_tree, annot=True, fmt='d', cmap='Greens', ax=axes[1])
        axes[1].set_title('Decision Tree Confusion Matrix')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        axes[1].set_xticklabels(['Normal', 'Pneumonia'])
        axes[1].set_yticklabels(['Normal', 'Pneumonia'])

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'confusion_matrix_comparison.png'))
        print("[SUCCESS] Saved Confusion Matrix comparison.")
        plt.show()

    def plot_roc_curve_comparison(self):
        """Plot ROC curve comparing 2 models"""
        if self.y_prob_knn is None or self.y_prob_tree is None:
            print("Skipping ROC plot: probability files not available.")
            return

        plt.figure(figsize=(10, 8))
        
        # KNN ROC
        fpr_knn, tpr_knn, _ = roc_curve(self.y_test, self.y_prob_knn)
        roc_auc_knn = auc(fpr_knn, tpr_knn)
        plt.plot(fpr_knn, tpr_knn, color='blue', lw=2, label=f'KNN (AUC = {roc_auc_knn:.3f})')

        # Tree ROC
        fpr_tree, tpr_tree, _ = roc_curve(self.y_test, self.y_prob_tree)
        roc_auc_tree = auc(fpr_tree, tpr_tree)
        plt.plot(fpr_tree, tpr_tree, color='green', lw=2, label=f'Decision Tree (AUC = {roc_auc_tree:.3f})')

        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve Comparison: KNN vs Decision Tree')
        plt.legend(loc="lower right")
        
        plt.savefig(os.path.join(self.results_dir, 'roc_curve_comparison.png'))
        print("[SUCCESS] Saved ROC Curve comparison.")
        plt.show()

    def print_metrics_report(self):
        """Print detailed metrics report"""
        print("\n" + "="*50)
        print(" PERFORMANCE COMPARISON TABLE") 
        print("="*50)
        
        print("\n--- KNN CLASSIFICATION REPORT ---")
        print(classification_report(self.y_test, self.y_pred_knn, target_names=['Normal', 'Pneumonia']))
        
        print("\n--- DECISION TREE CLASSIFICATION REPORT ---")
        print(classification_report(self.y_test, self.y_pred_tree, target_names=['Normal', 'Pneumonia']))

    def analyze_errors(self, model_name='knn', num_samples=5):
        """Display images with incorrect predictions"""
        print(f"\n[ERROR ANALYSIS] Analyzing errors for {model_name.upper()}...")
        
        # Select corresponding predictions
        y_pred = self.y_pred_knn if model_name == 'knn' else self.y_pred_tree
        
        # Find incorrect indices
        # Incorrect: Prediction differs from actual label
        incorrect_indices = np.where(y_pred != self.y_test)[0]
        
        if len(incorrect_indices) == 0:
            print(f"Perfect! {model_name.upper()} has no errors.")
            return

        # Load original images
        # This file contains images (224, 224, 3) before PCA
        try:
            X_test_raw = np.load(os.path.join(self.data_dir, 'processed_data', 'X_test.npy'))
        except FileNotFoundError:
            print("Error: Original image file 'X_test.npy' not found. Check file path.")
            return

        # Randomly select a few incorrect samples to display
        sample_indices = np.random.choice(incorrect_indices, min(len(incorrect_indices), num_samples), replace=False)
        
        plt.figure(figsize=(15, 5))
        for i, idx in enumerate(sample_indices):
            plt.subplot(1, num_samples, i + 1)
            
            # Display image
            img = X_test_raw[idx]
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            
            true_label = "Pneumonia" if self.y_test[idx] == 1 else "Normal"
            pred_label = "Pneumonia" if y_pred[idx] == 1 else "Normal"
            
            plt.title(f"True: {true_label}\nPred: {pred_label}", color='red', fontsize=10)
        
        plt.suptitle(f"Misclassified cases by {model_name.upper()}", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'{model_name}_errors.png'))
        print(f"[SUCCESS] Saved error analysis images for {model_name}.")
        plt.show()

# --- RUN PROGRAM ---
if __name__ == '__main__':
    # 1. Define project root directory
    ROOT_DIR = r'C:\Users\PC\Lung-disease-classification---Chest-X-Ray-Images'
    
    # 2. Define correct path to data folder
    DATA_DIR = os.path.join(ROOT_DIR, 'data')
    
    # 3. Define correct path to results folder (SỬA Ở ĐÂY)
    # Results nằm TRONG data, nên phải join với DATA_DIR, không phải ROOT_DIR
    RESULTS_DIR = os.path.join(DATA_DIR, 'results') 

    print("[INFO] Checking paths...")
    print(f"[INFO] Data Dir: {DATA_DIR}")
    print(f"[INFO] Results Dir: {RESULTS_DIR}") # Check xem dòng này có chữ "data" chưa

    # Initialize Evaluator
    evaluator = ModelEvaluator(
        results_dir=RESULTS_DIR,
        data_dir=DATA_DIR
    )
    
    print("\n[START] Generating evaluation reports...\n")
    evaluator.print_metrics_report()
    evaluator.plot_confusion_matrix_comparison()
    evaluator.plot_roc_curve_comparison()
    
    # Analyze errors for both models
    evaluator.analyze_errors(model_name='knn')
    evaluator.analyze_errors(model_name='tree')
    
    print("\n[COMPLETED] Evaluation finished!")