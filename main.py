import os
import numpy as np
from src.models import ModelTrainer

def main():
    # 1. Load dữ liệu đã qua PCA
    input_dir = r'D:/Lung-disease-classification---Chest-X-Ray-Images/data/feature'
    output_results_dir = r'D:/Lung-disease-classification---Chest-X-Ray-Images/data/results'
    os.makedirs(output_results_dir, exist_ok=True)

    print("Loading PCA features...")
    X_train = np.load(os.path.join(input_dir, 'X_train_pca.npy'))
    y_train = np.load(os.path.join(input_dir, 'y_train.npy'))
    X_test = np.load(os.path.join(input_dir, 'X_test_pca.npy'))
    y_test = np.load(os.path.join(input_dir, 'y_test.npy'))

    # 2. Huấn luyện và dự đoán với KNN
    knn_trainer = ModelTrainer(model_type='knn')
    knn_trainer.train(X_train, y_train)
    y_pred_knn = knn_trainer.evaluate(X_test, y_test)
    
    # Lưu kết quả KNN
    knn_path = os.path.join(output_results_dir, 'y_pred_knn.npy')
    np.save(knn_path, y_pred_knn)
    print(f"✓ Saved KNN predictions to {knn_path}")

    # 3. Huấn luyện và dự đoán với Decision Tree
    tree_trainer = ModelTrainer(model_type='decision_tree')
    tree_trainer.train(X_train, y_train)
    y_pred_tree = tree_trainer.evaluate(X_test, y_test)
    
    # Lưu kết quả Decision Tree
    tree_path = os.path.join(output_results_dir, 'y_pred_tree.npy')
    np.save(tree_path, y_pred_tree)
    print(f"✓ Saved Decision Tree predictions to {tree_path}")

    print("\n" + "="*30)
    print("ALL MODELS FINISHED!")
    print("="*30)

if __name__ == '__main__':
    main()