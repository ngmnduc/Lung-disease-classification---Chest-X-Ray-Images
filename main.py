import os
import numpy as np
import joblib  # Required for exporting model
from src.models import ModelTrainer

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    INPUT_DIR = os.path.join(BASE_DIR, 'data', 'feature')
    
    OUTPUT_RESULTS_DIR = os.path.join(BASE_DIR, 'data', 'results')
    
    os.makedirs(OUTPUT_RESULTS_DIR, exist_ok=True)

    print("="*40)
    print(" [START] MODEL TRAINING PIPELINE")
    print("="*40)
    print(f"Input Directory:  {INPUT_DIR}")
    print(f"Output Directory: {OUTPUT_RESULTS_DIR}")

    print("\n[INFO] Loading PCA features...")
    try:
        X_train = np.load(os.path.join(INPUT_DIR, 'X_train_pca.npy'))
        y_train = np.load(os.path.join(INPUT_DIR, 'y_train.npy'))
        X_test = np.load(os.path.join(INPUT_DIR, 'X_test_pca.npy'))
        y_test = np.load(os.path.join(INPUT_DIR, 'y_test.npy'))
        
        print(f"   [OK] Train shape: {X_train.shape}")
        print(f"   [OK] Test shape:  {X_test.shape}")
        
    except FileNotFoundError as e:
        print(f"\n[CRITICAL ERROR] Data file not found: {e}")
        return

    print("\n[INFO] Training KNN Model...")
    knn_trainer = ModelTrainer(model_type='knn')
    
    knn_trainer.train(X_train, y_train)
    
    model_pkl_path = os.path.join(OUTPUT_RESULTS_DIR, 'knn_model.pkl')
    joblib.dump(knn_trainer.model, model_pkl_path)
    print(f"   [SUCCESS] Saved KNN Model to: {model_pkl_path}")
    y_pred_knn = knn_trainer.evaluate(X_test, y_test)
    np.save(os.path.join(OUTPUT_RESULTS_DIR, 'y_pred_knn.npy'), y_pred_knn)
    
    y_prob_knn = knn_trainer.model.predict_proba(X_test)[:, 1] 
    np.save(os.path.join(OUTPUT_RESULTS_DIR, 'y_prob_knn.npy'), y_prob_knn)
    
    print(f"   [OK] Saved KNN predictions to {OUTPUT_RESULTS_DIR}")

    print("\n[INFO] Training Decision Tree Model...")
    tree_trainer = ModelTrainer(model_type='decision_tree')
    
    tree_trainer.train(X_train, y_train)
    y_pred_tree = tree_trainer.evaluate(X_test, y_test)
    np.save(os.path.join(OUTPUT_RESULTS_DIR, 'y_pred_tree.npy'), y_pred_tree)
    
    y_prob_tree = tree_trainer.model.predict_proba(X_test)[:, 1]
    np.save(os.path.join(OUTPUT_RESULTS_DIR, 'y_prob_tree.npy'), y_prob_tree)
    
    print(f"   [OK] Saved Decision Tree predictions to {OUTPUT_RESULTS_DIR}")

    print("\n" + "="*40)
    print(" ALL MODELS FINISHED & SAVED!")
    print(" READY FOR STREAMLIT DEPLOYMENT")
    print("="*40)

if __name__ == '__main__':
    main()