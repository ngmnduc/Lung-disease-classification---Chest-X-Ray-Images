import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
import warnings
import joblib

warnings.filterwarnings('ignore')

class FeatureExtractor:
    def __init__(self, model_name='ResNet50'):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        print(f"Loading {self.model_name} model with ImageNet weights...")
        
        if self.model_name == 'ResNet50':
            base_model = ResNet50(
                weights='imagenet', 
                include_top=False,
                input_shape=(224, 224, 3),
                pooling='avg'  
            )
            self.model = base_model
        else:
            raise ValueError(f"Model {self.model_name} is not supported!")
        
        print(f"Model loaded successfully. Output shape: {self.model.output_shape}")
    
    def extract_features(self, X, batch_size=32):
        print(f"\nExtracting features from {len(X)} images...")
        # Scale back to [0, 255] for ResNet preprocessing
        X_preprocessed = X * 255.0
        X_preprocessed = preprocess_input(X_preprocessed)
        
        features = self.model.predict(X_preprocessed, batch_size=batch_size, verbose=1)
        print(f"Features extracted. Shape: {features.shape}")
        return features

class DimensionalityReducer:
    def __init__(self, n_components=None, variance_ratio=0.95):
        self.n_components = n_components
        self.variance_ratio = variance_ratio
        self.pca = None
    
    def fit_transform(self, X_train_features):
        print(f"\nApplying PCA...")
        print(f"Original features shape: {X_train_features.shape}")
        
        if self.n_components is None:
            self.pca = PCA(n_components=self.variance_ratio)
        else:
            self.pca = PCA(n_components=self.n_components)
        
        X_train_pca = self.pca.fit_transform(X_train_features)
        
        print(f"PCA fitted. Components: {self.pca.n_components_}")
        print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")
        print(f"Reduced features shape: {X_train_pca.shape}")
        
        return X_train_pca
    
    def transform(self, X_test_features):
        if self.pca is None:
            raise ValueError("PCA has not been fitted! Call fit_transform() first.")
        
        print(f"\nTransforming test features with PCA...")
        X_test_pca = self.pca.transform(X_test_features)
        print(f"Test features transformed. Shape: {X_test_pca.shape}")
        
        return X_test_pca
    
    def plot_variance_explained(self, save_path='data/features/variance_explained.png'):
        if self.pca is None:
            raise ValueError("PCA has not been fitted!")
        
        plt.figure(figsize=(10, 6))
        cumsum_variance = np.cumsum(self.pca.explained_variance_ratio_)
        
        plt.plot(range(1, len(cumsum_variance) + 1), cumsum_variance, 
                marker='o', linestyle='-', linewidth=2, markersize=4)
        plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
        
        plt.xlabel('Number of Components', fontsize=12)
        plt.ylabel('Cumulative Explained Variance Ratio', fontsize=12)
        plt.title('PCA - Cumulative Explained Variance', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nVariance plot saved to: {save_path}")
        plt.close()

class FeatureVisualizer:
    @staticmethod
    def plot_pca_2d(X_pca, y, save_path='data/features/pca_visualization.png'):
        print(f"\nCreating 2D PCA visualization...")
        plt.figure(figsize=(12, 8))
        
        normal_mask = (y == 0)
        pneumonia_mask = (y == 1)
        
        plt.scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1],
                   c='blue', label='Normal', alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
        plt.scatter(X_pca[pneumonia_mask, 0], X_pca[pneumonia_mask, 1],
                   c='red', label='Pneumonia', alpha=0.6, s=30, edgecolors='k', linewidth=0.5)
        
        plt.xlabel('First Principal Component (PC1)', fontsize=12)
        plt.ylabel('Second Principal Component (PC2)', fontsize=12)
        plt.title('PCA Visualization - Pneumonia vs Normal\n(ResNet50 Features)', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, loc='best')
        plt.grid(True, alpha=0.3)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PCA visualization saved to: {save_path}")
        plt.close()

def extract_and_reduce_features(data_dir, output_dir, n_components=None, variance_ratio=0.95, model_name='ResNet50'):
    print("="*70)
    print("FEATURE EXTRACTION & DIMENSIONALITY REDUCTION PIPELINE")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Processed Data
    print("\n[STEP 1] Loading processed data...")
    try:
        X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
        y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
        X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    except FileNotFoundError as e:
        print(f"[ERROR] Processed data not found in {data_dir}. {e}")
        return None

    print(f"Train data: {X_train.shape}, Labels: {y_train.shape}")
    print(f"Test data: {X_test.shape}")
    
    # 2. Extract Features
    print("\n[STEP 2] Extracting features using ResNet50...")
    extractor = FeatureExtractor(model_name=model_name)
    X_train_features = extractor.extract_features(X_train, batch_size=32)
    X_test_features = extractor.extract_features(X_test, batch_size=32)
    
    # 3. PCA Reduction
    print("\n[STEP 3] Applying PCA for dimensionality reduction...")
    reducer = DimensionalityReducer(n_components=n_components, variance_ratio=variance_ratio)
    
    X_train_pca = reducer.fit_transform(X_train_features)
    X_test_pca = reducer.transform(X_test_features)
    
    # --- [IMPORTANT] Export PCA Transformer for Streamlit ---
    pca_pkl_path = os.path.join(output_dir, 'pca_transformer.pkl')
    joblib.dump(reducer.pca, pca_pkl_path)
    print(f"\n[SUCCESS] Saved PCA Transformer to: {pca_pkl_path}")
    
    # 4. Save Features
    print("\n[STEP 4] Saving reduced features...")
    np.save(os.path.join(output_dir, 'X_train_pca.npy'), X_train_pca)
    np.save(os.path.join(output_dir, 'X_test_pca.npy'), X_test_pca)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    # 5. Visualize
    print("\n[STEP 5] Creating visualizations...")
    visualizer = FeatureVisualizer()
    reducer.plot_variance_explained(save_path=os.path.join(output_dir, 'variance_explained.png'))
    visualizer.plot_pca_2d(X_train_pca, y_train, save_path=os.path.join(output_dir, 'pca_visualization.png'))
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    return X_train_pca, X_test_pca, y_train, y_test

if __name__ == '__main__':
    # Define Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Go up to root
    DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed_data')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'feature')
    
    if os.path.exists(DATA_DIR):
        extract_and_reduce_features(
            data_dir=DATA_DIR,
            output_dir=OUTPUT_DIR,
            n_components=None,      
            variance_ratio=0.95,    
            model_name='ResNet50'  
        )
    else:
        print(f"[ERROR] Directory not found: {DATA_DIR}")