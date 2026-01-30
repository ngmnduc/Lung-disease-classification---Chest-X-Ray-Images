import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Configuration
IMG_SIZE = 224
RANDOM_STATE = 42

def load_all_data(root_dir):
    """
    Traverses subfolders (train, val, test), then enters (NORMAL, PNEUMONIA) 
    to collect ALL images into a single dataset.
    """
    images = []
    labels = []
    
    # List of top-level folders
    sub_dirs = ['train', 'test', 'val']
    # List of category folders (Labels)
    categories = {'NORMAL': 0, 'PNEUMONIA': 1}
    
    print(f"--- Scanning all data from: {root_dir} ---")
    
    total_count = 0
    
    # Loop 1: Iterate through train, test, val
    for sub_dir in sub_dirs:
        path = os.path.join(root_dir, sub_dir)
        if not os.path.exists(path):
            continue
            
        # Loop 2: Iterate through NORMAL, PNEUMONIA within each folder
        for category, label in categories.items():
            folder_path = os.path.join(path, category)
            if not os.path.exists(folder_path):
                continue
                
            file_list = os.listdir(folder_path)
            for filename in file_list:
                img_path = os.path.join(folder_path, filename)
                # Only keep valid image files
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                        img = img.astype('float32') / 255.0
                        
                        images.append(img)
                        labels.append(label)
                        total_count += 1
                        
    print(f"-> Total images found (combined): {total_count}")
    return np.array(images), np.array(labels)

def augment_and_balance(X_train, y_train):
    """
    Augment ONLY on the TRAIN set after re-splitting.
    """
    unique, counts = np.unique(y_train, return_counts=True)
    class_counts = dict(zip(unique, counts))
    
    print(f"   [INFO] Original Train set: Normal(0)={class_counts.get(0,0)}, Pneumonia(1)={class_counts.get(1,0)}")
    
    max_count = max(class_counts.values())
    
    datagen = ImageDataGenerator(
        rotation_range=15,
        horizontal_flip=True,
        zoom_range=0.1,
        fill_mode='nearest'
    )
    
    X_balanced = list(X_train)
    y_balanced = list(y_train)
    
    for label in class_counts:
        count = class_counts[label]
        diff = max_count - count
        
        if diff > 0:
            print(f"   [AUGMENT] Generating {diff} images for Class {label}...")
            class_indices = np.where(y_train == label)[0]
            X_minority = X_train[class_indices]
            
            aug_iter = datagen.flow(X_minority, batch_size=1, shuffle=True)
            
            generated_count = 0
            while generated_count < diff:
                img = next(aug_iter)[0]
                X_balanced.append(img)
                y_balanced.append(label)
                generated_count += 1
                
    return np.array(X_balanced), np.array(y_balanced)

def process_dataset(root_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Combine all data
    X_all, y_all = load_all_data(root_dir)

    if len(y_all) == 0:
        print("ERROR: No images found. Please check the path.")
        return

    # 2. Re-split according to standard ratio 80 - 10 - 10
    print("\n--- Resplitting data (80-10-10) ---")
    
    # Step 2a: Split 80% Train and 20% Temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=RANDOM_STATE
    )
    
    # Step 2b: Split 20% Temp into 10% Val and 10% Test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
    )
    
    print(f"   Sizes -> Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    # 3. Augment ONLY Train set
    print("\n--- Augmenting Train set ---")
    X_train_aug, y_train_aug = augment_and_balance(X_train, y_train)
    
    unique, counts = np.unique(y_train_aug, return_counts=True)
    print(f"   [DONE] Train set after Augmentation: Normal: {counts[0]}, Pneumonia: {counts[1]}")

    # 4. Save files
    print("\n--- Saving .npy files ---")
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train_aug)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train_aug)
    
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
    
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    
    print(f"COMPLETED. Data saved at: {output_dir}")

if __name__ == "__main__":
    # Automatically get user's home directory
    user_home = os.path.expanduser('~')
    ROOT_DATA_DIR = os.path.join(user_home, 'Downloads', 'chest_xray')
    
    # Output path
    PROCESSED_DATA_DIR = './data/processed'
    
    if os.path.exists(ROOT_DATA_DIR):
        process_dataset(ROOT_DATA_DIR, PROCESSED_DATA_DIR)
    else:
        print(f"Error: Folder not found at {ROOT_DATA_DIR}")
        print("Please check if 'chest_xray' folder is extracted in Downloads.")