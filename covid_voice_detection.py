#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import tarfile

def extract_tar(tar_path, extract_path):

    date = os.path.basename(tar_path).split('_')[0]
    expected_csv = os.path.join(extract_path, f"{date}.csv")
    
    if os.path.exists(expected_csv):
        print(f"Files already extracted in {extract_path}, skipping extraction...")
        return extract_path
        
    print(f"Extracting files to {extract_path}...")
    with tarfile.open(tar_path, 'r:*') as tar:
        tar.extractall(extract_path)
    return extract_path

class CoswaraProcessor:
    def __init__(self, base_path="coswara_data", samples_per_class=100):
        self.base_path = base_path
        self.samples_per_class = samples_per_class
        self.healthy_count = 0
        self.covid_count = 0

    def combine_tar_parts(self, date_folder):
        date = os.path.basename(date_folder)
        combined = os.path.join(date_folder, f"{date}_combined.tar.gz")
        
        if os.path.exists(combined):
            print(f"Combined tar file already exists: {combined}")
            return combined
            
        print(f"Combining tar parts for {date}...")
        with open(combined, 'wb') as outfile:
            for part in sorted(f for f in os.listdir(date_folder) if f.startswith(f"{date}.tar")):
                with open(os.path.join(date_folder, part), 'rb') as infile:
                    outfile.write(infile.read())
        return combined

    def process_audio(self, file_path):
        try:
            y, sr = librosa.load(file_path, sr=44100, duration=5)
            
            # Basic features
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_db = librosa.power_to_db(mel_spec, ref=np.max)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            
            # Standardize sizes
            target_length = 128
            mel_db = librosa.util.fix_length(mel_db, size=target_length, axis=1)
            mfcc = librosa.util.fix_length(mfcc, size=target_length, axis=1)
            
            return {
                'mel': mel_db.astype(np.float32),
                'mfcc': mfcc.astype(np.float32)
            }
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    def process_folder(self, date_folder):
        date = os.path.basename(date_folder)
        print(f"\nProcessing {date}")
        
        metadata_path = os.path.join(date_folder, f"{date}.csv")
        if not os.path.exists(metadata_path):
            return [], [], []
            
        metadata_df = pd.read_csv(metadata_path)
       
        metadata_df = metadata_df.sample(frac=1, random_state=42)
        
        extract_path = os.path.join(date_folder, "extracted")
        
        if os.path.exists(extract_path) and len(os.listdir(extract_path)) > 0:
            print(f"Using existing extracted files in {extract_path}")
        else:
            os.makedirs(extract_path, exist_ok=True)
            combined_tar = self.combine_tar_parts(date_folder)
            extract_tar(combined_tar, extract_path)
            if not os.path.exists(combined_tar):
                os.remove(combined_tar)
        
        features = []
        labels = []
        meta = []

        for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc=f"Processing {date} data"):
            # Vérifier si on a atteint le nombre d'échantillons souhaité pour les deux classes
            if self.healthy_count >= self.samples_per_class and self.covid_count >= self.samples_per_class:
                break

            try:
                # Déterminer la classe et vérifier les quotas
                if row['covid_status'] in ['healthy', 'recovered_full']:
                    if self.healthy_count >= self.samples_per_class:
                        continue
                    label = 0
                elif row['covid_status'] in ['positive_mild', 'positive_moderate', 'positive_asymp',
                                           'no_resp_illness_exposed', 'resp_illness_not_identified']:
                    if self.covid_count >= self.samples_per_class:
                        continue
                    label = 1
                else:
                    continue

                user_path = os.path.join(extract_path, str(row['id']))
                if not os.path.exists(user_path):
                    user_path = os.path.join(extract_path, date, str(row['id']))
                    if not os.path.exists(user_path):
                        continue
                
                audio_files = {
                    'cough': sorted([f for f in os.listdir(user_path) if f.startswith('cough')])[:2],
                    'breathing': sorted([f for f in os.listdir(user_path) if f.startswith('breathing')])[:2],
                    'voice': sorted([f for f in os.listdir(user_path) if f.startswith(('counting', 'vowel'))])[:3]
                }
                
                if not all(len(files) >= min_count for files, min_count in [
                    (audio_files['cough'], 2), 
                    (audio_files['breathing'], 2), 
                    (audio_files['voice'], 3)
                ]):
                    continue
                
                user_features = {}
                for category, files in audio_files.items():
                    for audio_file in files:
                        result = self.process_audio(os.path.join(user_path, audio_file))
                        if result is None:
                            continue
                        user_features[f"{category}_{audio_file}"] = result
                
                if len(user_features) >= 7:
                    features.append(user_features)
                    labels.append(label)
                    meta.append({
                        'id': row['id'],
                        'age': row['a'],
                        'gender': row['g'],
                        'status': row['covid_status'],
                        'date': date
                    })
                    
                    # Incrémenter le compteur approprié
                    if label == 0:
                        self.healthy_count += 1
                    else:
                        self.covid_count += 1
                    
            except Exception as e:
                print(f"Error processing user {row['id']}: {e}")
                continue

        print(f"Current counts - Healthy: {self.healthy_count}, Covid: {self.covid_count}")
        return features, labels, meta

    def load_dataset(self):
        all_features = []
        all_labels = []
        all_metadata = []
        
        date_folders = sorted(f for f in os.listdir(self.base_path) if f.startswith("2020"))
        
        for date in date_folders:
            if self.healthy_count >= self.samples_per_class and self.covid_count >= self.samples_per_class:
                break
                
            date_path = os.path.join(self.base_path, date)
            features, labels, metadata = self.process_folder(date_path)
            
            if features:
                all_features.extend(features)
                all_labels.extend(labels)
                all_metadata.extend(metadata)
                
            print(f"Processed {len(features)} valid samples from {date}")
            
        return all_features, np.array(all_labels), all_metadata

def create_model(input_shapes):
    inputs = []
    features = []
    
    for name, shape in input_shapes.items():
        input_layer = layers.Input(shape=shape, name=name)
        
        x = layers.Conv2D(64, 3, activation='relu', padding='same')(input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        
        x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        
        x = layers.GlobalAveragePooling2D()(x)
        
        inputs.append(input_layer)
        features.append(x)
    
    if len(features) > 1:
        x = layers.concatenate(features)
    else:
        x = features[0]
        
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    # Initialiser le processeur avec 100 échantillons par classe
    processor = CoswaraProcessor(samples_per_class=100)
    features, labels, metadata = processor.load_dataset()
    
    if not features:
        print("No valid samples found. Exiting...")
        return

    # Vérifier l'équilibre des classes
    unique, counts = np.unique(labels, return_counts=True)
    print("\nClass distribution:")
    for u, c in zip(unique, counts):
        print(f"Class {u}: {c} samples")

    input_features = {}
    for audio_type in features[0].keys():
        for feat_type in ['mel', 'mfcc']:
            name = f"{audio_type}_{feat_type}"
            feat_array = np.stack([f[audio_type][feat_type] for f in features])
            feat_array = np.expand_dims(feat_array, -1)
            
            if feat_array.shape[1] > 64:
                feat_array = feat_array[:, :64, :64, :]
                
            input_features[name] = feat_array

    train_features = {}
    test_features = {}
    first_split = True
    
    for name, feat in input_features.items():
        if first_split:
            train_data, test_data, train_labels, test_labels = train_test_split(
                feat, labels, test_size=0.2, random_state=42, stratify=labels
            )
            first_split = False
        else:
            train_data, test_data = train_test_split(
                feat, test_size=0.2, random_state=42
            )
        train_features[name] = train_data
        test_features[name] = test_data

    print("\nFeature shapes:")
    for name, feat in train_features.items():
        print(f"{name}: {feat.shape}")

    input_shapes = {name: feat.shape[1:] for name, feat in train_features.items()}
    model = create_model(input_shapes)
    model.summary()
    
    
    history = model.fit(
        train_features,
        train_labels,
        validation_split=0.2,
        epochs=20,  
        batch_size=8,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=0.00001
            )
        ]
    )

 
    predictions = model.predict(test_features) > 0.5
    
    os.makedirs('results', exist_ok=True)
    
    with open('results/classification_report.txt', 'w') as f:
        f.write(classification_report(test_labels, predictions))
    
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(test_labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy', 'COVID'],
                yticklabels=['Healthy', 'COVID'])
    plt.title('Confusion Matrix')
    plt.savefig('results/confusion_matrix.png')
    plt.close()
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/training_history.png')
    plt.close()
    
    model.save('results/covid_model.keras')

if __name__ == "__main__":
    main()