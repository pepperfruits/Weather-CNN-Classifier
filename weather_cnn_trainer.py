"""
Weather Pattern CNN Trainer
==========================

This script trains a Convolutional Neural Network (CNN) to classify weather patterns from images.
"""

import os
import random
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import VGG16, ResNet50, MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# =============================================================================
# CONFIGURATION SECTION
# =============================================================================

# Data Configuration
DATA_CONFIG = {
    'training_data_path': 'Training Data',
    'image_size': (128, 128),
    'batch_size': 32,
    'validation_split': 0.2,
    'test_split': 0.1,
    'random_state': RANDOM_SEED,
}

# Model Architecture Configuration
MODEL_CONFIG = {
    'model_type': 'custom',
    'num_classes': None,
    'input_shape': (*DATA_CONFIG['image_size'], 3),
    'l2_regularization': 1e-5,
    'use_batch_norm': True,
    'custom_cnn': {
        'conv_layers': [
            {'filters': 32, 'kernel_size': 3, 'activation': 'relu', 'dropout': 0.05},
            {'filters': 64, 'kernel_size': 3, 'activation': 'relu', 'dropout': 0.075},
            {'filters': 128, 'kernel_size': 3, 'activation': 'relu', 'dropout': 0.1},
            {'filters': 256, 'kernel_size': 3, 'activation': 'relu', 'dropout': 0.125}
        ],
        'dense_layers': [
            {'units': 128, 'activation': 'relu', 'dropout': 0.3},
            {'units': 64, 'activation': 'relu', 'dropout': 0.3},
        ],
        'output_activation': 'softmax'
    },
    'transfer_learning': {
        'base_model_trainable': False,
        'fine_tune_layers': 20,
    }
}

# Training Configuration
TRAINING_CONFIG = {
    'epochs': 100,
    'learning_rate': 1e-2,
    'optimizer': 'sgd',
    'loss_function': 'categorical_crossentropy',
    'metrics': ['accuracy'],
    'lr_scheduler': {
        'use_scheduler': True,
        'scheduler_type': 'reduce_on_plateau',
        'patience': 2,
        'factor': 0.5,
        'min_lr': 1e-6,
    },
    'early_stopping': {
        'use_early_stopping': True,
        'patience': 20,
        'restore_best_weights': True,
    },
    'augmentation': {
        'use_augmentation': True,
        'rotation_range': 25,
        'width_shift_range': 0.2,
        'height_shift_range': 0.2,
        'horizontal_flip': True,
        'vertical_flip': False,
        'zoom_range': 0.2,
        'brightness_range': [0.9, 1.1],
        'fill_mode': 'reflect'
    }
}

# Output Configuration
OUTPUT_CONFIG = {
    'save_models_path': 'Saved Models',
    'save_history': True,
    'save_predictions': True,
    'plot_training_curves': True,
    'plot_confusion_matrix': True,
    'model_name_prefix': 'weather_cnn',
}

# =============================================================================
# DATA PREPROCESSING SECTION
# =============================================================================

def load_and_preprocess_data(data_path, image_size, validation_split, test_split, random_state):
    """
    Load and preprocess image data from a folder structure.
    Returns train, validation, and test splits along with class names.
    """
    print("=" * 60)
    print("DATA PREPROCESSING SECTION")
    print("=" * 60)
    class_names = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    class_names.sort()
    print(f"Found {len(class_names)} weather classes: {class_names}")
    images = []
    labels = []
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(data_path, class_name)
        image_files = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        print(f"Loading {len(image_files)} images from class '{class_name}'")
        for image_file in image_files:
            image_path = os.path.join(class_path, image_file)
            try:
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, image_size)
                    image = image.astype(np.float32)
                    images.append(image)
                    labels.append(class_idx)
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
    X = np.array(images)
    y = np.array(labels)
    print(f"Total images loaded: {len(X)}")
    print(f"Image shape: {X.shape}")
    print(f"Label shape: {y.shape}")
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_split, random_state=random_state, stratify=y)
    val_split_adjusted = validation_split / (1 - test_split)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_split_adjusted, random_state=random_state, stratify=y_temp)
    y_train_cat = to_categorical(y_train, num_classes=len(class_names))
    y_val_cat = to_categorical(y_val, num_classes=len(class_names))
    y_test_cat = to_categorical(y_test, num_classes=len(class_names))
    print(f"Training set: {X_train.shape[0]} images")
    print(f"Validation set: {X_val.shape[0]} images")
    print(f"Test set: {X_test.shape[0]} images")
    return X_train, X_val, X_test, y_train_cat, y_val_cat, y_test_cat, class_names

def create_data_generators(X_train, y_train, X_val, y_val, config):
    """
    Create data generators for training and validation, with optional augmentation.
    """
    print("\nCreating data generators...")
    if config['augmentation']['use_augmentation']:
        train_datagen = ImageDataGenerator(
            rotation_range=config['augmentation']['rotation_range'],
            width_shift_range=config['augmentation']['width_shift_range'],
            height_shift_range=config['augmentation']['height_shift_range'],
            horizontal_flip=config['augmentation']['horizontal_flip'],
            vertical_flip=config['augmentation']['vertical_flip'],
            zoom_range=config['augmentation']['zoom_range'],
            brightness_range=config['augmentation']['brightness_range'],
            fill_mode=config['augmentation']['fill_mode'],
            rescale=1./255
        )
        print("Applied data augmentation for training")
    else:
        train_datagen = ImageDataGenerator(rescale=1./255)
        print("No data augmentation applied")
    val_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow(X_train, y_train, batch_size=DATA_CONFIG['batch_size'], shuffle=True)
    val_generator = val_datagen.flow(X_val, y_val, batch_size=DATA_CONFIG['batch_size'], shuffle=False)
    return train_generator, val_generator

# =============================================================================
# MODEL ARCHITECTURE SECTION
# =============================================================================

def create_custom_cnn(input_shape, num_classes, config):
    print("Creating custom CNN architecture (reduced dropout)...")
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))
    for i, conv_cfg in enumerate(config['custom_cnn']['conv_layers']):
        model.add(layers.Conv2D(
            filters=conv_cfg['filters'],
            kernel_size=conv_cfg['kernel_size'],
            activation=conv_cfg['activation'],
            padding='same',
            kernel_regularizer=tf.keras.regularizers.l2(config['l2_regularization']),
            name=f'conv_{i+1}'
        ))
        if config.get('use_batch_norm', True):
            model.add(layers.BatchNormalization(name=f'bn_conv_{i+1}'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), name=f'pool_{i+1}'))
        model.add(layers.Dropout(conv_cfg.get('dropout', 0.10), name=f'dropout_conv_{i+1}'))
    model.add(layers.GlobalAveragePooling2D(name='gap'))
    for i, dense_cfg in enumerate(config['custom_cnn']['dense_layers']):
        model.add(layers.Dense(
            units=dense_cfg['units'],
            activation=dense_cfg['activation'],
            kernel_regularizer=tf.keras.regularizers.l2(config['l2_regularization']),
            name=f'dense_{i+1}'
        ))
        if config.get('use_batch_norm', True):
            model.add(layers.BatchNormalization(name=f'bn_dense_{i+1}'))
        if 'dropout' in dense_cfg:
            model.add(layers.Dropout(dense_cfg.get('dropout', 0.30), name=f'dropout_dense_{i+1}'))
    model.add(layers.Dense(num_classes, activation=config['custom_cnn']['output_activation'], name='output'))
    return model

def create_transfer_learning_model(input_shape, num_classes, config):
    """
    Create a model using transfer learning with pre-trained architectures.
    """
    print(f"Creating transfer learning model with {config['model_type']}...")
    if config['model_type'] == 'vgg16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif config['model_type'] == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif config['model_type'] == 'mobilenet':
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Unsupported model type: {config['model_type']}")
    base_model.trainable = config['transfer_learning']['base_model_trainable']
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def create_model(input_shape, num_classes, model_config, training_config):
    print("=" * 60)
    print("MODEL ARCHITECTURE SECTION")
    print("=" * 60)
    if model_config['model_type'] == 'custom':
        model = create_custom_cnn(input_shape, num_classes, model_config)
    else:
        model = create_transfer_learning_model(input_shape, num_classes, model_config)
    if training_config['optimizer'] == 'adam':
        optimizer = optimizers.Adam(learning_rate=training_config['learning_rate'])
    elif training_config['optimizer'] == 'sgd':
        optimizer = optimizers.SGD(learning_rate=training_config['learning_rate'], momentum=0.9)
    elif training_config['optimizer'] == 'rmsprop':
        optimizer = optimizers.RMSprop(learning_rate=training_config['learning_rate'])
    else:
        raise ValueError(f"Unsupported optimizer: {training_config['optimizer']}")
    model.compile(
        optimizer=optimizer,
        loss=focal_loss(alpha=0.25, gamma=2.0),
        metrics=training_config['metrics']
    )
    model.summary()
    return model

# =============================================================================
# TRAINING SECTION
# =============================================================================

class SaveTrainingProgress(callbacks.Callback):
    """
    Custom callback to save training progress plots to files every few epochs.
    """
    def __init__(self, save_path, plot_interval=5):
        super().__init__()
        self.plot_interval = plot_interval
        self.save_path = save_path
        self.epoch = 0
    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch + 1
        if self.epoch % self.plot_interval == 0:
            self.save_training_curves()
    def save_training_curves(self):
        """Save training and validation curves to file."""
        try:
            history = self.model.history.history
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            fig.suptitle(f'Training Progress - Epoch {self.epoch}', fontsize=16, fontweight='bold')
            if 'accuracy' in history and 'val_accuracy' in history:
                epochs = range(1, len(history['accuracy']) + 1)
                ax1.plot(epochs, history['accuracy'], 'b-', label='Training Accuracy', linewidth=2)
                ax1.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
                ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
                ax1.set_xlabel('Epoch', fontsize=12)
                ax1.set_ylabel('Accuracy', fontsize=12)
                ax1.legend(fontsize=10)
                ax1.grid(True, alpha=0.3)
                ax1.set_ylim(0, 1)
                ax1.set_xlim(1, max(epochs))
            if 'loss' in history and 'val_loss' in history:
                epochs = range(1, len(history['loss']) + 1)
                ax2.plot(epochs, history['loss'], 'b-', label='Training Loss', linewidth=2)
                ax2.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
                ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
                ax2.set_xlabel('Epoch', fontsize=12)
                ax2.set_ylabel('Loss', fontsize=12)
                ax2.legend(fontsize=10)
                ax2.grid(True, alpha=0.3)
                ax2.set_xlim(1, max(epochs))
            plot_path = os.path.join(self.save_path, f'training_progress_epoch_{self.epoch}.png')
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"Training progress saved to: {plot_path}")
        except Exception as e:
            print(f"Error saving training curves: {e}")
    def on_train_end(self, logs=None):
        self.save_training_curves()

def create_callbacks(config, save_path):
    """
    Create training callbacks for model checkpointing, learning rate scheduling, and early stopping.
    """
    callbacks_list = []
    save_cb = SaveTrainingProgress(save_path, plot_interval=5)
    callbacks_list.append(save_cb)
    chk_path = os.path.join(save_path, f"{OUTPUT_CONFIG['model_name_prefix']}_best.h5")
    checkpoint = callbacks.ModelCheckpoint(
        filepath=chk_path,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        verbose=1
    )
    callbacks_list.append(checkpoint)
    if config['lr_scheduler']['use_scheduler'] and config['lr_scheduler']['scheduler_type']=='reduce_on_plateau':
        lr_scheduler = callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=config['lr_scheduler']['factor'],
            patience=config['lr_scheduler']['patience'],
            min_lr=config['lr_scheduler']['min_lr'],
            verbose=1
        )
        callbacks_list.append(lr_scheduler)
    if config['early_stopping']['use_early_stopping']:
        early_stop = callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=config['early_stopping']['patience'],
            restore_best_weights=config['early_stopping']['restore_best_weights'],
            verbose=1
        )
        callbacks_list.append(early_stop)
    return callbacks_list


def train_model(model, train_generator, val_generator, config, save_path):
    print("=" * 60)
    print("TRAINING SECTION")
    print("=" * 60)
    steps_per_epoch = len(train_generator)
    validation_steps = len(val_generator)
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Validation steps: {validation_steps}")
    print(f"Training for {config['epochs']} epochs...")
    y_train_raw = np.argmax(train_generator.y, axis=1)
    classes = np.unique(y_train_raw)
    weights = compute_class_weight('balanced', classes=classes, y=y_train_raw)
    class_weight = {int(cls): float(w) for cls, w in zip(classes, weights)}
    print(f"Using class weights: {class_weight}")
    callbacks_list = create_callbacks(config, save_path)
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=config['epochs'],
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=callbacks_list,
        class_weight=class_weight,
        verbose=1
    )
    return history

# =============================================================================
# EVALUATION AND VISUALIZATION SECTION
# =============================================================================

def evaluate_model(model, X_test, y_test, class_names, save_path, config):
    """
    Evaluate the trained model on test data and output accuracy, classification report, and confusion matrix.
    """
    print("=" * 60)
    print("EVALUATION SECTION")
    print("=" * 60)
    X_test_norm = X_test.astype('float32') / 255.0
    y_pred_proba = model.predict(X_test_norm)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)
    accuracy = np.mean(y_pred == y_true)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    if config['save_predictions']:
        predictions_df = pd.DataFrame({
            'true_label': y_true,
            'predicted_label': y_pred,
            'true_class': [class_names[i] for i in y_true],
            'predicted_class': [class_names[i] for i in y_pred],
            'confidence': np.max(y_pred_proba, axis=1)
        })
        predictions_path = os.path.join(save_path, 'test_predictions.csv')
        predictions_df.to_csv(predictions_path, index=False)
        print(f"Predictions saved to: {predictions_path}")
    if config['plot_confusion_matrix']:
        plot_confusion_matrix(y_true, y_pred, class_names, save_path)
    return accuracy, y_pred_proba

def plot_training_curves(history, save_path):
    """
    Plot training and validation accuracy/loss curves.
    """
    print("Plotting training curves...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(history.history['loss'], label='Training Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    plot_path = os.path.join(save_path, 'training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training curves saved to: {plot_path}")

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """
    Plot and save the confusion matrix.
    """
    print("Plotting confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plot_path = os.path.join(save_path, 'confusion_matrix.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Confusion matrix saved to: {plot_path}")

def focal_loss(alpha=0.25, gamma=2.0):
    """
    Focal Loss for multi-class classification.
    """
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        ce = -y_true * K.log(y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        weight = alpha * K.pow(1 - p_t, gamma)
        fl = weight * ce
        return K.sum(fl, axis=1)
    return loss

# =============================================================================
# MAIN EXECUTION SECTION
# =============================================================================

def main():
    """
    Main function to execute the complete CNN training pipeline.
    """
    print("=" * 80)
    print("WEATHER PATTERN CNN TRAINER")
    print("=" * 80)
    os.makedirs(OUTPUT_CONFIG['save_models_path'], exist_ok=True)
    if not os.path.exists(DATA_CONFIG['training_data_path']):
        print(f"Error: Training data path '{DATA_CONFIG['training_data_path']}' does not exist!")
        print("Please create a 'Training Data' folder with weather pattern subfolders.")
        return
    X_train, X_val, X_test, y_train, y_val, y_test, class_names = load_and_preprocess_data(
        DATA_CONFIG['training_data_path'],
        DATA_CONFIG['image_size'],
        DATA_CONFIG['validation_split'],
        DATA_CONFIG['test_split'],
        DATA_CONFIG['random_state']
    )
    MODEL_CONFIG['num_classes'] = len(class_names)
    MODEL_CONFIG['input_shape'] = (*DATA_CONFIG['image_size'], 3)
    train_generator, val_generator = create_data_generators(
        X_train, y_train, X_val, y_val, TRAINING_CONFIG
    )
    model = create_model(
        MODEL_CONFIG['input_shape'],
        MODEL_CONFIG['num_classes'],
        MODEL_CONFIG,
        TRAINING_CONFIG
    )
    history = train_model(
        model, train_generator, val_generator, 
        TRAINING_CONFIG, OUTPUT_CONFIG['save_models_path']
    )
    final_model_path = os.path.join(
        OUTPUT_CONFIG['save_models_path'], 
        f"{OUTPUT_CONFIG['model_name_prefix']}_final.h5"
    )
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")
    if OUTPUT_CONFIG['save_history']:
        history_path = os.path.join(OUTPUT_CONFIG['save_models_path'], 'training_history.npy')
        np.save(history_path, history.history)
        print(f"Training history saved to: {history_path}")
    accuracy, predictions = evaluate_model(
        model, X_test, y_test, class_names, 
        OUTPUT_CONFIG['save_models_path'], OUTPUT_CONFIG
    )
    if OUTPUT_CONFIG['plot_training_curves']:
        plot_training_curves(history, OUTPUT_CONFIG['save_models_path'])
    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"Final Test Accuracy: {accuracy:.4f}")
    print(f"Models saved in: {OUTPUT_CONFIG['save_models_path']}")
    print("=" * 80)

if __name__ == "__main__":
    main() 