import os
import click
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tf.keras.models import Model
from tf.keras.layers import Input, Dense
from tf.keras.optimizers import Adam
from tf.keras.callbacks import EarlyStoppig

# CLI ENTRY POINT
@click.command()
@click.option('--segment-path', default='data/processed/beat_segments.npy', help="Path to full segment array.")
@click.option('--labels-path', default='data/processed/beats_dataset.csv', help='Path to beat labels')
@click.option('--output-dir', default='models/autoencoder/', help='Where to save trained model and scalers.')
def train_autoencoder_cli(segments_path, labels_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # load data
    segments = np.load(segments_path)
    labels_df = pd.read_csv(labels_path)

    # filter for normal beats only ('N')
    mask_normal = labels_df['label'] == 'N'
    X_normal = segments[mask_normal.values]
    X_normal_flat = X_normal.reshape(X_normal.shape[0], -1)

    # train/val split import
    X_train, X_val = train_test_split(X_normal_flat, test_size=0.2, random_state=42)

    # Normalize
    scaler = StandardScaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # save scaler
    np.save(os.path.join(output_dir, "scaler_mean.npy"), scaler.mean_)
    np.save(os.path.join(output_dir, "scaler_scale.npy"), scaler.scale_)

    # build autoencoder
    input_dim = X_train_scaled.shape[1]
    encoding_dim = 32

    input_layer = Input(shape=(input_dim,))
    encoded = Dense(128, activation = 'relu')(input_layer)
    encoded = Dense(encoding_dim, activation = 'relu')(encoded)
    decoded = Dense(128, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='linear')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimzer=Adam(1e-3), loss='mse')

    # Train
    autoencoder.fit(
        X_train_scaled, 
        X_train_scaled,
        validation_data=(X_val_scaled, X_val_scaled),
        epochs=50,
        batch_size=64,
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)]
    )

    # Save model
    autoencoder.save(os.path.join(output_dir, 'autoencoder_model.h5'))
    print(f"Model and scaler saved to: {output_dir}")


if __name__ == '__main__':
    train_autoencoder_cli()



