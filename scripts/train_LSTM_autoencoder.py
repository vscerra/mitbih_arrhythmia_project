import os
import click
import numpy as np
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense 
from tensorflow.keras.callbacks import EarlyStopping 
from sklearn.model_selection import train_test_split

@click.command()
@click.option('--x-path', default = 'data/processed/LSTM_X_sequences.npy', help = 'Path to LSTM input sequences (.npy)')
@click.option('--y-path', default = 'data/processed/LSTM_y_labels.npy', help = 'Path to sequence labels (.npy)')
@click.option('--output-dir', default = 'models/lstm', help = 'Directory to save model artifacts')

def train_lstm_autoencoder(x_path, y_path, output_dir):
    os.makedirs(output_dir, exist_ok = True)

    # load data
    X = np.load(x_path)
    y = np.load(y_path)
    mask_normal = y == 'N'
    X_normal = X[mask_normal]

    # train/val split
    X_train, X_val = train_test_split(X_normal, test_size = 0.2)

    # Define model
    timesteps, input_dim = X_train.shape[1], X_train.shape[2]
    input_layer = Input(shape = (timesteps, input_dim))
    encoded = LSTM(64, return_sequences = True)(input_layer)
    encoded = LSTM(32)(encoded)

    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(32, return_sequences = True)(decoded)
    decoded = LSTM(64, return_sequences = True)(decoded)
    decoded = TimeDistributed(Dense(input_dim))(decoded)

    model = Model(inputs = input_layer, outputs = decoded)
    model.compile(optimizer='adam', loss='mse')

    # Train
    model.fit(
        X_train, X_train,
        validation_data = (X_val, X_val),
        epochs = 50,
        batch_size = 64,
        callbacks = [EarlyStopping(monitor='val_loss', patience = 5, restore_best_weights=True)]
    )

    # Save model
    model.save(os.path.join(output_dir, "lstm_autoencoder.h5"))
    print(f"Model saved to {output_dir}")

if __name__ == '__main__':
    train_lstm_autoencoder()