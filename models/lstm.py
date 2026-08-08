from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import numpy as np

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    mode="min",
    restore_best_weights=True,
    verbose=1
)


def create_sequences(X,y, sequence_length=24):
    X = np.asarray(X)
    y = np.asarray(y)
    X_seq=[]
    y_seq=[]
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i + sequence_length])
        y_seq.append(y[i + sequence_length])
    return np.array(X_seq), np.array(y_seq)

def train_lstm(X_train, y_train):

    model = Sequential()

    model.add(
        LSTM(
            units=64,
            input_shape=(X_train.shape[1], X_train.shape[2])
        )
    )

    model.add(Dense(32, activation="relu"))

    model.add(Dense(1))

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )

    model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )
    return model

def preprocess_lstm_feat(X_train,X_test,y_train,y_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_train_seq, y_train_seq = create_sequences(X_train,y_train)
    X_test_seq, y_test_seq = create_sequences(X_test,y_test)
    return X_train_seq, X_test_seq, y_train_seq, y_test_seq 
