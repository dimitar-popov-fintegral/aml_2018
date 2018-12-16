import os
import pandas as pd
import data as dt
import numpy as np
from matplotlib import pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Flatten, TimeDistributed, LSTM
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import balanced_accuracy_score


from enum import Enum
#######################################################################
class ModelType(Enum):
    CNN = 'cnn'
    CNN_LSTM = 'cnn_lstm'


TIME_STEPS = 10


#######################################################################
def create_cnn_model(input_shape, n_classes):
    model = Sequential()

    model.add(Conv1D(32, 3, padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv1D(32, 3, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Dropout(0.25))

    model.add(Conv1D(64, 3, padding='same', activation='relu'))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Dropout(0.25))

    model.add(Conv1D(64, 3, padding='same', activation='relu'))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model


#######################################################################
def create_cnn_lstm_model(input_shape, n_classes):

    print('input data shape : ', input_shape)

    model = Sequential()

    model.add(TimeDistributed(Conv1D(32, 3, padding='same', activation='relu'), input_shape=input_shape))
    model.add(TimeDistributed(Conv1D(32, 3, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(pool_size=4)))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv1D(64, 3, padding='same', activation='relu')))
    model.add(TimeDistributed(Conv1D(64, 3, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(pool_size=4)))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Conv1D(64, 3, padding='same', activation='relu')))
    model.add(TimeDistributed(Conv1D(64, 3, activation='relu')))
    model.add(TimeDistributed(MaxPooling1D(pool_size=4)))
    model.add(TimeDistributed(Dropout(0.25)))

    model.add(TimeDistributed(Flatten()))
    model.add(LSTM(5, return_sequences=True))
    model.add(TimeDistributed(Dense(512, activation='relu')))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(Dense(n_classes, activation='softmax')))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model


#######################################################################
def train_model(x_train, y_train, x_validate, y_validate, epochs, type, model_path=None):

    num_train_samples = x_train.shape[0]
    num_validate_samples = x_validate.shape[0]

    print('Training data shape : ', x_train.shape, y_train.shape)
    print('Testing data shape : ', x_validate.shape, y_validate.shape)

    # Find the unique numbers from the train labels
    classes = np.unique(y_train)
    n_classes = len(classes)
    print('Total number of outputs : ', n_classes)
    print('Output classes : ', classes)

    # Change the labels from integer to categorical data
    y_train_one_hot = to_categorical(y_train)
    y_validate_one_hot = to_categorical(y_validate)

    time_steps = TIME_STEPS
    if type == ModelType.CNN_LSTM:
        assert num_train_samples % time_steps == 0, "total number of train samples must divide by number of time steps"
        x_train = x_train.reshape((int(num_train_samples/time_steps), time_steps, *x_train.shape[1:]))
        y_train_one_hot = y_train_one_hot.reshape((int(num_train_samples/time_steps), time_steps, *y_train_one_hot.shape[1:]))

        assert num_validate_samples % time_steps == 0, "total number of validate samples must divide by number of time steps"
        x_validate = x_validate.reshape((int(num_validate_samples/time_steps), time_steps, *x_validate.shape[1:]))
        y_validate_one_hot = y_validate_one_hot.reshape((int(num_validate_samples/time_steps), time_steps, *y_validate_one_hot.shape[1:]))


    input_shape = x_train.shape[1:]
    print('input shape: ', input_shape)

    if type == ModelType.CNN:
        model = create_cnn_model(input_shape, n_classes)
        batch_size = 256
    elif type == ModelType.CNN_LSTM:
        model = create_cnn_lstm_model(input_shape, n_classes)
        batch_size = 32
    else:
        raise ValueError("Unrecognized model type: %s" % type)

    history = model.fit(x_train, y_train_one_hot, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_validate, y_validate_one_hot))
    model.evaluate(x_validate, y_validate_one_hot)

    plt.figure(figsize=[8,6])
    plt.plot(history.history['loss'],'r',linewidth=3.0)
    plt.plot(history.history['val_loss'],'b',linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)

    plt.figure(figsize=[8,6])
    plt.plot(history.history['acc'],'r',linewidth=3.0)
    plt.plot(history.history['val_acc'],'b',linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves',fontsize=16)

    y_pred = np.argmax(model.predict(x_validate).reshape(num_validate_samples,y_validate_one_hot.shape[-1]), axis=1)
    BMAC = balanced_accuracy_score(y_validate.flatten(), y_pred)
    print("BMAC: %s" % (BMAC))

    if model_path is not None:
        model.save(model_path)

    return model


#######################################################################
def train(eeg1, eeg2, emg, labels, validate_size, epochs, label, type):

    eeg1_ = df_row_norm(eeg1.fillna(0)).values
    eeg2_ = df_row_norm(eeg2.fillna(0)).values
    emg_ = df_row_norm(emg.fillna(0)).values
    labels_ = (labels - 1).values

    X = np.dstack((eeg1_, eeg2_, emg_))
    Y = labels_

    x_validate = X[-validate_size:,:].copy()
    y_validate = Y[-validate_size:,:].copy()

    x_train = X[:-validate_size,:]
    y_train = Y[:-validate_size,:]

    model_path = os.path.join(dt.output_dir(), "%s.h5" % label)
    model = train_model(x_train, y_train, x_validate, y_validate, epochs=epochs, type=type, model_path=model_path)

    return model


#######################################################################
def predict(X_test_A, X_test_B, model, type, weights, label):

    num_samples = X_test_A.shape[0]
    assert X_test_A.shape[0] == num_samples

    if type == ModelType.CNN_LSTM:
        assert len(model.input_shape) == 4
        time_steps = TIME_STEPS
        assert model.input_shape[1] == time_steps

        assert num_samples % time_steps == 0, "total number of samples must divide by number of time steps"

        X_test_A = X_test_A.reshape((int(num_samples/time_steps), time_steps, *X_test_A.shape[1:]))
        X_test_B = X_test_B.reshape((int(num_samples/time_steps), time_steps, *X_test_B.shape[1:]))


    y_score_A = model.predict(X_test_A).reshape(num_samples, model.output_shape[-1]) * weights
    y_score_B = model.predict(X_test_B).reshape(num_samples, model.output_shape[-1]) * weights

    y_score = np.concatenate((y_score_A, y_score_B))
    y_test = np.argmax(y_score, axis=1)

    result = pd.Series(y_test)

    expected = [0.526, 0.418, 0.0548]
    for i in range(3):
        print("class expected/realized class ratio [%s]: [%s/%s]" % (i, expected[i], sum(result==i)/len(result)))
    print("")

    result += 1
    result.index.name = 'Id'
    result.name = 'y'
    pd.DataFrame(result).to_csv(os.path.join(dt.output_dir(), "%s.csv" % label))

    return y_score


#######################################################################
def df_row_norm(input_df):
    df = input_df.transpose()
    return ((df - df.mean()) / (df.max() - df.min())).transpose()


#######################################################################
def main():

    N = 21600
    validate_size = 2000
    epochs = 50
    type = ModelType.CNN

    ###################################
    ### Read train data and fit models
    ###################################

    eeg1 = pd.read_csv(os.path.join(dt.data_dir(), 'task5', 'train_eeg1.csv'), header=0, index_col=0)
    eeg2 = pd.read_csv(os.path.join(dt.data_dir(), 'task5', 'train_eeg2.csv'), header=0, index_col=0)
    emg = pd.read_csv(os.path.join(dt.data_dir(), 'task5', 'train_emg.csv'), header=0, index_col=0)
    labels = pd.read_csv(os.path.join(dt.data_dir(), 'task5', 'train_labels.csv'), header=0, index_col=0)

    ##########################
    ### subject one model
    start = 0
    end = N
    label = 'subject1_%s_%s_epochs' % (type, epochs)
    subject1_model = train(eeg1=eeg1.iloc[start:end, :], eeg2=eeg2.iloc[start:end, :], emg=emg.iloc[start:end, :], labels=labels.iloc[start:end, :],
                           type=type, validate_size=validate_size, epochs=epochs, label=label)

    ##########################
    ### subject two model
    start = N
    end = N*2
    label = 'subject2_%s_%s_epochs' % (type, epochs)
    subject2_model = train(eeg1=eeg1.iloc[start:end, :], eeg2=eeg2.iloc[start:end, :], emg=emg.iloc[start:end, :], labels=labels.iloc[start:end, :],
                           type=type, validate_size=validate_size, epochs=epochs, label=label)

    ##########################
    ### subject three model
    start = N*2
    end = N*3-500
    label = 'subject3_%s_%s_epochs' % (type, epochs)
    subject3_model = train(eeg1=eeg1.iloc[start:end, :], eeg2=eeg2.iloc[start:end, :], emg=emg.iloc[start:end, :], labels=labels.iloc[start:end, :],
                           type=type, validate_size=validate_size, epochs=epochs, label=label)


    ##############################################
    ### Models fitted, read test data and predict
    ##############################################

    eeg1_test = pd.read_csv(os.path.join(dt.data_dir(), 'task5', 'test_eeg1.csv'), header=0, index_col=0)
    eeg2_test = pd.read_csv(os.path.join(dt.data_dir(), 'task5', 'test_eeg2.csv'), header=0, index_col=0)
    emg_test = pd.read_csv(os.path.join(dt.data_dir(), 'task5', 'test_emg.csv'), header=0, index_col=0)
    eeg1_test_A = df_row_norm(eeg1_test.iloc[:N, :].fillna(0)).values
    eeg2_test_A = df_row_norm(eeg2_test.iloc[:N, :].fillna(0)).values
    emg_test_A = df_row_norm(emg_test.iloc[:N, :].fillna(0)).values

    eeg1_test_B = df_row_norm(eeg1_test.iloc[N:, :].fillna(0)).values
    eeg2_test_B = df_row_norm(eeg2_test.iloc[N:, :].fillna(0)).values
    emg_test_B = df_row_norm(emg_test.iloc[N:, :].fillna(0)).values

    X_test_A = np.dstack((eeg1_test_A, eeg2_test_A, emg_test_A))
    X_test_B = np.dstack((eeg1_test_B, eeg2_test_B, emg_test_B))

    #################################
    ### subject one model prediction
    label = 'subject_1_%s_weighted_%s_epochs' % (type, epochs)
    y_subject1_score = predict(X_test_A, X_test_B, model=subject1_model, type=type, weights=[1, 0.5, 2.5], label=label)

    #################################
    ### subject two model prediction
    label = 'subject_2_%s_weighted_%s_epochs' % (type, epochs)
    y_subject2_score = predict(X_test_A, X_test_B, model=subject2_model, type=type, weights=[1, 0.5, 2.0], label=label)

    ###################################
    ### subject three model prediction
    label = 'subject_3_%s_weighted_%s_epochs' % (type, epochs)
    y_subject3_score = predict(X_test_A, X_test_B, model=subject3_model, type=type, weights=[1, 0.5, 4.5], label=label)

    ##################################
    ### all subjects model prediction
    label = 'all_subjects_%s_%s_epochs' % (type, epochs)
    y_score = y_subject1_score * 0.33 + y_subject2_score * 0.33 + y_subject3_score * 0.33
    y_test = np.argmax(y_score, axis=1)

    result = pd.Series(y_test)

    expected = [0.526, 0.418, 0.0548]
    for i in range(3):
        print("class expected/realized class ratio [%s]: [%s/%s]" % (i, expected[i], sum(result==i)/len(result)))
    print("")

    result += 1
    result.index.name = 'Id'
    result.name = 'y'
    pd.DataFrame(result).to_csv(os.path.join(dt.output_dir(), "%s.csv" % label))


    ##################################
    ### all subjects model prediction
    label = 'all_subjects_%s_weighted_%s_epochs' % (type, epochs)
    y_score = (y_subject1_score * 0 + y_subject2_score * 0.5 + y_subject3_score * 0.5) * [1.5, 0.8, 1.6]
    y_test = np.argmax(y_score, axis=1)

    result = pd.Series(y_test)

    expected = [0.526, 0.418, 0.0548]
    for i in range(3):
        print("class expected/realized class ratio [%s]: [%s/%s]" % (i, expected[i], sum(result==i)/len(result)))
    print("")

    result += 1
    result.index.name = 'Id'
    result.name = 'y'
    pd.DataFrame(result).to_csv(os.path.join(dt.output_dir(), "%s.csv" % label))


    print("DONE")


#######################################################################
if __name__ == "__main__":

    main()

