import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, LeakyReLU, BatchNormalization, Reshape, Conv1D, Flatten
from tensorflow.keras.models import Sequential, load_model
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.regularizers import l1
from tensorflow.keras.optimizers import Adam


import sklearn.metrics 

from tensorflow.keras import backend as K


class Gan():
    def __init__(self, m=3, tau=1, learning_rate_d=0.0002, learning_rate_g=0.0002,
                 lambda_p=1, lambda_adv=1, lambda_dpl=1, batch_size=32, epochs=500,
                 patience=5, min_delta=0.001, strides=2, lrelu_alpha=0.01,
                 batchnorm_epsilon=1e-05, batchnorm_momentum=0.9, kernal_size=5,
                 dropout=0.2, window_size=250):
        self.nist = 50
        self.m = m
        self.tau = tau
        self.learning_rate_d = learning_rate_d
        self.learning_rate_g = learning_rate_g
        self.lambda_p = lambda_p
        self.lambda_adv = lambda_adv
        self.lambda_dpl = lambda_dpl
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.min_delta = min_delta
        self.strides = strides
        self.lrelu_alpha = lrelu_alpha
        self.batchnorm_epsilon = batchnorm_epsilon
        self.batchnorm_momentum = batchnorm_momentum
        self.kernal_size = kernal_size
        self.dropout = dropout
        self.window_size = window_size
        
        self.df = None
        self.rows = None
        self.cols = None
        self.train_size = None
        self.test_size = None

    def load(self, prcsSoFar):
        data_load = np.array(prcsSoFar).T
        column_names = [f'stock_{i+1}' for i in range(self.nist)]
        self.df = pd.DataFrame(data_load, columns=column_names)
        self.rows = self.df.shape[0] - self.m + 1
        self.cols = self.df.shape[1]

    def embed_phase_space(self, data):
        n = len(data)
        embedded_data = np.zeros((n - (self.m - 1) * self.tau, self.m))
        for i in range(self.m):
            embedded_data[:, i] = data[i * self.tau: n - (self.m - 1) * self.tau + i * self.tau]

        return embedded_data

    def normalize_delay_vectors(self, embedded_data):
        norms = np.linalg.norm(embedded_data, axis=1)
        normalized_data = embedded_data / norms[:, np.newaxis]
        return normalized_data, norms
    
    # normal_df_list = []

    def nomralize_data(self, stock_no):
        if isinstance(stock_no, (pd.DataFrame, pd.Series)):
            stock_no = stock_no.to_numpy()

        scaler = StandardScaler()
        stock_no_standardized = scaler.fit_transform(stock_no.reshape(-1, 1)).flatten()

        close_fft = np.fft.fft(stock_no_standardized)
        fft_df = pd.DataFrame({'fft': close_fft})
        fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
        fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))

        fft_list = np.asarray(fft_df['fft'].tolist())
        num_components = 100
        fft_list[num_components:-num_components] = 0

        reconstructed_signal = np.fft.ifft(fft_list)
        reconstructed_signal = reconstructed_signal.real

        embedded_data = self.embed_phase_space(reconstructed_signal)

        normalized_data, norms = self.normalize_delay_vectors(embedded_data)

        # normal_df = pd.concat([normalized_data, norms, scaler, fft_list], axis=1)
        # normal_df.columns = [f'normalized_{j+1}' for j in range(m)] + ['norms', 'scaler', 'fft']
        # normal_df_list.append(normal_df)

        return normalized_data, norms, scaler, fft_list
        
    def denormalize_data(self, normalized_data, norms, scaler, fft_list):
        denormalized_vectors = normalized_data * norms[:, np.newaxis]
        
        # reconstructed_signal = np.zeros_like(fft_list.real)
        # reconstructed_signal[:denormalized_vectors.shape[0]] = denormalized_vectors[:, 0]
        
        reconstructed_signal = np.fft.ifft(fft_list).real
        
        original_data = scaler.inverse_transform(reconstructed_signal.reshape(-1, 1)).flatten()
        
        return original_data
    

    # this is the part where the code breaks. 
    # def create_sliding_windows(self, data):
    #     windows = []
    #     for i in range(len(data) - self.window_size + 1):
    #         windows.append(data[i:i + self.window_size])
    #     return np.array(windows)


    # def create_data_vector(self, df):
    #     stock_data = []

    #     for i in range(self.nist):
    #         stock_series = df[f'stock_{i+1}'].values.flatten()
    #         sliding_windows = self.create_sliding_windows(stock_series)
    #         for window in sliding_windows:
    #             normalized_stock_data, norms, scaler, fft_list = self.nomralize_data(window)
    #             stock_data.append(normalized_stock_data)

    #     return np.array(stock_data)

    def create_data_vector(self, df):
        stock_data = []
        # normalized_datas = []

        for i in range(self.nist):
            normalized_stock_data, norms, scaler, fft_list = self.nomralize_data(df[f'stock_{i+1}'])
            normalized_stock_df = pd.DataFrame(normalized_stock_data, columns=[f'normalized_{j+1}' for j in range(self.m)])

            stock_data.append(normalized_stock_df)

        return np.array(stock_data)

    
    def adversarial_loss(self, y_true, y_pred):
        return tf.keras.losses.binary_crossentropy(y_true, y_pred)

    def forecast_error_loss(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))

    def direction_prediction_loss(self, y_true, y_pred):
        if y_true.shape[0] > 1:
            sign_true = tf.sign(y_true[1:] - y_true[:-1])
            sign_pred = tf.sign(y_pred[1:] - y_pred[:-1])
            return tf.reduce_mean(tf.abs(sign_true - sign_pred))
        else:
            return 0.0

    def custom_gan_loss(self, y_true, y_pred):
        adv_loss = self.adversarial_loss(y_true, y_pred)
        p_loss = self.forecast_error_loss(y_true, y_pred)
        dpl_loss = self.direction_prediction_loss(y_true, y_pred)
        return self.lambda_adv * adv_loss + self.lambda_p * p_loss + self.lambda_dpl * dpl_loss

    '''
        for our generator we are building an lstm model
        https://arxiv.org/pdf/1607.04381v1
    '''
    def build_generator(self, input_shape):
        model = Sequential()
        model.add(LSTM(input_shape[0], input_shape=input_shape, return_sequences=True, kernel_regularizer=l1(0.01)))
        model.add(Dropout(0.2))
        model.add(Dense(input_shape[1], activation='linear'))

        # print(model.summary())
        return model
    
    '''
        Sequential(
        (0): Conv1D(None -> 32, kernel_size=(5,), stride=(2,))
        (1): LeakyReLU(0.01)
        (2): Conv1D(None -> 64, kernel_size=(5,), stride=(2,))
        (3): LeakyReLU(0.01)
        (4): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)
        (5): Conv1D(None -> 128, kernel_size=(5,), stride=(2,))
        (6): LeakyReLU(0.01)
        (7): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)
        (8): Dense(None -> 220, linear)
        (9): BatchNorm(axis=1, eps=1e-05, momentum=0.9, fix_gamma=False, use_global_stats=False, in_channels=None)
        (10): LeakyReLU(0.01)
        (11): Dense(None -> 220, linear)
        (12): Activation(relu)
        (13): Dense(None -> 1, linear)
        )
    '''
    def build_discriminator(self, input_shape):
        model = Sequential()
        model.add(Conv1D(32, kernel_size=5, strides=2, input_shape=input_shape, padding='same'))
        model.add(LeakyReLU(0.01))
        model.add(Conv1D(64, kernel_size=5, strides=2, padding='same'))
        model.add(LeakyReLU(0.01))
        model.add(BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9))
        model.add(Conv1D(128, kernel_size=5, strides=2, padding='same'))
        model.add(LeakyReLU(0.01))
        model.add(BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9))

        model.add(Flatten())
        model.add(Dense(220, activation='linear'))
        model.add(BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.9))
        model.add(LeakyReLU(0.01))
        model.add(Dense(220, activation='relu'))

        model.add(Dense(input_shape[1], activation='linear'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # print(model.summary())
        return model

    def train_gan(self, generator, discriminator, data):
        optimizer_g = Adam(learning_rate=self.learning_rate_g)
        optimizer_d = Adam(learning_rate=self.learning_rate_d)
        best_loss = float('inf')
        epochs_no_improve = 0


        @tf.function
        def train_step(real_data):
            current_batch_size = tf.shape(real_data)[0]

            noise = tf.random.normal([current_batch_size, data.shape[1], data.shape[2]])

            with tf.GradientTape() as tape_g, tf.GradientTape() as tape_d:
                # Generate fake data
                generated_data = generator(noise, training=True)

                # Discriminator predictions
                real_output = discriminator(real_data, training=True)
                fake_output = discriminator(generated_data, training=True)

                # Calculate losses
                g_loss = self.custom_gan_loss(tf.ones_like(fake_output), fake_output)

                d_loss_real = tf.keras.losses.binary_crossentropy(tf.ones_like(real_output), real_output)
                d_loss_fake = tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_output), fake_output)
                d_loss = d_loss_real + d_loss_fake

                g_loss = tf.reduce_mean(g_loss)
                d_loss = tf.reduce_mean(d_loss)

            # Compute gradients
            gradients_of_generator = tape_g.gradient(g_loss, generator.trainable_variables)
            gradients_of_discriminator = tape_d.gradient(d_loss, discriminator.trainable_variables)

            # Apply gradients
            optimizer_g.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            optimizer_d.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

            return g_loss, d_loss



        for epoch in range(self.epochs):
            for batch in range(0, len(data), self.batch_size):
                end_index = batch + self.batch_size
                if end_index > len(data):
                    end_index = len(data)
                real_data_batch = data[batch:end_index]
                g_loss, d_loss = train_step(real_data_batch)

            print(f'Epoch {epoch + 1}, Generator Loss: {g_loss}, Discriminator Loss: {d_loss}')

            total_loss = g_loss + d_loss
            if best_loss - total_loss > self.min_delta:
                best_loss = total_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1


            if epochs_no_improve >= self.patience:
                print(f'Early stopping has occured at epoch {epoch}')
                break

        # generator.save('stock_generator.keras')
        # discriminator.save('stock_discriminator.keras')


        return generator

    def model_evaluate(self, generator, test_size):

        results = {}

        for x in range(self.nist):
            results[f'stock_{x+1}'] = {
                'Day': [],
                'Predicted Price': [],
                'Actual Price': [],
                'Absolute Difference': [],
                'Direction Match': []
            }

        total_rmse = 0

        for stock in range(self.nist):

            stock_key = f'stock_{stock + 1}'

            for i in range(test_size):
                # sliding window for prediction.
                start = i + 1
                end = i + self.window_size + 1

                test_data_batch = self.df.iloc[start:end, stock]

                # print(test_data_batch.shape)

                normalized_data, norms, scaler, fft_list = self.nomralize_data(test_data_batch.values.flatten())
                # normalized_stock_df = pd.DataFrame(normalized_stock_data, columns=[f'normalized_{j+1}' for j in range(m)])
                normalized_data = normalized_data.reshape(1, normalized_data.shape[0], normalized_data.shape[1])

                # print(normalized_stock_data.shape)
                # normalized_stock_df = normalized_stock_df.values.reshape(1, window_size, 3)

                predicted = generator.predict(normalized_data)
                predicted_price = self.denormalize_data(predicted, norms[-1:], scaler, fft_list).flatten()[-1]


                actual_price = test_data_batch.values.flatten()[-1]

                # time_index = np.arange(len(predicted_price))

                # plt.figure(figsize=(10, 5))
                # plt.plot(time_index, actual_prices, label='Actual Prices', color='blue')
                # plt.plot(time_index, predicted_price, label='Predicted Prices', color='red')
                # plt.title('Comparison of Actual and Predicted Prices')
                # plt.xlabel('Time Index')
                # plt.ylabel('Price')
                # plt.legend()
                # plt.grid(True)
                # plt.show()



                print(f'{start} to {end} - actual price: {actual_price} vs predicted price: {predicted_price}')
                difference = np.abs(predicted_price - actual_price)
                direction_match = False

                prev_price =  self.df.iloc[start - 1, stock]
                actual_change = (actual_price - prev_price) / prev_price
                predicted_change = (predicted_price - prev_price) / prev_price
                if (actual_change > 0 and predicted_change > 0) or (actual_change < 0 and predicted_change < 0):
                    direction_match = True



                results[stock_key]['Day'].append(i + self.window_size)
                results[stock_key]['Predicted Price'].append(predicted_price)
                results[stock_key]['Actual Price'].append(actual_price)
                results[stock_key]['Absolute Difference'].append(difference)
                results[stock_key]['Direction Match'].append(direction_match)

            total_rmse += sklearn.metrics.mean_squared_error(results[stock_key]['Actual Price'], results[stock_key]['Predicted Price'])  
        

        results_df = pd.DataFrame(results)
        results_df.to_csv('test_results.csv', index=False)
        print("Test results saved to 'test_results.csv'.")

        return total_rmse / self.nist


    def run(self, prcsSoFar):
        self.load(prcsSoFar)
        self.train_size = len(prcsSoFar[0]) - 1
        self.test_size = self.df.shape[0] - self.train_size
        train_df = self.df.iloc[:self.train_size]
        train_data = self.create_data_vector(train_df)
        input_shape = (train_data.shape[1], train_data.shape[2])
        generator = self.build_generator(input_shape)
        discriminator = self.build_discriminator(input_shape)

        return self.train_gan(generator, discriminator, train_data)

    def evaluate(self):
        generator = load_model('stock_generator.keras')
        
        return self.model_evaluate(generator, self.test_size)
    

    def load_model_gan(self):
        
        return load_model('stock_generator.keras')



# gan = Gan()
# gan.run()

