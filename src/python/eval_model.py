import polars as pl
import numpy as np
from numba import njit

from kalman import KalmanFilter


# Assuming KalmanFilter class is defined as before

# Example DataFrame
df = pl.DataFrame({
    "prices": [100, 101, 102, 103, 104, 105]
}).with_columns(returns=pl.col('prices').pct_change()).fill_null(0)

# Convert Polars DataFrame column to NumPy array
prices = df.get_column("prices").to_numpy()
returns = df.get_column("returns").to_numpy()

# Initialize Kalman Filter parameters
A = np.ascontiguousarray([[1]], dtype=np.float64)
H = np.ascontiguousarray([[1]], dtype=np.float64)
Q = np.ascontiguousarray([[0.1]], dtype=np.float64)
R = np.ascontiguousarray([[0.1]], dtype=np.float64)
x = np.ascontiguousarray([0], dtype=np.float64) # initial state estimate using the first price
P = np.ascontiguousarray([[1]], dtype=np.float64)

# Initialize Kalman Filter
kf = KalmanFilter(A, H, Q, R, x, P)

# Function to apply Kalman Filter
@njit
def apply_kalman_filter(kf, prices):
    predictions = np.empty_like(prices)
    estimates = np.empty_like(prices)
    log_likelihoods = np.empty_like(prices)
    for i in range(len(prices)):
        kf.predict()
        predictions[i] = kf.x[0]  # Store the predicted state
        y, S = kf.update(np.array([prices[i]], dtype=np.float64))
        log_likelihoods[i] = kf.log_likelihood(y,S)
        estimates[i] = kf.x[0]  # Store the estimated state
    return predictions, estimates, log_likelihoods

# Apply the Kalman Filter to the prices
pred, est, lik = apply_kalman_filter(kf, returns)

# Results
print("Estimated states:", est)

t = np.arange(0,len(df))
import matplotlib.pyplot as plt

plt.plot(t, returns, label='True')
plt.plot(t, est, label='estimate')
plt.plot(t, pred, label='prediction')
plt.legend()
plt.show()

from explore_data import load_data
df = load_data()

#print(lik)
t_vec = df['date'].to_numpy()
close_ret = df['close'].pct_change().fill_null(0).to_numpy()


kf = KalmanFilter(A, H, Q, R, x, P)
pred, est, lik = apply_kalman_filter(kf, close_ret)



burnin=0
plt.plot(t_vec, close_ret, label='True')
plt.plot(t_vec[burnin:], est[burnin:], label='estimate')
plt.plot(t_vec[burnin:], pred[burnin:], label='prediction')
plt.legend()
plt.show()

plt.plot(pred, close_ret, '.')
plt.show()

plt.plot(t_vec[burnin:], lik[burnin:])
plt.show()

print(f'sum loglike: {np.sum(lik)}')

q_vec = np.linspace(0.01, 1, 100)
lik_vec = []
for q in q_vec:
    Q = np.ascontiguousarray([[q]], dtype=np.float64)
    kf = KalmanFilter(A, H, Q, R, x, P)
    _, _, lik = apply_kalman_filter(kf, close_ret)
    lik_vec.append(np.sum(lik))

plt.plot(q_vec, lik_vec)
plt.show()



# plt.hist(np.log(close_price[1:] / close_price[:-1]), bins=150, label='log ret')
# plt.hist(close_price[1:] / close_price[:-1] - 1, bins=150, label='ret')
# plt.legend()
# plt.show()




# Time step
delta_t = 1.0  # Adjust as necessary for your application's time resolution

# State Transition Matrix (A)
A = np.ascontiguousarray([[1, delta_t],
              [0, 1]], dtype=np.float64)


# Observation Matrix (H)
# If both position and velocity are observed:
H = np.ascontiguousarray([[1,0],[0,0]], dtype=np.float64)

# Process Noise Covariance (Q)
Q = np.ascontiguousarray([[0.1, 0.1],   # Adjust these values based on the expected process noise
              [0.1, 0.1]], dtype=np.float64)

# Measurement Noise Covariance (R)
R = np.ascontiguousarray([[0.1, 0],     # Adjust these values based on the expected measurement noise
                          [0, 0.1]], dtype=np.float64)

x = np.ascontiguousarray([0,0], dtype=np.float64) # initial state estimate using the first price
P = np.ascontiguousarray([[1, 0.1],
                          [0.1, 1]], dtype=np.float64)

kf = KalmanFilter(A, H, Q, R, x, P)

pred, est, lik = apply_kalman_filter(kf, close_ret)

kf.predict(np.array([0],dtype=np.float64))


burnin=0
plt.plot(t_vec, close_ret, label='True')
plt.plot(t_vec[burnin:], est[burnin:], label='estimate')
plt.plot(t_vec[burnin:], pred[burnin:], label='prediction')
plt.legend()
plt.show()
