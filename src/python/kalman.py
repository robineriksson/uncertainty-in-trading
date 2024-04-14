import numpy as np
from numba import float64
from numba.experimental import jitclass


spec = [
    ('A', float64[:, ::1]),   # State transition matrix
    ('H', float64[:, ::1]),   # Observation matrix
    ('Q', float64[:, ::1]),   # Process noise covariance
    ('R', float64[:, ::1]),   # Measurement noise covariance
    ('x', float64[::1]),      # Initial state estimate
    ('P', float64[:, ::1]),   # Initial covariance estimate
]

@jitclass(spec)
class KalmanFilter:
    def __init__(self, A, H, Q, R, x, P):
        self.A = A
        self.H = H
        self.Q = Q
        self.R = R
        self.x = x
        self.P = P

    def predict(self):
        # Predict the next state
        self.x = np.dot(self.A, self.x)
        self.P = np.dot(self.A, np.dot(self.P, self.A.T)) + self.Q

    def update(self, z):
        # Update and correct the state estimate
        y = z - np.dot(self.H, self.x)  # Innovation
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R  # Innovation covariance
        K = np.dot(self.P, np.linalg.pinv(np.dot(self.H.T, S)))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(K, np.dot(self.H, self.P))
        return y, S  # Return innovation and its covariance for likelihood calculation

    def log_likelihood(self, y, S):
        # Calculate the marginal likelihood of an observation
        det_S = np.linalg.det(S)
        # if det_S == 0:
        #     return 0  # To avoid division by zero
        return -0.5 * np.dot(y.T, np.dot(np.linalg.inv(S), y)) - \
                0.5 * np.log((2 * np.pi) ** len(y) * det_S)

# # Ensure arrays are contiguous and properly typed before instantiation
# A = np.ascontiguousarray([[1]], dtype=np.float64)
# B = np.ascontiguousarray([[0]], dtype=np.float64)
# H = np.ascontiguousarray([[1]], dtype=np.float64)
# Q = np.ascontiguousarray([[0.001]], dtype=np.float64)
# R = np.ascontiguousarray([[0.01]], dtype=np.float64)
# x = np.ascontiguousarray([100], dtype=np.float64)
# P = np.ascontiguousarray([[1]], dtype=np.float64)

# kf = KalmanFilter(A, B, H, Q, R, x, P)
# z = np.ascontiguousarray([102], dtype=np.float64)  # New observed price

# kf.predict()
# y, S = kf.update(z)
# print("Updated state estimate:", kf.x)
# print("Likelihood of the observation:", kf.likelihood(y, S))
