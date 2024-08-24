# For Any issues contact:
# yc2349@ac.ic.uk
# coding: UTF-8
#
# Author:   Yassine Charouif
# Contact:  https://github.com/gems-yc4923/PyDecomposer/issues
#
# Feel free to contact for any information.

import numpy as np
from vmdpy import VMD # type: ignore
from EntropyHub import DispEn # type: ignore
from PyEMD import CEEMDAN # type: ignore
import matplotlib.pyplot as plt

class DecompositionModel:
    def __init__(self, alpha=2000, tau=0, K=7, DC=0, init=1, tol=1e-7, ceemdan_trials=100, ceemdan_epsilon=0.005):
        """
        Initializes the DecompositionModel class with given parameters for VMD and CEEMDAN decomposition.
        :param alpha: Bandwidth constraint for VMD, defaults to 2000
        :type alpha: int, optional
        :param tau: Noise tolerance for VMD, defaults to 0
        :type tau: int, optional
        :param K: Initial number of modes for VMD, defaults to 7
        :type K: int, optional
        :param DC: Imposition of DC component in VMD, 0 for no DC, defaults to 0
        :type DC: int, optional
        :param init: Initialization method for VMD, 1 for uniform, defaults to 1
        :type init: int, optional
        :param tol: Convergence tolerance for VMD, defaults to 1e-7
        :type tol: float, optional
        :param ceemdan_trials: Number of trials for CEEMDAN decomposition, defaults to 100
        :type ceemdan_trials: int, optional
        :param ceemdan_epsilon: Noise level for CEEMDAN decomposition, defaults to 0.005
        :type ceemdan_epsilon: float, optional
        """
        self.alpha = alpha
        self.tau = tau
        self.K_initial = K
        self.DC = DC
        self.init = init
        self.tol = tol
        self.ceemdan_trials = ceemdan_trials
        self.ceemdan_epsilon = ceemdan_epsilon

    def vmd_decomposition(self, signal):
        """
        Decomposes the input signal using Variational Mode Decomposition (VMD) until at least two low-frequency IMFs
        with entropy less than or equal to the threshold are obtained.

        :param signal: The input signal to be decomposed.
        :type signal: np.ndarray
        :return: Decomposed IMFs from VMD and the final number of IMFs.
        :rtype: tuple (np.ndarray, int)
        """
        flag = 0
        print("Decomposing signal using VMD...")
        while flag < 2 and self.K < 11:
            u_train, _, _ = VMD(signal, self.alpha, self.tau, self.K, self.DC, self.init, self.tol)
            disen_values = [DispEn(imf, m=20, c=10, tau=1) for imf in u_train]
            entropy_values = [entropy for entropy, _ in disen_values]
            threshold = 5.479
            low_freq_imfs = [i for i, entropy in enumerate(entropy_values) if entropy <= threshold]
            flag = len(low_freq_imfs)
            print(f"Number of low frequency IMFs: {flag} with: {self.K} IMFs")
            if flag < 2:
                self.K += 1
        print(f'Number of IMFs from VMD: {self.K}')
        return u_train, self.K-1

    def classify_imfs(self, imfs, threshold=5.479):
        """
        Classifies the Intrinsic Mode Functions (IMFs) into high-frequency and low-frequency components based on a given entropy threshold.

        :param imfs: The decomposed IMFs from VMD or CEEMDAN.
        :type imfs: np.ndarray
        :param threshold: The entropy threshold to classify IMFs into high and low frequency, defaults to 5.479.
        :type threshold: float, optional
        :return: High-frequency IMFs indices, low-frequency IMFs indices, and entropy values.
        :rtype: tuple (list, list, list)
        """
        disen_values = [DispEn(imf, m=20, c=10, tau=1) for imf in imfs]
        entropy_values = [entropy for entropy, _ in disen_values]
        self.high_freq_imfs = [i for i, entropy in enumerate(entropy_values) if entropy > threshold]
        self.low_freq_imfs = [i for i, entropy in enumerate(entropy_values) if entropy <= threshold]
        return self.high_freq_imfs, self.low_freq_imfs, entropy_values

    def ceemdan_decomposition(self, signal):
        """
        Decomposes the input signal using Complete Ensemble Empirical Mode Decomposition with Adaptive Noise (CEEMDAN).

        :param signal: The input signal to be decomposed.
        :type signal: np.ndarray
        :return: Decomposed IMFs from CEEMDAN.
        :rtype: np.ndarray
        """
        ceemdan = CEEMDAN(trials=self.ceemdan_trials, epsilon=self.ceemdan_epsilon, parallel=False)
        ceemdan.noise_seed(42)
        print("Decomposing signal using CEEMDAN...")
        imfs_iceemdan = ceemdan.ceemdan(signal)
        imfs_iceemdan = self.adjust_imfs(imfs_iceemdan)
        return imfs_iceemdan
    
    def adjust_imfs(self, imfs_iceemdan, num_imfs=9):
        """
        Adjusts the number of IMFs to ensure consistency. If there are more than the specified number of IMFs,
        combines the excess IMFs into one. If there are fewer, adds zero IMFs to match the desired number.

        :param imfs_iceemdan: The IMFs resulting from the CEEMDAN decomposition.
        :type imfs_iceemdan: np.ndarray
        :param num_imfs: The desired number of IMFs, defaults to 9.
        :type num_imfs: int, optional
        :return: Adjusted IMFs.
        :rtype: np.ndarray
        """
        num_existing_imfs = imfs_iceemdan.shape[0]
        if num_existing_imfs == num_imfs:
            print(f"Number of IMFs from CEEMDAN: {num_existing_imfs}")
            print("Therefore, this signal has more or less the same variability as our target signal")
        elif num_existing_imfs > num_imfs:
            print(f"Number of IMFs from CEEMDAN: {num_existing_imfs}")
            print("Therefore this signal has much more effect on the short term.")
            print(f"Combining the last {num_existing_imfs - num_imfs} IMFs.")
            extras = imfs_iceemdan[num_imfs-1:]
            combined_imf = np.sum(extras, axis=0)
            imfs_iceemdan = imfs_iceemdan[:num_imfs-1]
            imfs_iceemdan = np.vstack((imfs_iceemdan, combined_imf))
        elif len(imfs_iceemdan) < num_imfs:
            print(f"Number of IMFs from CEEMDAN: {num_existing_imfs}")
            print("This signal does not have much effects on the short term, and is more of a long term trend.")
            print(f"Adding {num_imfs - num_existing_imfs} additional IMFs.")
            additional_imfs = [np.zeros_like(imfs_iceemdan[0])] * (num_imfs - len(imfs_iceemdan))
            imfs_iceemdan = np.vstack([imfs_iceemdan, additional_imfs])
        return imfs_iceemdan

    def plot_imfs(self, imfs, title='IMFs'):
        """
        Plots the Intrinsic Mode Functions (IMFs) along with their sum.

        :param imfs: The IMFs to be plotted.
        :type imfs: np.ndarray
        :param title: The title for the plot, defaults to 'IMFs'.
        :type title: str, optional
        """
        plt.figure(figsize=(12, 8))
        plt.subplot(len(imfs) + 1, 1, 1)
        plt.plot(np.sum(imfs, axis=0))
        plt.title(title)
        for i, imf in enumerate(imfs):
            plt.subplot(len(imfs) + 1, 1, i + 2)
            plt.plot(imf)
            plt.title(f'IMF {i + 1}')
        plt.tight_layout()
        plt.show()

    def execute(self, signal):
        """
        Executes the full decomposition process on the input signal. This includes VMD decomposition,
        classification of IMFs, recomposition of high-frequency IMFs, and secondary decomposition using CEEMDAN.
        The final decomposed signals are stored as instance variables.

        :param signal: The input signal to be decomposed.
        :type signal: np.ndarray
        """
        self.K = self.K_initial
        vmd_imfs, _ = self.vmd_decomposition(signal)
        high_freq, low_freq, _ = self.classify_imfs(vmd_imfs)
        recomposed_signal = np.sum(vmd_imfs[high_freq,:], axis=0)
        imfs_iceemdan = self.ceemdan_decomposition(recomposed_signal)
        self.residual = vmd_imfs[low_freq[0]] + imfs_iceemdan[8]
        if len(low_freq) == 2:
            self.low_frequency = vmd_imfs[low_freq[1]] + np.sum(imfs_iceemdan[6:8], axis=0)
        else:
            print("This signal has few low frequency components.")
            self.low_frequency = np.sum(imfs_iceemdan[6:8], axis=0)
        self.medium_frequency = np.sum(imfs_iceemdan[3:6], axis=0)
        self.high_frequency = np.sum(imfs_iceemdan[0:3], axis=0)
        self.original_signal = signal
        for _ in range(1):
            print("...")
        print("Decomposition completed successfully.")
        print("Original signal, residual (long term trend), low, medium and high frequency signals are stored as instance variables.")
        print("Use function get_signals() to retrieve the decomposed signals.")
        print("Use function plot_signals() to visualize the decomposed signals.")
    
    def get_signals(self):
        """
        Retrieves the decomposed signals: high-frequency, medium-frequency, low-frequency, and residual.
        In that order.
        :return: high-frequency, medium-frequency, low-frequency, and residual.
        :rtype: tuple (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)
        """
        return self.high_frequency, self.medium_frequency, self.low_frequency, self.residual
    
    def plot_signals(self):
        """
        Plots the original signal and the decomposed signals: high-frequency, medium-frequency, low-frequency, and residual.

        The plots are enhanced for better readability with distinct colors, line widths, grid lines, and titles.

        The signals plotted are:
        - Original Signal: Plotted in green.
        - Residual/Long-Term Price Evolution: Plotted in blue.
        - Low-Frequency Monthly Changes of Price: Plotted in purple.
        - Medium-Frequency Weekly Changes of Price: Plotted in purple.
        - High-Frequency Daily Changes of Price: Plotted in purple.
        """
        plt.figure(figsize=(14, 12))

        plt.subplot(5, 1, 1)
        plt.plot(self.original_signal, color='green', linewidth=1.5)
        plt.title('Original Signal For Daily Price over Time', fontsize=14)
        plt.grid(True)

        plt.subplot(5, 1, 2)
        plt.plot(self.residual, color='blue', linewidth=2)
        plt.title('Residual/Long-Term Price Evolution', fontsize=14)
        plt.grid(True)

        plt.subplot(5, 1, 3)
        plt.plot(self.low_frequency, color='purple', linewidth=2)
        plt.title('Low-Frequency Monthly Changes of Price', fontsize=14)
        plt.grid(True)

        plt.subplot(5, 1, 4)
        plt.plot(self.medium_frequency, color='purple', linewidth=2)
        plt.title('Medium-Frequency Weekly Changes of Price', fontsize=14)
        plt.grid(True)

        plt.subplot(5, 1, 5)
        plt.plot(self.high_frequency, color='purple', linewidth=2)
        plt.title('High-Frequency Daily Changes of Price', fontsize=14)
        plt.grid(True)

        plt.tight_layout(pad=2.0)
        plt.show()

    def run(self, signal):
        """
        Executes the full decomposition process on the input signal and plots the decomposed signals.

        :param signal: The input signal to be decomposed.
        :type signal: np.ndarray
        """
        self.execute(signal)
        print("Decomposed signals are stored as instance variables. Please use get_signals() to retrieve the decomposed signals.")
        self.plot_signals()