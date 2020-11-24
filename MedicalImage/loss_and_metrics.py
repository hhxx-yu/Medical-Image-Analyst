#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 10:59:36 2020

@author: ryan
"""
import tensorflow.keras.backend as K
import numpy as np
def create_weighted_binary_crossentropy(zero_weight, one_weight):

    def weighted_binary_crossentropy(y_true, y_pred):

        # Original binary crossentropy (see losses.py):
        # K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

        # Calculate the binary crossentropy
        b_ce = K.binary_crossentropy(y_true, y_pred)

        # Apply the weights
        weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
        weighted_b_ce = weight_vector * b_ce

        # Return the mean error
        return K.sum(weighted_b_ce)

    return weighted_binary_crossentropy

def dice_coef(y_true, y_pred):
        y_true_f=K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + 100) / (K.sum(y_true_f) + K.sum(y_pred_f) + 100)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def dice_coef_numpy(y_true, y_pred):
        y_true_f=np.ndarray.flatten(y_true)
        y_pred_f = np.ndarray.flatten(y_pred)
        intersection = np.sum(y_true_f * y_pred_f)
        return (2. * intersection + 100) / (np.sum(y_true_f) + np.sum(y_pred_f) + 100)