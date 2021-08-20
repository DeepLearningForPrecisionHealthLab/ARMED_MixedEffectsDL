'''
Custom Keras loss functions and metrics
'''

import tensorflow as tf

def dice(yTrue, yPred):
    """Computes the hard Dice coefficient for segmentation accuracy

    Args:
        yTrue (tensor): ground truth segmentation masks
        yPred (tensor): predicted segmentations

    Returns:
        float: Dice coefficient
    """    
    yPredBinary = tf.keras.backend.cast_to_floatx(yPred >= 0.5)
    intersection = tf.reduce_sum(tf.multiply(yTrue, yPredBinary), axis=(1, 2, 3))
    total = tf.reduce_sum(yTrue, axis=(1, 2, 3)) + tf.reduce_sum(yPredBinary, axis=(1, 2, 3))
    dicecoef = tf.reduce_mean((2.0 * intersection) / (total + 1e-8))
    return dicecoef

def dice_sd(yTrue, yPred):
    """Computes the standard deviation of the sample-wise Dice coefficient

    Args:
        yTrue (tensor): ground truth segmentation masks
        yPred (tensor): predicted segmentations

    Returns:
        float: Dice coefficient s.d.
    """    
    yPredBinary = tf.keras.backend.cast_to_floatx(yPred >= 0.5)
    intersection = tf.reduce_sum(tf.multiply(yTrue, yPredBinary), axis=(1, 2, 3))
    total = tf.reduce_sum(yTrue, axis=(1, 2, 3)) + tf.reduce_sum(yPredBinary, axis=(1, 2, 3))
    sd = tf.math.reduce_std((2.0 * intersection) / (total + 1e-8))
    return sd


def dice_loss(yTrue, yPred):
    """Soft Dice loss (1 - Dice)

    Args:
        yTrue (tensor): ground truth segmentation masks
        yPred (tensor): predicted segmentations

    Returns:
        float: loss
    """    
    intersection = tf.reduce_sum(tf.multiply(yTrue, yPred), axis=(1, 2, 3))
    total = tf.reduce_sum(yTrue, axis=(1, 2, 3)) + tf.reduce_sum(yPred, axis=(1, 2, 3))
    dicecoef = tf.reduce_mean((2.0 * intersection + 1.0) / (total + 1.0))
    return 1 - dicecoef


def dice_bce_loss(yTrue, yPred):
    """Sum of the soft Dice loss and binary crossentropy loss
    Computed as (1 - Dice + BCE)

    Args:
        yTrue (tensor): ground truth segmentation masks
        yPred (tensor): predicted segmentations

    Returns:
        float: loss
    """    
    return dice_loss(yTrue, yPred) + tf.keras.losses.binary_crossentropy(yTrue, yPred)


def multiclass_dice_loss(yTrue, yPred):
    dice = 0
    nClass = 2
    for iClass in range(nClass):
        intersection = tf.reduce_sum(yTrue[..., iClass] * yPred[..., iClass])
        total = tf.reduce_sum(yTrue[..., iClass]) + tf.reduce_sum(yPred[..., iClass])
        dice += (2.0 * intersection / (total + 1e-8))
        
    return 1 - dice / nClass # average dice over the classes


def multiclass_dice(yTrue, yPred):
    """Computes the hard Dice coefficient for segmentation accuracy

    Args:
        yTrue (tensor): ground truth segmentation masks
        yPred (tensor): predicted segmentations

    Returns:
        float: Dice coefficient
    """    
    yPredMask = tf.math.argmax(yPred, axis=-1)
    yPredMask = tf.keras.backend.cast_to_floatx(yPredMask)
    yTrueMask = tf.keras.backend.cast_to_floatx(yTrue[..., 1])
    intersection = tf.reduce_sum(tf.multiply(yTrueMask, yPredMask), axis=(1, 2))
    total = tf.reduce_sum(yTrueMask, axis=(1, 2)) + tf.reduce_sum(yPredMask, axis=(1, 2))
    dicecoef = tf.reduce_mean((2.0 * intersection) / (total + 1e-8))
    return dicecoef

def multiclass_dice_sd(yTrue, yPred):
    """Computes the standard deviation of the sample-wise Dice coefficient

    Args:
        yTrue (tensor): ground truth segmentation masks
        yPred (tensor): predicted segmentations

    Returns:
        float: Dice coefficient s.d.
    """    
    yPredMask = tf.math.argmax(yPred, axis=-1)
    yPredMask = tf.keras.backend.cast_to_floatx(yPredMask)
    yTrueMask = tf.keras.backend.cast_to_floatx(yTrue[..., 1])
    intersection = tf.reduce_sum(tf.multiply(yTrueMask, yPredMask), axis=(1, 2))
    total = tf.reduce_sum(yTrueMask, axis=(1, 2)) + tf.reduce_sum(yPredMask, axis=(1, 2))
    sd = tf.math.reduce_std((2.0 * intersection) / (total + 1e-8))
    return sd