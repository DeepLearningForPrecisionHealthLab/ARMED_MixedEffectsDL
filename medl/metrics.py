import numpy as np
import sklearn.metrics
import cv2

def compute_youden_point(y_true, y_pred):
    fpr, tpr, thresh = sklearn.metrics.roc_curve(y_true, y_pred)
    youden = tpr - fpr
    youdenPoint = thresh[np.argmax(youden)]
    youdenMax = youden.max()
    
    return youdenPoint, youdenMax


def classification_metrics(y_true, y_pred, youden_point: float=None):
    auroc = sklearn.metrics.roc_auc_score(y_true, y_pred)

    if youden_point is None:
        youden_point, youden_max = compute_youden_point(y_true, y_pred)
        yPredBinary = y_pred >= youden_point
        fpr = ((1 - y_true) * yPredBinary).sum() / (1 - y_true).sum()
        
    else: 
        yPredBinary = y_pred >= youden_point
        tpr = (y_true * yPredBinary).sum() / y_true.sum()
        fpr = ((1 - y_true) * yPredBinary).sum() / (1 - y_true).sum()
        youden_max = tpr - fpr

    acc = sklearn.metrics.balanced_accuracy_score(y_true, yPredBinary)
    f1 = sklearn.metrics.f1_score(y_true, yPredBinary)
    ppv = sklearn.metrics.precision_score(y_true, yPredBinary)
    npv = sklearn.metrics.precision_score(y_true, yPredBinary, pos_label=0)
    sensitivity = sklearn.metrics.recall_score(y_true, yPredBinary)

    return {'AUROC': auroc, 
            'Accuracy': acc, 
            'Youden\'s index': youden_max, 
            'F1': f1, 
            'PPV': ppv, 
            'NPV': npv,
            'Sensitivity': sensitivity,
            'Specificity': 1 - fpr}, youden_point

def single_sample_dice(y_true, y_pred):
    yPredBinary = y_pred >= 0.5
    intersection = np.sum(y_true * yPredBinary)
    total = np.sum(y_true) + np.sum(yPredBinary)
    return 2 * intersection / (total + 1e-8)

def balanced_accuracy(y_true, y_pred):
    import tensorflow as tf
    from tensorflow.keras.backend import floatx
    
    predbin = tf.cast((y_pred >= 0.5), floatx())
    correct = tf.cast(tf.equal(y_true, predbin), floatx())
    return tf.reduce_mean(tf.reduce_mean(correct, axis=0))
    
    # predpos = tf.cast((y_pred >= 0.5), floatx())
    # truepos = tf.reduce_sum(y_true * predpos, axis=0)
    # tot = tf.reduce_sum(y_true, axis=0)
    # recall = truepos / (tot + 1e-7)
    # return tf.reduce_mean(recall)
    
def image_metrics(img: np.array):
    ''' Compute image metrics, including brightness, 
    contrast, sharpness, and SNR'''
    
    brightness = img.mean()
    contrast = img.std()
    sharpness = cv2.Laplacian(img, cv2.CV_32F).var()
    snr = brightness/contrast
    return {'Brightness': brightness,
            'Contrast': contrast,
            'Sharpness': sharpness,
            'SNR': snr}
