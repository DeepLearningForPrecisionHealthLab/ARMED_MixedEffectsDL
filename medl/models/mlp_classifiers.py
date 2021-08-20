"""Simple neural networks for classification
"""
import tensorflow as tf
import tensorflow.keras.layers as tkl
from .random_effects2 import RandomEffects

def base_model(inputs=2, outputs=2):
    tInput = tkl.Input(inputs)
    tDense1 = tkl.Dense(4, activation='relu')(tInput)
    tDense2 = tkl.Dense(4, activation='relu')(tDense1)
    tDense3 = tkl.Dense(4, activation='relu')(tDense2)
    tOutput = tkl.Dense(outputs, activation='softmax')(tDense3)
    
    model = tf.keras.Model(tInput, tOutput)
    model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  optimizer=tf.keras.optimizers.Adam())
    
    return model

def concat_model(clusters, inputs=2, outputs=2):
    tInput = tkl.Input(inputs)
    tInputZ = tkl.Input(clusters)
    
    tConcat = tkl.Concatenate(axis=-1)([tInput, tInputZ])
    
    tDense1 = tkl.Dense(4, activation='relu')(tConcat)
    tDense2 = tkl.Dense(4, activation='relu')(tDense1)
    tDense3 = tkl.Dense(4, activation='relu')(tDense2)
    tOutput = tkl.Dense(outputs, activation='softmax')(tDense3)
    
    model = tf.keras.Model((tInput, tInputZ), tOutput)
    model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  optimizer=tf.keras.optimizers.Adam())
    
    return model

def me_model(clusters, inputs=2, outputs=2, 
             post_loc_init_scale=0,
             post_scale_init_min=0.1,
             post_scale_init_range=0.1,
             kl_weight=0.001,
             prior_scale=1):
    
    tInput = tkl.Input(inputs)
    tInputZ = tkl.Input(clusters)
    
    tDense1 = tkl.Dense(4, activation='relu')(tInput)
    tDense2 = tkl.Dense(4, activation='relu')(tDense1)
    tDense3 = tkl.Dense(4, activation='relu')(tDense2)
    
    tRE = RandomEffects(2, post_loc_init_scale=post_loc_init_scale, 
                        post_scale_init_min=post_scale_init_min, 
                        post_scale_init_range=post_scale_init_range, 
                        kl_weight=kl_weight, 
                        prior_scale=prior_scale,
                        name='re_slope')(tInputZ)
    
    tConcat = tkl.Concatenate(axis=-1)([tDense3, tRE * tInput])
    tOutput = tkl.Dense(outputs, activation='softmax')(tConcat)
    
    # tOutputFE = tkl.Dense(2, activation='linear')(tDense3)
    # tOutputRE = tkl.Dot(axes=1)([tRE, tInput])
    # tOutput = tkl.Softmax()(tOutputFE + tOutputRE)
            
    model = tf.keras.Model((tInput, tInputZ), tOutput)
    model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  optimizer=tf.keras.optimizers.Adam())
    
    return model