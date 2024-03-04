import tensorflow as tf
import numpy as np 
import cv2 
import pandas as pd 
import json

class prediction_pipeline():
    FRAME_LEN = 128
    LPOSE = [13, 15, 17, 19, 21]
    RPOSE = [14, 16, 18, 20, 22]
    POSE = LPOSE + RPOSE
    X = [f'x_right_hand_{i}' for i in range(21)] + [f'x_left_hand_{i}' for i in range(21)] + [f'x_pose_{i}' for i in POSE]
    Y = [f'y_right_hand_{i}' for i in range(21)] + [f'y_left_hand_{i}' for i in range(21)] + [f'y_pose_{i}' for i in POSE]
    Z = [f'z_right_hand_{i}' for i in range(21)] + [f'z_left_hand_{i}' for i in range(21)] + [f'z_pose_{i}' for i in POSE]
    FEATURE_COLUMNS = X + Y + Z
    
    X_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if "x_" in col]
    Y_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if "y_" in col]
    Z_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if "z_" in col]

    RHAND_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if "right" in col]
    LHAND_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if  "left" in col]
    RPOSE_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if  "pose" in col and int(col[-2:]) in RPOSE]
    LPOSE_IDX = [i for i, col in enumerate(FEATURE_COLUMNS)  if  "pose" in col and int(col[-2:]) in LPOSE]
    pad_token = 'P'
    start_token = '<'
    end_token = '>'
    pad_token_idx = 59
    start_token_idx = 60
    end_token_idx = 61
    with open ("character_to_prediction_index.json", "r") as f:
        char_to_num = json.load(f)

    # Add pad_token, start pointer and end pointer to the dict
    pad_token = 'P'
    start_token = '<'
    end_token = '>'
    pad_token_idx = 59
    start_token_idx = 60
    end_token_idx = 61

    char_to_num[pad_token] = pad_token_idx
    char_to_num[start_token] = start_token_idx
    char_to_num[end_token] = end_token_idx
    num_to_char = {j:i for i,j in char_to_num.items()}
    
    def __init__(self , filename= None):
        self.file = filename
        
    def get_record_bytes(self , dataframe):
        parquet_numpy = dataframe.to_numpy()
        features={}
        #if 2*len(phrase)<no_nan:
        features = {FEATURE_COLUMNS[i]: tf.train.Feature(
        float_list=tf.train.FloatList(value=parquet_numpy[:, i])) for i in range(len(FEATURE_COLUMNS))}
        features["phrase"] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes('phrase', 'utf-8')]))
        record_bytes = tf.train.Example(features=tf.train.Features(feature=features)).SerializeToString()
        return record_bytes
    
    def decode_fn(self , record_bytes):
        schema = {COL: tf.io.VarLenFeature(dtype=tf.float32) for COL in FEATURE_COLUMNS}
        schema["phrase"] = tf.io.FixedLenFeature([], dtype=tf.string)
        features = tf.io.parse_single_example(record_bytes, schema)
        phrase = features["phrase"]
        landmarks = ([tf.sparse.to_dense(features[COL]) for COL in FEATURE_COLUMNS])
        # Transpose to maintain the original shape of landmarks data.
        landmarks = tf.transpose(landmarks)
        #print(landmarks)
        return landmarks , phrase
    

        # Detect the dominant hand from the number of NaN values.
        # Dominant hand will have less NaN values since it is in frame moving.
        #This function preprocesses a tensor x representing hand and pose coordinates. 
        #It includes steps such as handling NaN values, normalizing the data, and resizing/padding the sequence.
    def pre_process(self,x):
        
        
        def resize_pad(x):
            if tf.shape(x)[0] < FRAME_LEN:
                x = tf.pad(x, ([[0, FRAME_LEN-tf.shape(x)[0]], [0, 0], [0, 0]]))
            else:
                x = tf.image.resize(x, (FRAME_LEN, tf.shape(x)[1]))
            return x
            
        rhand = tf.gather(x, RHAND_IDX, axis=1)
        lhand = tf.gather(x, LHAND_IDX, axis=1)
        rpose = tf.gather(x, RPOSE_IDX, axis=1)
        lpose = tf.gather(x, LPOSE_IDX, axis=1)

        rnan_idx = tf.reduce_any(tf.math.is_nan(rhand), axis=1)
        lnan_idx = tf.reduce_any(tf.math.is_nan(lhand), axis=1)

        rnans = tf.math.count_nonzero(rnan_idx)
        lnans = tf.math.count_nonzero(lnan_idx)

        # For dominant hand
        if rnans > lnans:
            hand = lhand
            pose = lpose

            hand_x = hand[:, 0*(len(LHAND_IDX)//3) : 1*(len(LHAND_IDX)//3)]
            hand_y = hand[:, 1*(len(LHAND_IDX)//3) : 2*(len(LHAND_IDX)//3)]
            hand_z = hand[:, 2*(len(LHAND_IDX)//3) : 3*(len(LHAND_IDX)//3)]
            hand = tf.concat([1-hand_x, hand_y, hand_z], axis=1) # REVIEW NEEDED 

            pose_x = pose[:, 0*(len(LPOSE_IDX)//3) : 1*(len(LPOSE_IDX)//3)]
            pose_y = pose[:, 1*(len(LPOSE_IDX)//3) : 2*(len(LPOSE_IDX)//3)]
            pose_z = pose[:, 2*(len(LPOSE_IDX)//3) : 3*(len(LPOSE_IDX)//3)]
            pose = tf.concat([1-pose_x, pose_y, pose_z], axis=1) # REVIEW NEEDED 
        else:
            hand = rhand
            pose = rpose

        hand_x = hand[:, 0*(len(LHAND_IDX)//3) : 1*(len(LHAND_IDX)//3)]
        hand_y = hand[:, 1*(len(LHAND_IDX)//3) : 2*(len(LHAND_IDX)//3)]
        hand_z = hand[:, 2*(len(LHAND_IDX)//3) : 3*(len(LHAND_IDX)//3)]
        hand = tf.concat([hand_x[..., tf.newaxis], hand_y[..., tf.newaxis], hand_z[..., tf.newaxis]], axis=-1)

        mean = tf.math.reduce_mean(hand, axis=1)[:, tf.newaxis, :]
        std = tf.math.reduce_std(hand, axis=1)[:, tf.newaxis, :]
        hand = (hand - mean) / std

        pose_x = pose[:, 0*(len(LPOSE_IDX)//3) : 1*(len(LPOSE_IDX)//3)]
        pose_y = pose[:, 1*(len(LPOSE_IDX)//3) : 2*(len(LPOSE_IDX)//3)]
        pose_z = pose[:, 2*(len(LPOSE_IDX)//3) : 3*(len(LPOSE_IDX)//3)]
        pose = tf.concat([pose_x[..., tf.newaxis], pose_y[..., tf.newaxis], pose_z[..., tf.newaxis]], axis=-1)

        x = tf.concat([hand, pose], axis=1)
        x = resize_pad(x)

        x = tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
        x = tf.reshape(x, (FRAME_LEN, len(LHAND_IDX) + len(LPOSE_IDX)))
        
        return x


    def convert_fn(self,landmarks,phrase):
        
        table = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=list(char_to_num.keys()),
                values=list(char_to_num.values()),
            ),
            default_value=tf.constant(-1),
            name="class_weight"
        )

        #Add start and end pointers to phrase.
        phrase = start_token + phrase + end_token
        phrase = tf.strings.bytes_split(phrase)
        phrase = table.lookup(phrase)
        #Vectorize and add padding.
        phrase = tf.pad(phrase, paddings=[[0, 64 - tf.shape(phrase)[0]]], mode = 'CONSTANT',
                        constant_values = pad_token_idx)
        #Apply pre_process function to the landmarks.
        #Output: the preprocessed landmarks and the preprocessed padded phrase
        return self.pre_process(landmarks) , phrase
    
    def prediction(self , batch):
        
        interpreter = tf.lite.Interpreter("models/model.tflite")

        REQUIRED_SIGNATURE = "serving_default"
        REQUIRED_OUTPUT = "outputs"

        with open ("character_to_prediction_index.json", "r") as f:
            character_map = json.load(f)
        rev_character_map = {j:i for i,j in character_map.items()}
        prediction_fn = interpreter.get_signature_runner("serving_default")
        output = prediction_fn(inputs=batch[0])
        prediction_str = "".join([rev_character_map.get(s, "") for s in np.argmax(output[REQUIRED_OUTPUT], axis=1)])
        
        return prediction_str
        

    def call(self , dataframe):
        record_bytes = self.get_record_bytes(dataframe)
        landmarks , phrases = self.decode_fn(record_bytes)
        preprocessed_data = self.convert_fn(landmarks,phrases)
        prediction = self.prediction(preprocessed_data)
        
        return prediction