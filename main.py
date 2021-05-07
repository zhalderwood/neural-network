import pandas as pd
import tensorflow as tf
import numpy as np
import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_df():
    # takes in .csv file and converts to discrete numerical values
    frame = pd.read_csv('aug_train.csv')
    columns = {
        'gender',
        'relevant_experience',
        'enrolled_university',
        'education_level',
        'major_discipline',
        'experience',
        'company_size',
        'company_type',
        'last_new_job'
    }

    for column in columns:
        frame[column] = pd.Categorical(frame[column])
    frame['gender'] = frame.gender.cat.codes
    frame['relevant_experience'] = frame.relevant_experience.cat.codes
    frame['enrolled_university'] = frame.enrolled_university.cat.codes
    frame['education_level'] = frame.education_level.cat.codes
    frame['major_discipline'] = frame.major_discipline.cat.codes
    frame['experience'] = frame.experience.cat.codes
    frame['company_size'] = frame.company_size.cat.codes
    frame['company_type'] = frame.company_type.cat.codes
    frame['last_new_job'] = frame.last_new_job.cat.codes

    pd.set_option('display.max_columns', None)

    return frame.sample(frac=1)


def get_df_one_hot():
    # takes in .csv file and converts to discrete numerical values
    frame = pd.read_csv('aug_train.csv')

    # clean up file columns that don't need one-hot encoding
    frame['experience'].replace({'>20': 20, '<1': 0, np.nan: '-1'}, inplace=True)
    frame['last_new_job'].replace({'>4': '5', 'never': '0', np.nan: '-1'}, inplace=True)
    frame['experience'] = frame['experience'].astype(int)
    frame['last_new_job'] = frame['last_new_job'].astype(int)
    frame['relevant_experience'] = pd.Categorical(frame['relevant_experience'])
    frame['relevant_experience'] = frame.relevant_experience.cat.codes

    # one-hot encode the rest
    enc_frame = pd.get_dummies(frame)
    pd.set_option('display.max_columns', None)
    print(enc_frame.head())

    # split dataset 85/15 and return
    return enc_frame  # .iloc[:16150, :], enc_frame.iloc[16151:, :]


def get_compiled_model():
    mod = tf.keras.Sequential([
        # add a normalization layer?
        tf.keras.layers.Dense(30, activation='relu'),
        tf.keras.layers.Dense(30, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    mod.compile(optimizer='adam',
                # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
    return mod


# split dataset 85/15
df = get_df_one_hot()
print(df.shape)
print(df.dtypes)

df_target = df.pop('target')

df_all = tf.data.Dataset.from_tensor_slices((df.values, df_target.values))


for feature, target in df_all.take(5):
    print('Features: {}, Target: {}'.format(feature, target))

model = get_compiled_model()
loops = 5
df_working = df_all.iloc[:16150, :]
df_test = df.iloc[16151:, :]
while loops > 0:
    df_all = df_all.batch(60)
    model.fit(df_all, epochs=300)
    model.summary()
    loops -= 1
model.evaluate()