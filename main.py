import pandas as pd
import tensorflow as tf


def get_dataframe():
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
    print(df.head())

    return frame


df = get_dataframe()
# target = df.pop('target')
# dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))
#
# for feat, targ in dataset.take(5):
#     print('Features: {}, Target: {}'.format(feat, targ))
#     tf.constant(df[''])
