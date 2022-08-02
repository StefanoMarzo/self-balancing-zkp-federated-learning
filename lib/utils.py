import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import random
import time

# PROPERTIES #################################################################

properties = {
    # Ethnicity labels
    'ETHNICITIES' : { 
        0: 'White', 
        1: 'Black',
        2: 'Asian',
        3: 'Indian',
        4: 'Other'
    },
    
    # Gender labels
    'GENDERS' : { 
        0: 'Male', 
        1: 'Female'
    },
    
    'features' : {
        'ETHNICITY': ['White', 'Black', 'Asian', 'Indian'],
        'GENDER': ['Male', 'Female']
    },
    
    'REVERSE_GENDERS' : {
        'Male' : 0,
        'Female' : 1
    },
    
    #Color channel
    'color_channel' : 255,
    
    #Image size
    'img_size' : 48,
    
    #Age range
    'age_min' : 0,
    'age_max' : 39,
    'age_range' : lambda: properties['age_max'] - properties['age_min'],
    'age_bins' : 10,
    'bins' : lambda: create_bins(properties['age_min'], 
                                 properties['age_bins'], 
                                 int(properties['age_range']() / properties['age_bins'])),
    
    #Training model
    'learning_rate' : 0.025,
    'epochs_n' : 1,
    
    #Train features
    #Use
    'feature_to_use' : 'pixels',
    #To predict
    'feature_to_predict' : 'age',

    #Number of data samples
    'data_samples' : 100000,
    'privileged_group_proportion': 0.85,

    #Randomness
    'seed' : 19101995,    
    
    #ZKP
    'enable_zk': False,
    
    #Balanced training
    'gap': 0.05,
    'max_gap': 1,    
    
    'test_size': 0.2,
    
}

# DATA PREPARATION ###########################################################

def create_train_test(df, test_size=.2):  
    return train_test_split(df, test_size=test_size, random_state=properties['seed'])

def prepare_X(df):
    df = df.copy()
    df[properties['feature_to_use']] = df[properties['feature_to_use']]\
        .apply(lambda x: np.array(x.split(), dtype='float32'))
    X = np.array(df[properties['feature_to_use']].tolist())
    X = np.reshape(X, (-1, 1, 1, properties['img_size'] * properties['img_size'])) / properties['color_channel']
    return X

def prepare_y(df):
    y = np.array(df[properties['feature_to_predict']].apply(lambda x: one_hot(x, properties['bins']())).tolist())
    y = np.reshape(y, (-1, 1, 1, len(properties['bins']())))
    return y
    
def prepare_data(df_train, df_test):
    X_train = prepare_X(df_train)
    y_train = prepare_y(df_train)
    X_test = prepare_X(df_test)
    y_test = prepare_y(df_test)
    return X_train, y_train, X_test, y_test

def create_ethnicity_X_y(dataset, ethnicity):
    df = dataset[dataset['ethnicity'] == ethnicity]
    ethn_X = prepare_X(df)
    ethn_y = prepare_y(df)
    return ethn_X, ethn_y

def create_all_ethnicities_X_y(dataset):
    ae = {}
    for e in properties['ETHNICITIES'].values():
        ae[e] = list(create_ethnicity_X_y(dataset, e))
    return ae

# GRAPHICS ###################################################################

def render_row(row, prediction=None):
    pixels = bytearray([int(px) for px in row['pixels'].split(' ')])
    img = Image.frombytes('L', (properties['img_size'], properties['img_size']), bytes(pixels))
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(2, 2), dpi=100)
    axes.axis('off')
    imgplot = plt.imshow(img, cmap='gray')
    x_offset_text = 52
    plt.text(x_offset_text, 5, 'Age: ' + str(row['age']))
    plt.text(x_offset_text, 12, 'Ethnicity: ' + row['ethnicity'])
    plt.text(x_offset_text, 19, 'Gender: ' + row['gender'])
    if prediction is not None:
        plt.text(x_offset_text, 26, 'Predicted age: ' + str(prediction))
        corr = int(prediction[0]) <= int(row['age']) <= int(prediction[1])
        evaluate = 'correct' if corr else 'not correct'
        color = 'green' if corr else 'red'
        plt.text(x_offset_text, 33, 'Prediction ' + evaluate, color=color)
    plt.show()


# FUNCTIONS ##################################################################
    
def age_to_out(age, age_range):
    return age / age_range

def create_bins(lower_bound, width, quantity):
    bins = []
    for low in range(lower_bound, 
                     lower_bound + quantity * width + 1 , width):
        bins.append((low, low + width - 1))
    return bins

def one_hot(number, bins):
    oh = [0] * len(bins)
    for i in range(len(bins)):
        if bins[i][0] <= number <= bins[i][1]:
            oh[i] = 1
            return oh
    raise ValueError('invalid number passed to one-hot function, check bins for details')
    
def one_hot_to_num(oh):
    M = max(oh)
    return oh.index(M)

def one_hot_vect_to_class(ohv):
    return [one_hot_to_num(n) for n in ohv]

def convert_output(y_out):
    out = []
    for i in range(len(y_out)):
        label = one_hot_to_num(list(y_out[i][0][0]))
        out += [one_hot(label * properties['age_bins'], properties['bins']())]
    return out
    
def print_summary(dataset, name='dataset'):
    n_bins = len(properties['bins']())
    fig, axs = plt.subplots(1, 3, sharey=True, tight_layout=True, figsize=(12, 3))
    fig.suptitle(name + f', #records: {len(dataset)}')
    axs[0].title.set_text('Age')
    axs[0].hist(dataset['age'], bins=n_bins)
    axs[1].title.set_text('Ethnicity')
    axs[1].hist(dataset['ethnicity'], bins=5*2-1)
    axs[2].title.set_text('Gender')
    axs[2].hist(dataset['gender'], bins=2*2-1)
    plt.show()
    
def predict_row(df, row_n, model):
    row = df.loc[row_n:row_n]
    row_x = prepare_X(row)
    pred = properties['bins']()[convert_output(model.predict(row_x))[0].index(1)]
    render_row(df.loc[row_n], pred)
    
def get_rnd_ethnicity():
    return random.choice(list(properties['ETHNICITIES'].values()))

def get_rnd_gender():
    return random.choice(list(properties['GENDERS'].values()))

def macro_variance(ref, vals):
    refs = [ref] * len(vals)
    sub = [r_i - v_i for r_i, v_i in zip(refs, vals)]
    return np.round(sum(list(map(lambda x : x**2 / (len(vals) - 1), sub))), 6)

def tm():
    return time.time()

def plot_accuracy_comparison(mb, mu):
    fig, axs = plt.subplots(1, 1, tight_layout=True, figsize=(4, 4))
    xu = np.arange(len(mu.accuracy))
    xb = np.arange(len(mb.accuracy))
    #fig.suptitle('')
    plt.ylim((0,1))
    fig.suptitle('Accuracy')
    axs.plot(xb, mb.accuracy, label='Self-balanced')
    axs.plot(xu, mu.accuracy, label='Imbalanced', color='orange')
    axs.legend(loc='lower right')
    plt.show()