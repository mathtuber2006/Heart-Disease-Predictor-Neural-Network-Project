import numpy as np
import pandas as pd 

model = np.load("heart diseases/model_v3_updated.npz",allow_pickle=True)

w1 = model['W1']
b1 = model['b1']

w2 = model['W2']
b2 = model['b2']

mean = model['mean']
std = model['std']

columns = model['columns'].tolist()

test_vector = []

age = int(input("Enter age: "))
sex = input("Enter sex(M/F): ")
chest_pain = input("Enter chest pain type (ASY/ATA/NAP/TA): ")
bs = int(input("Your fasting blood sugar level(mg/dl): "))
if bs>122:
    bs_ = 1
else:
    bs_ = 0
ex = input("Do you have exercis angina (Y/N): ")
oldpeak = float(input("Enter old peak (0.0-6.0): "))
stslope = input("Enter ST_Slope (Down/Flat/Up): ")

bp = int(input("Enter resting blood pressure: "))
hr = int(input("Enter maximum heart rate: "))

bphr_ratio = bp/hr 

test_data = [age,sex,chest_pain,bs_,ex,oldpeak,stslope,bphr_ratio]

my_data = pd.DataFrame(
    [test_data],
    columns=[
        'Age', 'Sex', 'ChestPainType', 'FastingBS',
        'ExerciseAngina', 'Oldpeak', 'ST_Slope', 'bphr_ratio'
    ]
)

my_data = pd.get_dummies(
    my_data,
    columns=['Sex', 'ChestPainType', 'ExerciseAngina', 'ST_Slope'],
    dtype=int
)
my_data = my_data.reindex(columns=columns, fill_value=0)

continuous_cols = ['Age', 'Oldpeak', 'bphr_ratio']
my_data[continuous_cols] = (
    my_data[continuous_cols] - mean
) / std


def predict(data):
    
    def Relu(z):
        return np.maximum(0,z)

    def sigmoid(z):
        return 1/(1+np.exp(-z))

    z1 = np.dot(data,w1)+b1
    a1 = Relu(z1)

    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)   

    print(f"Heart Disease Probability: {a2.item():.2f}")  

predict(my_data)