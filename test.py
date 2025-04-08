from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

from sklearn.datasets import fetch_openml


fashion_mnist = fetch_openml("Fashion-MNIST", version=1, as_frame=False)
x = fashion_mnist.data.astype('float32') / 255.0
y = fashion_mnist.target.astype('int')

# print(f'x: {x}')
# print(f'y: {y}')

print(f'len(x): {len(x)}')
print(f'len(x): {len(y)}')

from datetime import datetime
time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

print(f'time: {time}')