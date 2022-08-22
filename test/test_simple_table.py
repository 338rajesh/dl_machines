import os
import numpy as np
from NeuralNetworks.utils import SimpleTable


CFD = os.path.dirname(__file__)

arr = [["ABC56987582451", "DEF", "", "KKK", ""]] + np.random.rand(6, 5).tolist()

st = SimpleTable(title="TESTING SIMPLE TABLE")

simple_table = st(arr, content_type="ROW")

print(simple_table)


