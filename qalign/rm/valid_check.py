import json
from qflow.utils.math import get_last_math

file_path = "qflow/rm/data/cotllama-factory_math_llama-3.1-8b-instruct_128_1_train.json"

with open(file_path, "r") as f:
    data = json.load(f)


count = 0
for entry in data:
    right = entry["chosen"]
    wrong = entry["rejected"]

    right_answer = get_last_math(right)
    wrong_last_line = wrong.split("\n")[-1]

    count += int(right_answer in wrong_last_line)

    if right_answer in wrong_last_line:
        print(f"right: [{right_answer}], wrong: >>>\n{'*'*20}\n {wrong_last_line}")
        # import pdb

        # pdb.set_trace()

print(count, 7500)
import pdb

pdb.set_trace()
