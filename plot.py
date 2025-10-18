import matplotlib.pyplot as plt
import numpy as np
import json

with open('./results.json') as file:
    results = json.loads(file.read())

fig, ax = plt.subplots()
ax.boxplot(results.values(), labels=results.keys())

ax.set_title('Steps to 10 successes')
ax.set_xlabel('Agent')
ax.set_ylabel('Steps')

plt.show()