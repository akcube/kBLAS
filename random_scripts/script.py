import re
import pandas as pd
import matplotlib.pyplot as plt
import math
from tabulate import tabulate

with open("sscal") as file:
    lines = file.readlines()

def first_word(sstr):
    sstr.replace('\t', ' ')
    return sstr.split(' ', 1)[0]

mem = 0
flops = 0
bench = ""
stuff = {
    "BLIS ": [],
    "CBLAS": [],
    "KBLAS": [],
}
for line in lines:
    if line[0] == '-' or line == '\n':
        continue    
    if first_word(line) == 'Verified':
        continue
    if first_word(line) == 'Benchmark':
        mem = math.log2(float(re.findall("\d+\.\d+", line)[0]) * 1024 * 1024)
        bench = line[12:17]
    if line[0:6] == 'GFLOPS':
        flops = float(re.findall("\d+\.\d+", line)[0])

    if line[0:9] == 'Bandwidth':
        stuff[bench].append([mem, flops])

stuff["BLIS "] = pd.DataFrame(stuff["BLIS "], columns = ['Memory', 'Flops'])
stuff["CBLAS"] = pd.DataFrame(stuff["CBLAS"], columns = ['Memory', 'Flops'])
stuff["KBLAS"] = pd.DataFrame(stuff["KBLAS"], columns = ['Memory', 'Flops'])


# Draw Plot
plt.figure(figsize=(16,10), dpi= 80)
plt.plot('Memory', 'Flops', data=stuff["BLIS "], color='tab:blue')
plt.plot('Memory', 'Flops', data=stuff["CBLAS"], color='tab:green')
plt.plot('Memory', 'Flops', data=stuff["KBLAS"], color='tab:red')

# Decoration
plt.title("sscal 8c/16t benchmark", fontsize=22)
plt.grid(axis='both', alpha=.3)

# Remove borders
plt.gca().spines["top"].set_alpha(0.0)    
plt.gca().spines["bottom"].set_alpha(0.3)
plt.gca().spines["right"].set_alpha(0.0)    
plt.gca().spines["left"].set_alpha(0.3)   
plt.show()

print(tabulate(stuff["BLIS "], headers='keys', tablefmt='psql'))
print(tabulate(stuff["KBLAS"], headers='keys', tablefmt='psql'))
print(tabulate(stuff["CBLAS"], headers='keys', tablefmt='psql'))