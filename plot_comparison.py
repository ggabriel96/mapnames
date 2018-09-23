import matplotlib.pyplot as plt
import pandas as pd

for m in ['c', 'igs', 'lgm']:
    df = pd.read_csv(f'{m}.csv')
    plt.xlabel('Test case')
    plt.ylabel('Accuracy')
    plt.plot(df.loc[:, 'accuracy'], label=m)

plt.legend()
plt.show()
