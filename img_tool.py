import io, base64
from matplotlib import pyplot as plt

# https://spapas.github.io/2021/02/08/django-matplotlib/
def matplotlib_to_base64(fig):
  flike = io.BytesIO()
  fig.savefig(flike)
  b64 = base64.b64encode(flike.getvalue()).decode()
  return b64

def plot(x):
  fig = plt.figure(figsize=(10,3))
#   plt.axis(20,100)
  plt.plot(x)
  return fig

waves_colors = ['red', 'green', 'black', 'orange', 'pink']
def plot_waves(data_array, baseline, waves):
  fig, ax = plt.subplots(figsize=(10,5))
  ax.plot(baseline, color = 'blue', label = 'baseline')
  for key, values in waves.items():
      # for index, (key, values) in enumerate(waves.items()):
    # print(index)
    print(key)
    ax.scatter(values, data_array[1][values], label=key, color=waves_colors[0], s=50, marker='*')
  return fig
