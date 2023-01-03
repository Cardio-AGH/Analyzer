import io, base64
import matplotlib

# https://spapas.github.io/2021/02/08/django-matplotlib/
def matplotlib_to_base64(fig):
  flike = io.BytesIO()
  fig.savefig(flike)
  b64 = base64.b64encode(flike.getvalue()).decode()
  return b64
