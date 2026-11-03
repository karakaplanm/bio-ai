import subprocess
from PIL import Image

im = Image.open('leaf.png')
w, h = im.size
s = min(w, h)
l, t = (w - s) // 2, (h - s) // 2

# Ortadan kare kirp, griye cevir, 128x128'e kucult
g = im.crop((l, t, l + s, t + s)).convert('L').resize((128, 128), Image.LANCZOS)
g.save('leaf_gray_128.png')

# Loupe ile goster
subprocess.Popen(['loupe', 'leaf_gray_128.png'])
