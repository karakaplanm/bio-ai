from google.colab import files
uploaded = files.upload()  # Bilgisayarından yükleme penceresi açılacak

import zipfile, os
zip_name = [k for k in uploaded.keys() if k.endswith('.zip')][0]
with zipfile.ZipFile(zip_name, 'r') as z:
    z.extractall('/content')
!ls -lah /content/project_breast_cancer

%cd /content/project_breast_cancer
!pip install -q -r requirements.txt

!python src/eda.py

from IPython.display import Image, display
from pathlib import Path

for p in sorted(Path('reports/figures').glob('*.png')):
    display(Image(filename=str(p)))


!python src/train.py

import json
print(json.dumps(json.load(open('reports/metrics.json','r',encoding='utf-8')), indent=2, ensure_ascii=False))

!python src/predict.py --input "mean radius=14.0,mean texture=20.0,mean perimeter=90.0,mean area=600.0,mean smoothness=0.1"