# Generate MoM with LLM
Modul ini dirancang untuk menyederhanakan dan mempercepat proses pembuatan ringkasan dan Risalah Rapat (MoM) menggunakan teknik pemrosesan bahasa alami tingkat lanjut, khususnya memanfaatkan pustaka Gensim. Baik Anda menangani dokumen berukuran besar atau transkrip rapat, perpustakaan ini memberikan solusi cepat dan efisien untuk menyaring informasi penting.

## How To Use
```bash
python SummerizeMoM/main.py  --file "<your_file>.rtf" 
```
atau jika ingin mengubah nama file hasil summary atau MoM dapat menggunakan perintah ini:
```bash
python SummerizeMoM/main.py  --file "<your_file>.rtf" --summary "<file_summary>.txt" --MoM "<file_MoM>.txt"
```

[help]
```bash
usage: main.py [-h] --file FILE [--summary [SUMMARY]] [--MoM [MOM]]

Create MoM with LLM Model

options:
  -h, --help           show this help message and exit
  --file FILE          file .RTF path to create the MoM file
  --summary [SUMMARY]  Name file for summary
  --MoM [MOM]          Name file for MoM
```

## Setup
### Memasang Virtual Environtment
```bash
virtualenv venv
source venv/bin/activate
```
### Install requirement
```bash
python setup.py
```
### Setup Gensim
Modul ini melakukan summery menggunakan gensim dengan versi **v3.8**. Dikarenakan library gensim hanya mensupport summary pada versi **< v4.0.0**. sehingga perlu melakukan configurasi mengenai library modul dapat kompatible dengan gensim.

ubah import collection Mapping menjadi collection.abc pada file `env/lib/python3.10/site-packages/gensim/corpora/dictionary.py`.
before
```python
from collections import Mapping, defaultdict
```
after
```python
from collections import defaultdict
from collections.abc import Mapping
```

kemudain ubah collection iterable menjadi collection.abc pada file `venv/lib/python3.10/site-packages/gensim/models/doc2vec.py`.
before
```python
from collections import namedtuple, defaultdict, Iterable
```
after
```python
from collections import namedtuple, defaultdict
from collections.abc import Iterable
```
kemudain ubah collection iterable menjadi collection.abc pada file `/venv/lib/python3.10/site-packages/gensim/models/fasttext.py`.
before
```python
from collections import Iterable
```
after
```python
from collections.abc import Iterable
```

## Config LLM
anda dapat mengganti model dan mengatur configurasi dari llm pada file `SummerizeMoM/config.yaml`. ini adalah configurasi llm saat ini.
```yaml
model: "TheBloke/vicuna-7B-v1.5-GPTQ"

config: 
  max_new_tokens: 1024
  temperature: 0.2
  top_p: 0.1
  repetition_penalty: 1.1
```




