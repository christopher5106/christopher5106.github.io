---
layout: post
title:  "Downloading City of Chicago Open Source dataset of procurement contracts for DataScience"
date:   2020-06-23 05:00:00
categories: datascience
---

The City of Chicago offers a tremendous amount of contracts for procurement in open source (about 140k). But downloading them with a script might be a bit tricky, while it looks so easy manually in the browser.

Adobe has created a PDF viewing layer on top of the URL that converts the SID into a contract UUID and launches the view in an iframe.

For that purpose, we need to execute the obfuscated javascript in a browser:
```python
url = "http://ecm.cityofchicago.org/eSMARTContracts/service/DPSWebDocumentViewer?sid=EDGE&id=00010GRP"

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
chrome_options = Options()
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--headless')
driver = webdriver.Chrome(options=chrome_options)
driver.get(url)
pdf_url = driver.execute_script("return document.getElementsByTagName('iframe')[0].src")
print("Final URL with UUID:", pdf_url)
driver.quit()
```

It returns me the URL of the PDF: `http://ecm.chicago.gov/eSMARTContracts/service/DPSWebDocumentViewer?id={24F173CB-74D4-4859-9DA9-179076E85E1D}&osName=eContentContracts&el=0&image=image`.

In order to avoid downloading too big files, I first get the file size with a lightweight HTTP header request:

```python
import requests
import time
start = time.time()
response = requests.head(pdf_url, allow_redirects=True)
print(f"### HEAD request in {time.time()-start}s")
print("\n".join([('{:<40}: {}'.format(k, v)) for k, v in response.headers.items()]))
size = int(response.headers.get('content-length', 0)) / float(1 << 20) # number of bytes in a megabyte
print('{:<40}: {:.2f} MB'.format('FILE SIZE', size))
```

If below a file size limit I set in MB, I download it:

```python
MAX_FILE_SIZE=10
if "application/pdf" == response.headers['Content-Type'] and size < MAX_FILE_SIZE:
    start = time.time()
    response = requests.get(pdf_url)
    with open('file.pdf', 'wb') as f:
        f.write(response.content)
    assert response.status_code == 200
    print(f"### GET request in {time.time()-start}s")
    print("\n".join([('{:<40}: {}'.format(k, v)) for k, v in response.headers.items()]))
else:
    print("Too large file")
```

**Well done!**
