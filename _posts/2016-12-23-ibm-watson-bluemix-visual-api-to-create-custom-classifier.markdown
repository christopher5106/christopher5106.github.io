---
layout: post
title:  "IBM Watson Bluemix Visual API : tutorial and visual accuracy of a custom classifier"
date:   2016-12-23 17:00:51
categories: computer vision
---

In this tutorial, I'll see how to use the standard visual classifier using IBM labels, then create our new custom classifier with Watson Visual API, last, see how it performs and compute its accuracy.


# Configure your Watson service

Go to the [Watson dashboard](https://console.ng.bluemix.net/dashboard/watson) in your Bluemix account, click on "Create a Watson service" and choose **Visual Recognition**, free usage. In the identification data for the service, create new data, which will return **url** and **api_key** values.

Let's set the `$API_KEY` shell variable to your IBM API KEY.

# Prepare Dataset

Let's download the [Food 101 dataset](https://www.vision.ee.ethz.ch/datasets_extra/food-101/) for our classification tutorial, to classify Food.

```bash
wget http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
tar xvzf food-101.tar.gz
```

Let's create the dataset :

- First, let's resize all images to 320 max dimension because IBM Watson Visual Recognition accepts images not more than 320x320 :

```bash
for file in food-101/images/*; do
  mogrify "$file/*.jpg[!320x320>]"
done
```

- Second, create train and test directories :

```bash
cd food-101
mkdir train test
for SPLIT in train test ;
do
  while read -r line
  do
      name="$line"
      DIRNAME="$SPLIT/"`dirname $name`
      if [ ! -d "$DIRNAME" ]; then
        echo "mkdir $DIRNAME"
        mkdir $DIRNAME
      fi
      cp "images/$line.jpg" "$SPLIT/$line.jpg"
      echo "Name read from file - $name"
  done < meta/$SPLIT.txt
done
```

- Third, for Watson API, keep the first 100 images per class :

```bash
for folder in train/*;
do
  count=0
  for file in $folder/*;
  do
    count=$((count+1))
    if [ $count -gt 100 ]
      then
        rm $file
    fi
done done
```

After many exchanges, the IBM Bluemix support tells to limit to 100 images per class, here is their answer trying to index the 1000 images per class :

*I have been working this issue with the engineers and it's looking more and more like we are hitting a performance wall due to the large number of images in each of your class datasets. It seems there is about a 1 second processing time per image when the service is under a typical user load (potentially longer under heavy user load). Given that each of your zip files contain roughly 1000 images each and the script is supplying 10 new files each time retraining is invoked, you can see that the training/retraining time can go pretty high very fast. Also to note... That when retraining occurs (using your script) we are adding another 10 new class files, to the first 10 classes that we've already trained on. So during the retraining process, we recall the first 10,000 image files from object storage and again process those, then we process the additional 10,000 images for a total of 20,000 images, and so on, and so on.*

*This is why it seems that the classifier is never done retraining and at some point we have seen the classifier get stuck in this retraining mode where it never comes out of retraining and there is a bug that has been opened for this issue with engineering.*

*I would suggest at this point that you reduce the number of examples in your classes to more like 100 - 200 for the sake of your proof of concept until engineering has addressed the training bug in an update.*

Also, at 0.10 USD per image, indexing 101 classes with 750 images would cost $7575 per month, so it is better to limit the number of images to index.


- Last, zip train images to prepare upload to Watson :

```bash
for file in train/*; do
  echo "zipping $file" ; zip -r "$file.zip" "$file"; rm -r $file;
done
```

# Use the standard classifier

The standard classifier comes with a huge dictionary of labels, for which models have been trained by IBM. let's give a try on one of our test images :  

    curl -X POST -F "images_file=@test/apple_pie/1011328.jpg" "https://gateway-a.watsonplatform.net/visual-recognition/api/v3/classify?api_key=$API_KEY&version=2016-05-20"


![]({{ site.url }}/img/apple_pie.jpg)

The classes retrieved for our apple pie are : sea hare, "shellfish, invertebrate, animal, giant conch...

    {
        "custom_classes": 0,
        "images": [
            {
                "classifiers": [
                    {
                        "classes": [
                            {
                                "class": "sea hare",
                                "score": 0.587,
                                "type_hierarchy": "/animal/invertebrate/shellfish  /sea hare"
                            },
                            {
                                "class": "shellfish  ",
                                "score": 0.919
                            },
                            {
                                "class": "invertebrate",
                                "score": 0.919
                            },
                            {
                                "class": "animal",
                                "score": 0.919
                            },
                            {
                                "class": "giant conch",
                                "score": 0.585,
                                "type_hierarchy": "/animal/invertebrate/shellfish  /giant conch"
                            },
                            {
                                "class": "conch",
                                "score": 0.554,
                                "type_hierarchy": "/animal/invertebrate/shellfish  /conch"
                            },
                            {
                                "class": "jade green color",
                                "score": 0.939
                            },
                            {
                                "class": "greenishness color",
                                "score": 0.856
                            }
                        ],
                        "classifier_id": "default",
                        "name": "default"
                    }
                ],
                "image": "1011328.jpg"
            }
        ],
        "images_processed": 1
    }

Since it is not satisfying for food classification, we'll have to create our own classifier.

# Create a custom classifier

Before uploading images to the classifier, we need to verify that each zip file is no more than 100 MB and 10 000 images per .zip file, no less than 10 images per size.

```bash
du -h food-101/images/*.zip
```

Each zip file is around 25MB (well balanced dataset - have the same number of images per class). Each zip file should not be above 100MB, otherwise we would have to divide them into multiple zip files.

Create the classifier under the name **food-101**. For creation, the service accepts a maximum of 300 MB so we can only upload up to 10 classes per update.

```bash
DATA=`pwd`/food-101/images
FIRST_CLASSES=`ls train/*.zip | awk 'NR >=1 && NR <=10 {split($0,a,"."); split(a[1],a,"/"); printf " -F " a[2] "_positive_examples=@" $0 }' -  `
echo $FIRST_CLASSES
```
returns what we need :
    -F apple_pie_positive_examples=@food-101/images/apple_pie.zip -F baby_back_ribs_positive_examples=@food-101/images/baby_back_ribs.zip -F baklava_positive_examples=@food-101/images/baklava.zip -F beef_carpaccio_positive_examples=@food-101/images/beef_carpaccio.zip -F beef_tartare_positive_examples=@food-101/images/beef_tartare.zip -F beet_salad_positive_examples=@food-101/images/beet_salad.zip -F beignets_positive_examples=@food-101/images/beignets.zip -F bibimbap_positive_examples=@food-101/images/bibimbap.zip -F bread_pudding_positive_examples=@food-101/images/bread_pudding.zip -F breakfast_burrito_positive_examples=@food-101/images/breakfast_burrito.zip

Let's upload them :

```bash
curl -X POST $FIRST_CLASSES -F "name=food-101" "https://gateway-a.watsonplatform.net/visual-recognition/api/v3/classifiers?api_key=$API_KEY&version=2016-05-20" > response.json
```

You can list your existing classifiers :

```bash
curl -X GET "https://gateway-a.watsonplatform.net/visual-recognition/api/v3/classifiers?api_key=$API_KEY&version=2016-05-20"
```

which gives

    {"classifiers": [
    {
        "classifier_id": "food101_1404391194",
        "name": "food-101",
        "status": "training"
    }
    ]}

or parse `response.json` to get the classifier ID :

    {
    "classifier_id": "food101_1404391194",
    "name": "food-101",
    "owner": "de67af8c-862c-4002-88a8-259653c880c4",
    "status": "training",
    "created": "2016-12-21T15:32:10.458Z",
    "classes": [
        {"class": "beef_carpaccio"},
        {"class": "baklava"},
        {"class": "baby_back_ribs"},
        {"class": "apple_pie"},
        {"class": "bibimbap"},
        {"class": "beignets"},
        {"class": "beet_salad"},
        {"class": "beef_tartare"},
        {"class": "breakfast_burrito"},
        {"class": "bread_pudding"}
    ]
    }


```bash
CLASSIFIER=`cat response.json | python -c "import json,sys;obj=json.load(sys.stdin);print obj['classifier_id'];"`
```

You can retrieve more info about the classifier :

```bash
curl -X GET "https://gateway-a.watsonplatform.net/visual-recognition/api/v3/classifiers/$CLASSIFIER?api_key=$API_KEY&version=2016-05-20"
```

    {
        "classifier_id": "food101_1404391194",
        "name": "food-101",
        "owner": "de67af8c-862c-4002-88a8-259653c880c4",
        "status": "ready",
        "created": "2016-12-21T15:32:10.458Z",
        "classes": [
            {"class": "beef_carpaccio"},
            {"class": "baklava"},
            {"class": "baby_back_ribs"},
            {"class": "apple_pie"},
            {"class": "bibimbap"},
            {"class": "beignets"},
            {"class": "beet_salad"},
            {"class": "beef_tartare"},
            {"class": "breakfast_burrito"},
            {"class": "bread_pudding"}
        ]
    }


# Update the created classifier with more classes

Let's upload the remaining 91 classes. The service accepts a maximum of 256 MB per training call so we can only upload up to 9 classes per update. Also, if you submit an update command without waiting the **ready** state, or without waiting that the retrained timestamp has not been updated, the previous update command will be erased/canceled, which is not an intented behavior. After the first update, a retrained timestamp appears in the classifier JSON. After each retraining, the retrained timestamp will be updated with the last retraining update. Last, the update command returns a **413 Request Entity Too Large** which is not an error and has to ignored :

    <html>
    <head><title>413 Request Entity Too Large</title></head>
    <body bgcolor="white">
    <center><h1>413 Request Entity Too Large</h1></center>
    <hr><center>nginx</center>
    </body>
    </html>


```bash
# Wait that classifier is in ready state
while [ "$STATUS" != "ready" ]
do
  STATUS=`curl -s -X GET "https://gateway-a.watsonplatform.net/visual-recognition/api/v3/classifiers/$CLASSIFIER?api_key=$API_KEY&version=2016-05-20" | python -c "import json,sys;obj=json.load(sys.stdin);print obj['status'];"`
  echo "Not ready. Waiting 10s."
  sleep 10s
done

for i in {0..10}; do

  CLASS_LIST=`ls train/*.zip | awk -v first="$(($i*9+10 + 1))" -v last="$(($i*9+10 + 9))" 'NR >=first && NR <=last {split($0,a,"."); split(a[1],a,"/"); printf " -F " a[2] "_positive_examples=@" $0 }' -  `

  COMMAND="curl -X POST $CLASS_LIST https://gateway-a.watsonplatform.net/visual-recognition/api/v3/classifiers/$CLASSIFIER?api_key=$API_KEY&version=2016-05-20"
  echo $COMMAND
  $COMMAND

  # wait the retrained timestamp is not void or has been updated
  while [ "$RETRAINED" == "" ] || [ "$RETRAINED" == "$TIMESTAMP" ]
  do
    RETRAINED=`curl -s -X GET "https://gateway-a.watsonplatform.net/visual-recognition/api/v3/classifiers/$CLASSIFIER?api_key=$API_KEY&version=2016-05-20" | python -c "import json,sys;obj=json.load(sys.stdin);print obj['retrained'];"`
    echo "Waiting update. Waiting 10s."
    sleep 10s
  done

  TIMESTAMP=$RETRAINED
  echo "New timestamp $TIMESTAMP"
done
```

# Classify an image with the custom classifier

    curl -X POST -F "images_file=@test/apple_pie/1011328.jpg" "https://gateway-a.watsonplatform.net/visual-recognition/api/v3/classify?api_key=$API_KEY&classifier_ids=$CLASSIFIER&version=2016-05-20"

It is possible to request multiple classifiers at the same time, by separating classifier IDs with a comma. The default classifier is 'Default'.


# Delete the custom classifier

Delete classifier :

```bash
curl -X DELETE "https://gateway-a.watsonplatform.net/visual-recognition/api/v3/classifiers/$CLASSIFIER?api_key=$API_KEY&version=2016-05-20"
```

# Compute accuracy of Watson Visual API

First let us create a python script *test_watson.py* to compute Watson top-1 and top-5 accuracy :

```python

import glob
import json,sys
import os
import pathlib
import os
from os.path import join, dirname

import collections
from watson_developer_cloud import VisualRecognitionV3

visual_recognition = VisualRecognitionV3('2016-05-20', api_key=os.environ.get('API_KEY'))

top_1_accuracy = 0
top_5_accuracy = 0
likelyhood = 0
nb_test_images=100
image_count=1
path  = "test/"
nb_samples = 0

for root, dirs, files in os.walk(path):
    for dir in dirs :
        dirpath = path+dir
        true_class = dir
        for file_name in glob.glob(dirpath+'/*'):
            print(file_name)

            with open(join(dirname(__file__), str(file_name)), 'rb') as image_file:
                input_data=json.dumps(visual_recognition.classify(images_file=image_file, threshold=0.1,
                                                             classifier_ids=[os.environ.get('CLASSIFIER')]), indent=2)
                obj1=json.loads(input_data)
                print(input_data)
                for doc in obj1["images"]:
                    nb_samples = nb_samples + 1
                    sorted_json = sorted(doc["classifiers"][0]["classes"], key=lambda k: (float(k["score"])),reverse=True)
                    if sorted_json[0]["class"] == true_class:
                        top_1_accuracy+=1
                        likelyhood+= sorted_json[0]["score"]
                    for count in range(5):
                        if sorted_json[count]["class"] == true_class:
                            top_5_accuracy += 1
                            print("True Class in top 5, position: ", count, ", class : ",sorted_json[count]["class"], "Score :  ",sorted_json[count]["score"])
                print("########    Results of the class : ", true_class ," sample ", nb_samples, "     ########")
                print("------------------------------------------------------------------")
                print("Top 1 Accuracy : ",top_1_accuracy * 1.0 / nb_samples ,"  Top 5 Accuracy : ",top_5_accuracy * 1.0 / nb_samples,"  Likely Hood :  " ,likelyhood)
                print("------------------------------------------------------------------")
```

and test it :

```bash
export API_KEY
export CLASSIFIER
python test_watson.py
```

I get a **top-1 accuracy : 12% and a top-5 accuracy of 23%**.


### Compare accuracy with state of the art

Let me compare accuracy we can get with Torch ResNet 101. Let's take the full dataset (750 training images per class) and 7% of these training images for validation :

```bash
mkdir val
for folder in train/*;
do
  mkdir val/${folder:6}
  ls $folder |sort -R |tail -50 |while read file; do
    mv $folder/$file val/${folder:6}/$file
    done
done
```

Training :

```bash
th main.lua -dataset imagenet -data /sharedfiles/food-101 -nClasses 101 -depth 101 -retrain resnet-101.t7 -resetClassifier true -batchSize 128 -LR 0.01 -momentum 0.96 -weightDecay 0.0001  -resume /sharedfiles/ -eGPU 2 -nThreads 8 -shareGradInput true  2>&1 | tee log-resnet-101-food-101-classes.txt
```

After a few hours of training, I get a **top-1 accuracy of 88% and a top-5 accuracy of 97.5% with our own model**. See transcript [here]({{ site.url }}/img/log-resnet-101-food-101-classes.txt).

**In conclusion, a general API as Watson Visual API is very far from the results you can achieve with an expert in computer vision.**

By the way, these results are far above what Stanford researchers got in their publication  [Deep Learning Based Food Recognition](http://cs229.stanford.edu/proj2016/report/YuMaoWang-Deep%20Learning%20Based%20Food%20Recognition-report.pdf) : top-1 72% and top-5 91%. In the previous world with [Random Forests](https://github.com/palexang/food-101), they got 40% top-1 accuracy.
