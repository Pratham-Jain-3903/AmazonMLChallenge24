# Image Feature Extraction Solution (PLEASE NOTE THAT .env AND .idea folders have been removed to make size <500mb>)

## Table of Contents
1. [Initial Approach](#initial-approach)
2. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
3. [OCR Experiments](#ocr-experiments)
4. [Machine Learning Approach](#machine-learning-approach)
5. [Final Solution: Qwen2-VL-for-OCR-VQA](#final-solution-qwen2-vl-for-ocr-vqa)
6. [Conclusion](#conclusion)

## Initial Approach

Our team tackled the problem of feature extraction from images using a multi-step process. We began with exploratory data analysis to understand the dataset, followed by experiments with various OCR technologies and machine learning models. Ultimately, we found success using a pre-trained vision-language model.

**Important Note:** Throughout our entire process, we did not use any external APIs or Gateways. All processing was done locally using publicly available tools and models.

## Exploratory Data Analysis (EDA)

We started by analyzing the distribution of entity types in our dataset. Our findings showed that the number of unique classes was 8, with the following distribution:

```
entity_name
depth                             45127
height                            43597
item_volume                        7682
item_weight                      102786
maximum_weight_recommendation      3263
voltage                            9466
wattage                            7755
width                             44183
```

To create a balanced dataset for our initial experiments, we selected 2,000 samples from each entity type.

## OCR Experiments

We conducted experiments with various OCR (Optical Character Recognition) technologies:

1. Tesseract
2. EasyOCR
3. KerasOCR

After testing on a small sample of 100 files, we found that EasyOCR performed the best. We then proceeded to run OCR on all files in the test set.

## Machine Learning Approach

Simultaneously with our OCR efforts, we explored a machine learning approach:

1. We used VGG16 for feature extraction from images.
2. We split the entity values into two parts: units and numeric values.
3. We attempted to build a regression and classification ensemble using PyCaret.

However, this approach took longer than expected to build and train.

## Final Solution: Qwen2-VL-for-OCR-VQA

As our machine learning approach was time-consuming, we experimented with pre-trained models available in public libraries. We tried several models, including:

1. LLaVA
2. Eagle-V5-13B
3. Qwen2-VL-for-OCR-VQA

The Qwen2-VL-for-OCR-VQA model proved to be the fastest and most effective for our task. This model combines OCR capabilities with visual question answering, allowing us to extract entity values from images efficiently.

**Key Point:** Qwen2-VL-for-OCR-VQA is a publicly available model in the TensorFlow library. We did not use any proprietary or closed-source solutions.

### Implementation Details

1. We downloaded all test images overnight to ensure we had local access to the data.
2. We used the Qwen2-VL-for-OCR-VQA model from the TensorFlow library to process each image and extract the required entity values.
3. We applied regex post-processing to ensure our output matched the required format and passed the sanity check.

## Conclusion

Our journey to solve this problem involved multiple approaches and technologies, all of which were publicly available and processed locally. While our initial machine learning approach showed promise, the pre-trained Qwen2-VL-for-OCR-VQA model from the TensorFlow library provided a more efficient and accurate solution. 

This experience highlights the importance of exploring various methods and being open to leveraging existing open-source technologies in solving complex problems. It also demonstrates that powerful solutions can be implemented without relying on external APIs or gateways, using only publicly available tools and models.