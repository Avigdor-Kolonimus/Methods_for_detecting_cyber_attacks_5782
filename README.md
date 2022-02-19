# Methods for detecting cyber attacks

## Introduction[](#introduction)

The course of "Methods of detecting cyber attacks" by Ariel University in academic year 2021-2022. The course was divided into two parts. In the first step, I took part in the [Malware Detection and Classification Challenge](https://eval.ai/web/challenges/challenge-page/1357/evaluation), which was sponsored by Cisco. The second step was to pick some kind of scientific paper related to the course topic and improved its results.

---

## Table of contents[](#table-of-contents)
1. [Challenge](#challenge)
2. [Project](#project)
3. [Licensing](#licensing)

---

## Challenge[](#challenge)

According to the introduction, this is the "Malware Detection and Classification Challenge". There are datasets in the [folder](./CISCO%20challenge/datasets) that we received for different phases, but the dataset for Zero-day isn't there. No changes were made to the file [code](./CISCO%20challenge/code.ipynb) for all phases. A [presentation](./CISCO%20challenge/slides.pptx) file was presented to the CISCO judges. In the Zero-day phase, the second place was taken, but the overall result is 1st place.

---

## Project[](#project)

Cyber criminals are always looking for effective vectors to deliver malware to victims in order to launch an attack. Images are used on a daily basis by millions of people around the world, and most users consider images to be safe for use; however, some types of images can contain a malicious payload and perform harmful actions. In this [paper](https://ieeexplore.ieee.org/document/8967109/authors#authors), Aviad Cohen, Nir Nissim and Yuval Elovici present MalJPEG, the first machine learning-based solution tailored specifically at the efficient detection of unknown malicious JPEG images. MalJPEG statically extracts 10 simple yet discriminative features from the JPEG file structure and leverages them with a machine learning classifier, in order to discriminate between benign and malicious JPEG images. Most of the papers can be found on JPEG images focused on steganography methods, steganography analysis (steganalysis) methods, or adversarial images. The project’s contributions are as follows:
* For comparative studies, Aviad Cohen, Nir Nissim and Yuval Elovici used the histogram image as features. My experiment with histogram equalization and quantization will demonstrate how image processing affects the result.
* The MalJPEG uses 10 markers. In my project, I found new markers and demonstrated their information gain rank for further work.

---

## Licensing[](#licensing)
 
MIT License is dual-licensed under free open source license (Apache version 2 license) and commercial license, which gives the commercial support, full rights to create and distribute software without open source license obligations. For licensing details see [LICENSE](./LICENSE.txt) document.

---

<!-- markdownlint-enable -->
