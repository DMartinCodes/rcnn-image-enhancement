# rcnn-image-enhancement
This repository contains code for my Computational Photography final project, which involved the creation of a program to enhance images programmatically.

## Requirements
This program uses code from [the Mask-RCNN-TF2 project](https://github.com/ahmedfgad/Mask-RCNN-TF2), so the basic requirements to run this program are similar to the requirements for that project.
You will need the following:

- Python 3.7
- Preferably a Ubuntu or Debian installation
- A path to an image you would like to try!

## Installation
- Install Python 3.7 with `sudo apt install python3.7`
- Set up a Python virtual environment in a folder of your choice using `python3.7 -m venv rcnnenv`
- Clone this repository using `git clone https://github.com/DMartinCodes/rcnn-image-enhancement.git`
- Use `pip install -r requirements.txt` to load the required libraries

## Program operation
This program ships with an example image, so you can immediately get started by doing the following:
- Enter `python3 main.py example.png` into the terminal in the project directory.
- When prompted, input the number corresponding to the enhancement profile you wish to apply.
- When prompted, select whether or not you wish to upscale the image.
- Upon program completion, the modified image will appear in the \final_output directory.

## Demo
