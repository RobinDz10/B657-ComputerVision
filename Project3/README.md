# How to run the code
```
$ python3 run.py --help

python3 run.py [options] -i file
options:
    --debug: save all temporary files (debug only, could let program run much slower)
    --help: print help function
    --lang [language]: specify pytesseract languages in your system(list below), default: 'eng'
    --direct: don't show interactive window after completed

['chi_sim', 'eng', 'osd', 'snum']
```
A solid example:
`$ python3 run.py --debug -i example.jpg` (a file called `example.jpg` should exist on the same directory)    

After running the code, a file named `remove_privacy.png` will exist on the same directory.

# required package/environment
opencv-python    
pytesseract     (system library `Tesseract` is also required)    
pyenchant     (system library `enchant` is also required)    
numpy    
wordfreq    
