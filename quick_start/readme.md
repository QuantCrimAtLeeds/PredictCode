# Quick start

A quick guide to installing and getting going with the software.



## Install

This is less easy than it should be, thanks to the Python ecosystem.  See the [guide here](install.md).


## GUI mode

There is a self-contained GUI application, which can be run (either install first, or run from the main directory of the reposity) as:

    python -m open_cp
    
The GUI mode lets you import CSV data (with a helpful graphical import utility, allowing you to specify which fields to read, etc. etc.) and then to select and run a variety of prediction algorithms on that data.  Results can be saved back to a CSV file, or in an internal (`pickle` based) format for later reloading.  Some disadvantages:

- Python is just rather slow; running a GUI application overnight is dull.
- Network based algorithms are not currently implemented in the GUI (as they are incredibly slow, and use a large amount of resource.)
- We have found that the actual _algorithm_ is normally less important than the _settings_ chosen for that given algorithm.  Given the speed issue, interactively exploring different settings is almost impossible.


## Python library

The main part of `open_cp` (not in terms of code, but it terms of importance) is a Python library implementing the different prediction algorithms.  This is designed for use by people who have some familiarity with Python, and in particular, is best used with the Jupyter Notebook technology.  There are many examples in the repository which are hopefully more than enough to get going.


## `Scripted` module

A half-way house between the full python library and the GUI mode.  This is a (much) simplified API to the main python library, and allows you to write _short_ python scripts which load data, run algorithsm and assessment methods, and then save data to e.g. CSV format.  Writing small scripts gives a huge advantage in being _reproducible_, while the output could be analysed in your programme of choice (the examples use Jupyter Notebooks and Python).

An example [step by step guide is available](scripted_intro.md).