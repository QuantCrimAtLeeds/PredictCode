# Installing

For Python pros, just look at [`requirements.txt`](../requirements.txt)

Otherwise, we _strongly recommend_ using [Anaconda](https://www.anaconda.com/).  Download and install this, which will give you access to standard libraries like `numpy` and `scipy`.  We then need to install some further libraries:

- [GeoPandas](http://geopandas.org/)  Which can be installed via:

        conda install -c conda-forge geopandas

On Linux, this has caused [problems in the past](https://github.com/QuantCrimAtLeeds/PredictCode/issues/3) and trying recently, I have had other problems.  On Windows, it seems to work fine.

- [TileMapBase](https://github.com/MatthewDaws/TileMapBase)  which is pure-python, so can be installed via:

        pip install tilemapbase



## Install `open_cp`

You can do this directly from github.  Open "anaconda prompt" and run

    pip install https://github.com/QuantCrimAtLeeds/PredictCode/zipball/master

Alternatively, download/clone the GitHub repository, and run `python setup.py install`.
