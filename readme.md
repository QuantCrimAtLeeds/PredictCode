[![Build Status](https://travis-ci.org/QuantCrimAtLeeds/PredictCode.svg?branch=master)](https://travis-ci.org/QuantCrimAtLeeds/PredictCode)

# Predictive algorithms for crime

Python based implementations of algorithms to predict crime (which has a strong spatial
component, e.g. Burglary).  Current work has concentrated on giving implementations of
algorithms from published sources.

- [Example Jupyter notebooks](examples) showing the different algorithms in action, together with
  discussion of the literature and implementation details.
- [Notebooks](notebooks) giving more technical details
- [Evaluation](evaluation) gives details about different evaluation/comparison methods
  for comparing predictions.  (ToDo: Link to academic paper once finished.)
- [Documentation](https://quantcrimatleeds.github.io/PredictCode/) extracted from the doc strings in the Python code.  (ToDo: Rebuild, as hasn't been compiled in ages).


## GUI mode

Currently a work in progress.  Can be run without installation via:

    python -m open_cp


## Install

Most notebooks can be run without installation.  To install,

    python setup.py install
