# Examples

IPython notebooks showing examples of the algorithms.

### Data sets

- [Example data sets](Example%20Data%20Sets.ipynb) quickly explores packages which provide data from Chicago, UK police forces (spatial only) and some synthetic data.
- [Chicago](Chicago) explores the chicago dataset in more detail.  This dataset is seemingly widely used in the literature, but the way events are geocoded seems to have changed, and so directly reproducing old research is hard.  We explore this problem.

### Other notebooks

- [Time of day considerations](Time%20of%20day%20considerations.ipynb) a currently rather short meditation on the (lack of) consideration of "time of day" effects in the literature.

### Prediction algorithms

- [Naive](Naive.ipynb) two very "naive" algorithms which set the scene for what a baseline prediction looks like.
- [Retrospective hotspotting](Retrospective%20hotspotting.ipynb) a classical algorithm which looks merely at the spatial location of recent events to estimate a risk.
- [Prospetive hotspotting](Prospective%20HotSpot.ipynb) the Bowers, Johnson & Pease algorithm which weights more recent events.
- [Time/Space KDE](Time-Space KDE.ipynb) a more general algorithm which generalises all of the above.
- [Self-exciting points processes 1](Self-exciting%20point%20processes%201.ipynb) the Mohler et al. algorithm which uses an "epidemic type aftershock model" together with variable bandwidth kernel density estimation.  Produces a continuous risk intensity estimation.
- [Self-exciting points processes 2](Self-exciting%20point%20processes%202.ipynb) algorithm from a subsequent Mohler et al. paper (which concerns itself with two field trials).  This is a grid-based, parametric epidemic type aftershock model.
- [Space-Time Scan Statistic](Space-Time%20Scan%20Statistic.ipynb) algorithm, replicating closely the SatScan software.

### Network predictions

See the sub-directory [networks](Networks/) for a network algorithm and example case study.
