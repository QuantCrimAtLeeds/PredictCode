# Examples

IPython notebooks showing examples of the algorithms.

### Data sets

- [Example data sets](Example%20Data%20Sets.ipynb) quickly explores packages which provide data from Chicago, UK police forces (spatial only) and some synthetic data.

### Other notebooks

- [Time of day considerations](Time%20of%20day%20considerations.ipynb) a currently rather short meditation on the (lack of) consideration of "time of day" effects in the literature.

### Prediction algorithms

- [Retrospective hotspotting](Retrospective%20hotspotting.ipynb) a classical algorithm which looks merely at the spatial location of recent events to estimate a risk.
- [Prospetive hotspotting](Prospective%20HotSpot.ipynb) the Bowers, Johnson & Pease algorithm which weights more recent events.
- [Self-exciting points processes 1](Self-exciting%20point%20processes%201.ipynb) the Mohler et al. algorithm which uses an "epidemic type aftershock model" together with variable bandwidth kernel density estimation.  Produces a continuous risk intensity estimation.
- [Self-exciting points processes 2](Self-exciting%20point%20processes%202.ipynb) algorithm from a subsequent Mohler et al. paper (which concerns itself with two field trials).  This is a grid-based, parametric epidemic type aftershock model.