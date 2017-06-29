# Evaluating hotspots

For the moment, we shall presume we are only comparing grid based predictions (though with an eye to possibly every small grid cells arising from continuous predictions).

We first discuss a very high-level overview of a "pipeline" for producing predictions, scoring them, and comparing different predictions.  We then provide more details on each stage.

### Prediction pipeline

1. **Certain grid cells are flagged.**  (E.g. A prediction technique assigns a "risk" to each cell, and the top 5% of cells are flagged.)
2. **The prediction is compared against the actuality.** (E.g. For the next day, we count the number of crime events which occurred in flagged cells, compared to the total crime count.)
3. **Other properties of the prediction are computed.** (E.g. a measure of "compactness" or "clumpiness" of the flagged areas; broadly we seek to assess how _practically_ useful the prediction is.)
4. **This is repeated for many time periods.** (E.g. For each consequentive day over 6 months of data.)
5. **The resulting time series are compared.** (E.g. through the use of summary statistics, or using a statistical test.)


#### Flagging grid cells

1. Flag the top X% of cells by predicted risk.  This gives a certain "coverage level".  It is usual to clip the cells to a geographic region, instead of using say the whole bounding rectangle (this decreases the number of cells under consideration, and so decreases the area coveraged at a given % coverage).


#### Compare against reality

1. Deem that a crime was "predicted" if and only if it falls in a flagged grid cell.


#### Other properties of the prediction

TODO: Read [4] and [5]



#### Compare the results

We presume that for each prediction, we end up with some sort of "score" (perhaps many scores for different measures).

1. Display the resulting time series.  These are often extremely noisy, so it might be useful to think of different plot types?
2. Produce summary statistics: e.g. the mean value.
3. Apply statistical tests, e.g. see [1] and [2].


## References

(Not in alphabetical order, sorry, so that adding a new article doesn't require changing all the numbers above :smile:)

1. Diebold, Mariano, "Comparing Predictive Accuracy", Journal of Business and Economic Statistics, 13 (1995)  253--265.  [Journal homepage](http://amstat.tandfonline.com/doi/abs/10.1080/07350015.1995.10524599) or [JSTOR](https://www.jstor.org/stable/1392185)
2. Diebold, "Comparing Predictive Accuracy, Twenty Years Later: A Personal Perspective on the Use and Abuse of Diebold–Mariano Tests" Journal of Business and Economic Statistics, 33 (2015)  253--265. [DOI:10.1080/07350015.2014.983236](http://dx.doi.org/10.1080/07350015.2014.983236)
3. Johnson et al, "Towards the Modest Predictability of Daily Burglary Counts" Policing 6 (2012) 167--176 [DOI: 10.1093/police/pas013](https://doi.org/10.1093/police/pas013)
4. Bowers at al, "Prospective hot-spotting: The future of crime mapping?", Brit. J. Criminol. 44 (2004) 641--658. [DOI:10.1093/bjc/azh036](https://doi.org/10.1093/bjc/azh036)
5. Adepeju et al, "Novel evaluation metrics for sparse spatiotemporal point process hotspot predictions - a crime case study", International Journal of Geographical Information Science, 30:11, 2133-2154, [DOI:10.1080/13658816.2016.1159684](https://doi.org/10.1080/13658816.2016.1159684)
6. Turner, "Landscape Ecology: The Effect of Pattern on Process", 
Annual Review of Ecology and Systematics 20 (1989) 171--197 [DOI: annurev.es.20.110189.001131](https://doi.org/10.1146/annurev.es.20.110189.001131)

