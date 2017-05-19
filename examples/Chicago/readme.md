# Chicago data set

The city of chicago releases a rich dataset of crimes investigated by the police in the city.
- See https://catalog.data.gov/dataset/crimes-2001-to-present-398a4
- Also https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2
- These two sources give slightly differently formatted data, but the underlying data appears to be the same.

## Notebooks

- [Exploring the dataset](Exploring%20the%20dataset.ipynb) - Quickly look at the data.
- [Check large datasets agree](Check%20large%20datasets%20agree.ipynb) - Check that the two sources (as above) do give the same data.
- [Geo-coding of the Chicago dataset](Geo-coding%20of%20the%20Chicago%20dataset.ipynb) - Explore the geo-coding of events in the data.
- [Old Chicago Data](Old%20Chicago%20Data.ipynb) - Look at an (extract of) old data I have found (being careful not to reveal private information.)
- [Simulate spatial location](Simulate%20spatial%20location.ipynb) - (Work in progress) Attempt to randomly, but in a realistic way, change the geocoding of the dataset of that the events are less "clumpy".

## Geocoding

From speaking to other researchers, and looking at data dumps I have found online, it is apparent that in the past, the data released by the City of
Chicago had events geocoded to individual building locations.  However, the data currently available does not-- instead each event has coordinates which
resolve to the centre of the road, and in a "clumpy" way (so that, roughly speaking, only about half of each road ever has an event in it).  This has
presumably (and rather reasonably) been changed to provide more privacy in the data.  However, it means that the coordinates of events are false, in a
systematic way.

The notebook "Geo-coding of the Chicago dataset" explores this, and the notebook "Old Chicago Data" looks at some old data and compares how the geocoding
has changed.  The notebook "Simulate spatial location" looks at how to randomly restore the geocoding.




