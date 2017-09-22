# Patrol plans

A traditional approach to forming "hotspots" is via "coverage levels":

1. Produce, using a prediction algorithm, an estimate of "risk" in each grid cell.
2. Decide upon a "coverage level", say 10%, and take simply the top 10% of grid cells by risk.

This may form a pattern of spatial locations which look like "hot spots", or it may not.  (For network based prediction techniques, the same applies, with "grid cell" replaced by "edge of network").

Alternatively, we might replace the 2nd step by an algorithm which seeks always to form "hotspots" (according to some criteria) using the risk of each grid cell (or network edge) merely as a guide.  We believe this is more likely to lead to usable hotspots, from a police operational viewpoint.