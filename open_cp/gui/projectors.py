"""
projections
~~~~~~~~~~~

Some code to do with handling projections, and extracting settings.  Was
originally mostly in the "comparator" module `geo_clip` but pulled out for
re-use.
"""

import logging
_logger = logging.getLogger(__name__)

try:
    import geopandas as gpd
    import shapely.ops
except Exception as ex:
    gpd = None
    _logger.error("geopandas/shapely not loaded because {}/{}".format(type(ex), ex))
try:
    import pyproj
except Exception as ex:
    pyproj = None
    _logger.error("pyproj not loaded because {}/{}".format(type(ex), ex))
    

class ProjectionFinder():
    """Encapulates producing a projector object from either an "epsg" code,
    or by using the main analysis model.
    
    :param analysis_model: The instance of :class:`analysis.Model` in use.
    """
    def __init__(self, analysis_model):
        self._model = analysis_model
        self._epsg = None
        
    def projected_dataset_coords(self):
        """Obtain, if possible using the current settings, the entire data-set
        of projected coordinates.  Returns `None` otherwise."""
        return self._model.analysis_tools_model.projected_coords()

    def chosen_projector(self):
        """Using the main model, finds the 1st selected projector (if possible)
        and returns a function object which sends pairs `(x,y)` to their
        projection."""
        proj = self._model.analysis_tools_model.coordinate_projector()
        if proj is None:
            return None
        return proj.make_tasks()[0]

    def projector(self):
        """Returns the best available projector: a callable object which sends
        coordinates `(x,y)` to their projection:
            
        - If set, then use the epsg code (with `pyproj`)
        - Otherwise, find the first selected projector from the main analysis
          model.
        - Otherwise return a function which just returns `(x,y)`.
        """
        if self.epsg is not None:
            return pyproj.Proj(self._crs_init_from_epsg())
        proj = self.chosen_projector()
        if proj is not None:
            return proj
        return self.NullOpProjector()

    class NullOpProjector():
        def __call__(self, x, y):
            return x, y

    def _crs_init_from_epsg(self):
        return {"init": "epsg:{}".format(self.epsg)}

    @property
    def epsg(self):
        """The selected output epsg code, or None."""
        return self._epsg

    @epsg.setter
    def epsg(self, value):
        self._epsg = value


class GeoFrameProjector(ProjectionFinder):
    """Subclass of :class:`ProjectionFinder` which stores a geoPandas data
    frame and can project it.  Attempts to extract the input "crs" code from
    the data frame (whenever it is set) and if fails, sets to epsg:4326, raw
    lon/lat.
    
    :param analysis_model: The instance of :class:`analysis.Model` in use.
    """
    def __init__(self, analysis_model):
        super().__init__(analysis_model)
        self._frame = None
        self._guessed = False

    @property
    def frame(self):
        """The geoPandas data frame in use, or `None`."""
        return self._frame
    
    @frame.setter
    def frame(self, v):
        self._frame = v
        if v is None:
            return
        if len(self._frame.crs) == 0:
            self._guessed = True
            self._frame.crs = {"init": "epsg:4326"}
        else:
            self._guessed = False

    @property
    def guessed_crs(self):
        """True/False: Did we guess that the input was longitude / latitude?"""
        return self._guessed

    @property
    def crs(self):
        """If we have a frame, then the "crs" dictionary; otherwise `None`"""
        if self._frame is None:
            return None
        return self._frame.crs

    def projected_frame(self):
        """Return the frame.  If an epsg code is set, project the frame
        using it.  Returns `None` if no frame, or in case of error."""
        if self._frame is None:
            return None
        try:
            if self.epsg is not None:
                return self._frame.to_crs(self._crs_init_from_epsg())
            else:
                return self._frame
        except:
            _logger.exception("While trying to project frame")
        return None

    def fully_projected_frame(self, projector=None):
        """Return the frame, projected using:
            
        - If an EPSG code is set, use it
        - If `projector` is not None, use it
        - Use the projector obtained from the base class
        - In the event of error, the unprojected frame
        - `None` if no frame
        """
        if self._frame is None:
            return None
        
        if self.epsg is not None:
            frame = self.projected_frame()
            if frame is None:
                return self._frame
            return frame
        
        if projector is None:
            projector = self.projector()
        if projector is None:
            return self._frame
        
        # Project back to lon/lat
        frame = self._frame.copy().to_crs({"init": "epsg:4326"})
        def proj(geo):
            return shapely.ops.transform(lambda x,y,z=None : projector(x,y), geo)
        frame.geometry = frame.geometry.map(proj)
        return frame
