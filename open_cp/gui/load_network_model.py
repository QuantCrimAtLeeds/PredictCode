"""
load_network_model
~~~~~~~~~~~~~~~~~~
"""

import open_cp.gui.predictors.geo_clip as geo_clip
import open_cp.network
import enum as enum
import logging

try:
    import geopandas as gpd
except:
    gpd = None

_logger = logging.getLogger(__name__)

class NetworkModel():
    """Encapsulates the graph/network data which will be used to make a network
    prediction.  Also stores geometry data for clipping purposes."""
    def __init__(self, analysis_model):
        self._crop_to_geometry = geo_clip.CropToGeometry(analysis_model)
        self.filename = None
        self.network_type = self.NetworkType.ORDNANCE
        self._error = None
        self._graph = None
        self.backup(False)
        
    @property
    def crop_to_geometry(self):
        """Instances of :class:`geo_clip.CropToGeometry` which gives clipping
        information.
        """
        return self._crop_to_geometry
    
    def to_dict(self):
        data = { "geo_clip" : self.crop_to_geometry.to_dict(),
            "type" : self.network_type.name
            }
        if self.filename is not None:
            data["filename"] = self.filename
        return data
    
    def from_dict(self, data):
        self.crop_to_geometry.from_dict(data["geo_clip"])
        self.network_type = self.NetworkType[data["type"]]
        if "filename" in data:
            self.filename = data["filename"]
        else:
            self.filename = None
    
    def backup(self, make_new=True):
        """Internally store old state, so we can quickly recover if we decide
        we don't like changed.  Much faster than calling `to_dict` and then
        `from_dict`."""
        if make_new:
            data = self.to_dict()
            self._backup = (data, self.filename, self.graph)
        else:
            self._backup = None
        
    def restore(self):
        """Partner of :meth:`backup`."""
        if self._backup is None:
            return
        data, filename, graph = self._backup
        if "filename" in data:
            del data["filename"]
        self.from_dict(data)
        self._graph = graph
        self._filename = filename
    
    def _load_network(self):
        try:
            _logger.debug("Attempting to load network file %s", self.filename)
            frame = gpd.GeoDataFrame.from_file(self.filename)
            if self.network_type == self.NetworkType.ORDNANCE:
                self._graph = self._load_OS_style(frame)
            elif self.network_type == self.NetworkType.TIGER_LINES:
                self._graph = self._load_tiger_style(frame)
            else:
                raise NotImplementedError()
            _logger.debug("Built graph with %s vertices and %s edges", len(self.graph.vertices), len(self.graph.edges))
        except Exception as ex:
            self._error = "{}/{}".format(type(ex), ex)
            self.filename = None
            self._graph = None

    @staticmethod
    def _load_OS_style(frame):
        _logger.debug("Converting to OS style network")
        b = open_cp.network.PlanarGraphGeoBuilder()
        for line in frame.geometry:
            b.add_path(line.coords)
        return b.build()

    @staticmethod
    def _load_tiger_style(frame):
        _logger.debug("Converting to TIGER/Lines (tm) style network")
        all_nodes = []
        for geo in frame.geometry:
            for pt in geo.coords:
                all_nodes.append(pt)
                
        b = open_cp.network.PlanarGraphNodeOneShot(all_nodes)
        for geo in frame.geometry:
            path = list(geo.coords)
            b.add_path(path)
        
        b.remove_duplicate_edges()
        return b.build()
    
    @property
    def filename(self):
        """The filename of the geometry data giving the network.  Or `None`"""
        return self._filename
    
    @filename.setter
    def filename(self, v):
        self._filename = v
        if v is not None:
            self._load_network()
        
    class NetworkType(enum.Enum):
        ORDNANCE = 0
        TIGER_LINES = 1
        
    @property
    def network_type(self):
        """The type of network, instance of :class`NetworkType`"""
        return self._network_type
        
    @network_type.setter
    def network_type(self, v):
        self._network_type = v
    
    @property
    def graph(self):
        return self._graph

    def consume_recent_error(self):
        e = self._error
        self._error = None
        return e
