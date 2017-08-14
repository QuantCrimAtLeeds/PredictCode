"""
load_network_model
~~~~~~~~~~~~~~~~~~
"""

import open_cp.gui.predictors.geo_clip as geo_clip
import enum as _enum
    
class NetworkModel():
    """Encapsulates the graph/network data which will be used to make a network
    prediction.  Also stores geometry data for clipping purposes."""
    def __init__(self, analysis_model):
        self._crop_to_geometry = geo_clip.CropToGeometry(analysis_model)
        self.filename = None
        self.network_type = self.NetworkType.ORDNANCE
        
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
    
    @property
    def filename(self):
        """The filename of the geometry data giving the network.  Or `None`"""
        return self._filename
    
    @filename.setter
    def filename(self, v):
        self._filename = v
        
    class NetworkType(_enum.Enum):
        ORDNANCE = 0
        TIGER_LINES = 1
        
    @property
    def network_type(self):
        """The type of network, instance of :class`NetworkType`"""
        return self._network_type
        
    @network_type.setter
    def network_type(self, v):
        self._network_type = v
        