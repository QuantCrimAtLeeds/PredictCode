"""
common
~~~~~~

Every project ends up with a "common" area where we bung stuff.  In this case,
we use this as a dumping ground for model-related datatypes which affect most
of the application.
"""
import enum

CoordType = enum.IntEnum("CoordType", "LonLat XY")

CoordType._translation = {CoordType.LonLat : "Longitude/Latitude",
                          CoordType.XY : "Projected XY Coordinates"}