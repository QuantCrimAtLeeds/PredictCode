"""
settings.py
~~~~~~~~~~~

Simple abstraction of a key/value storage, for storing configuration data
and settings.
"""

import collections as _collections
import os as _os
import logging as _logging
import json as _json

_logger = _logging.getLogger(__name__)

class Settings(_collections.UserDict):
    """Simple dictionary-like object for storing settings and other
    configuration.  Serialises itself as a JSON file.
    
    Supports the context manager protocol, and on exiting the context saves
    itself.
    
    :param filename: The JSON file to save/load from.  Defaults to
      `open_cp_ui_settings.json` in the users home space.
    """
    def __init__(self, filename = None):
        super().__init__()
        if filename is None:
            filename = self._default_filename()
        self._filename = filename
        _logger.info("Using filename '%s'", filename)
        self._load()
        
    def _load(self):
        try:
            with open(self._filename, "rt") as settings_file:
                d = _json.load(settings_file)
                self.data = d
        except FileNotFoundError:
            _logger.info("No settings file found, using defaults.")
            pass
        except _json.JSONDecodeError as ex:
            _logger.error("Failed to load settings file as using defaults; caused by %s", ex)
        
    @property
    def filename(self):
        """The filename is use."""
        return self._filename
        
    @staticmethod
    def _default_filename():
        home = _os.path.expanduser("~")
        return _os.path.join(home, "open_cp_ui_settings.json")
        
    def save(self):
        """Save the current state to the settings file."""
        json_string = _json.dumps(self.data, indent=2)
        with open(self._filename, "wt") as settings_file:
            settings_file.write(json_string)
        _logger.info("Wrote settings to %s", self.filename)
            
    def __enter__(self):
        return self
    
    def __exit__(self, extype, a, b):
        self.save()
        