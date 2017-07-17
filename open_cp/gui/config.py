"""
config
~~~~~~

Configuration options.
"""

import open_cp.gui.tk.config_view as config_view
import open_cp.gui.locator as locator
import logging as _logging

class Config():
    def __init__(self, root):
        self._root = root
        self._logger = _logging.getLogger(__name__)
        self.model = Model()
        self.model.themes = self._root.style.theme_names()
        self.model.theme = self._root.style.theme_use()
        self.model.load_from_settings()
        self._set_theme()
    
    def run(self):
        self.view = config_view.ConfigView(self._root, self.model, self)
        self.model_dict = self.model.to_dict()
        self.view.set_theme_selected(self.model.get_theme_index())
        self.view.wait_window(self.view)

    def _init_themes(self):
        self.model.themes = self.view.theme_names()
        self.view.set_themes(self.model.themes)
        self.model.theme = self.view.current_theme()
        self.view.set_theme_selected(self.model.get_theme_index())

    def _set_theme(self):
        self._logger.debug("Setting ttk theme to '%s'", self.model.theme)
        self._root.style.theme_use(self.model.theme)

    def selected_theme(self, index):
        self.model.theme = index
        self._set_theme()
        self.view.resize()
        
    def reset(self):
        self.model.from_dict(self.model_dict)
        self.selected_theme(self.model.get_theme_index())

    def okay(self):
        self.model.save_to_settings()
        
    def cancel(self):
        self.reset()


class Model():
    def __init__(self):
        self._logger = _logging.getLogger(__name__)
        self.settings = locator.get("settings")

    def to_dict(self):
        return {"theme" : self.theme}
        
    def from_dict(self, data):
        try:
            self.theme = data["theme"]
        except ValueError:
            self._logger.warning("Cannot find theme name '%s'.  Using default.", data["theme"])

    def save_to_settings(self):
        for key, value in self.to_dict().items():
            self.settings[key] = value
        self.settings.save()

    def load_from_settings(self):
        """Should be run after :attr:`themes` has been populated."""
        data = self.to_dict()
        for key in set(data.keys()) :
            try:
                data[key] = self.settings[key]
            except KeyError:
                self._logger.warning("Failed to find setting '%s' in settings", key)
        self.from_dict(data)

    @property
    def settings_filename(self):
        return self.settings.filename

    @property
    def themes(self):
        return self._themes

    @themes.setter
    def themes(self, v):
        self._themes = list(v)

    @property
    def theme(self):
        return self.themes[self._theme_index]

    @theme.setter
    def theme(self, v):
        if isinstance(v, str):
            v = self.themes.index(v)
        self._theme_index = v

    def get_theme_index(self):
        return self._theme_index
