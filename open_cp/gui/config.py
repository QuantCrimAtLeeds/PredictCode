"""
config
~~~~~~

Configuration options.
"""

import open_cp.gui.tk.config_view as config_view


class Config():
    def __init__(self, root):
        self._root = root
        self.model = Model()
    
    def run(self):
        self.view = config_view.ConfigView(self._root, self)
        self._init_themes()
        self.view.wait_window(self.view)

    def _init_themes(self):
        self.model.themes = self.view.theme_names()
        self.view.set_themes(self.model.themes)
        self.model.theme = self.view.current_theme()
        self.view.set_theme_selected(self.model.get_theme_index())

    def selected_theme(self, index):
        self.model.theme = index
        self.view.set_theme(self.model.theme)


class Model():
    def __init__(self):
        pass

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