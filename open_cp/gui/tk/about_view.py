import tkinter as tk
import tkinter.ttk as ttk
import tkinter.font as tkfont
from . import util

_text = {
    "title" : "About",
    "name" : "OpenCP",
    "desc" : "Standardised implementations of prediction techniques in the literature",
    "github" : "QuantCrimAtLeeds / PredictCode",
    "ghurl" : "https://github.com/QuantCrimAtLeeds/PredictCode",
    "homeoffice" : """This work was funded by the UK Home Office Police Innovation Fund, through the project "More with Less: Authentic Implementation of Evidence-Based Predictive Patrol Plans".""",
    "ld" : "Lead developer(s):",
    "devs" : [("Dr Matthew Daws", "http://matthewdaws.github.io/")],
    "stat" : "Crime prediction analyst(s):",
    "stats" : [("Dr Monsuru Adepeju", "https://www.geog.leeds.ac.uk/people/m.adepeju")],
    "proj" : "Project lead(s):",
    "projs" : [("Dr Andy Evans", "http://www.geog.leeds.ac.uk/people/a.evans/")],
    "python" : "Built using the Python packages: ",
    "packs" : [ ("numpy", "http://www.numpy.org/"),
            ("scipy", "https://www.scipy.org/"),
            ("pyproj", "https://github.com/jswhit/pyproj"),
            ("shapely", "https://github.com/Toblerity/Shapely"),
            ("dateutil", "https://dateutil.readthedocs.io/en/stable/"),
            ("matplotlib", "https://matplotlib.org/"),
            ("pytest", "https://docs.pytest.org/en/latest/"),
            ("codecov", "https://github.com/codecov/codecov-python"),
            ("pytest-cov", "https://pytest-cov.readthedocs.io/en/latest/"),
            ("Pillow", "https://pypi.python.org/pypi/Pillow/2.1.0"),
            ("geopandas", "http://geopandas.org/"),
            ("rtree", "http://toblerity.org/rtree/"),
            ("pytest", "https://docs.pytest.org/en/latest/"),
            ("codecov", "https://github.com/codecov/codecov-python"),
            ("pytest-cov", "http://pytest-cov.readthedocs.io/en/latest/"),
            ("tilemapbase", "https://github.com/MatthewDaws/TileMapBase")
            ],
    "other" : "Other resources",
    "others" : [("Icons by Madebyoliver, CC 3.0 BY", "http://www.flaticon.com/authors/madebyoliver")]
}

class AboutView(util.ModalWindow):
    def __init__(self, controller, parent):
        self._controller = controller
        super().__init__(parent, _text["title"])
        self.set_size_percentage(30,30)
        self.set_to_actual_width()
    
    def add_widgets(self):
        util.stretchy_rows_cols(self, [0], [0])
        frame = util.ScrolledFrame(self, "v")
        frame.grid(sticky=tk.NSEW)
        frame = frame.frame

        font = tkfont.nametofont("TkTextFont").copy()
        font["size"] = 18
        tk.Label(frame, text=_text["name"], font=font).grid(padx=2, pady=3)

        ttk.Label(frame, text=_text["desc"], anchor=tk.W).grid(padx=2, pady=1, sticky=tk.W)
        util.HREF(frame, text=_text["github"], url=_text["ghurl"]).grid(padx=2, pady=1, sticky=tk.W)
        
        ttk.Separator(frame).grid(sticky=tk.EW, padx=5, pady=3)
        self._funding(frame)
        
        ttk.Separator(frame).grid(sticky=tk.EW, padx=5, pady=3)
        self._main_credits(frame)

        ttk.Separator(frame).grid(sticky=tk.EW, padx=5, pady=3)
        self._others(frame)

    def _funding(self, frame):
        ttk.Label(frame, text=_text["homeoffice"], wraplength=350).grid(padx=2, pady=2, sticky=tk.W)

    def url_list(self, frame, title, li):
        ttk.Label(frame, text=title).grid(padx=2, pady=2, sticky=tk.W)
        for name, url in li:
            f = tk.Frame(frame)
            f.grid(sticky=tk.W)
            tk.Label(f, text="     ").grid(row=0, column=0)
            util.HREF(f, text=name, url=url).grid(row=0, column=1)

    def _main_credits(self, frame):
        self.url_list(frame, _text["ld"], _text["devs"])
        self.url_list(frame, _text["stat"], _text["stats"])
        self.url_list(frame, _text["proj"], _text["projs"])

    def _others(self, frame):
        self.url_list(frame, _text["python"], _text["packs"])
        self.url_list(frame, _text["other"], _text["others"])
