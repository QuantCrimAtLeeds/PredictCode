"""
geo_clip
~~~~~~~~

Crop any "prediction" to some real-world geometry.

For e.g. grid-based predictors, we produce a prediction on a grid which is by
default the bounding-box of the input data points.  This is likely to be wrong
for prediction purposes:

  - It ignores edge effects
  - It ignored the geometry of the world-- likely the input data comes from
    some e.g. administrative boundary.  We probably want to clip the output to
    such a boundary as well.

This module allows loading geometry and clipping an e.g. output grid to the
geometry.

Required `geopandas` but fails gracefully...

"""

from . import comparitor
import logging
import tkinter as tk
import tkinter.ttk as ttk
import open_cp.gui.tk.util as util
import open_cp.gui.tk.tooltips as tooltips
import open_cp.gui.tk.richtext as richtext
import open_cp.gui.funcs as funcs
import open_cp.gui.tk.mtp as mtp
import open_cp.geometry
import open_cp.data

_logger = logging.getLogger(__name__)

try:
    import geopandas as gpd
except Exception as ex:
    gpd = None
    _logger.error("geopandas not loaded because {}/{}".format(type(ex), ex))
try:
    import descartes
    import shapely.ops
except Exception as ex:
    gpd = None
    _logger.error("descartes not loaded because {}/{}".format(type(ex), ex))


_text = {
    "main" : ("Crop to Geometry\n\n"
              + "Allows loading of geometry, previewing, and clipping any predictions to the geometry."
        ),
    "nogpd" : "Could not load geopandas, so no geometry loading will be supported.",
    "load" : "Load Geometry",
    "filett" : "Filename of the loaded geometry",
    "icrs" : "Input crs: {}",
    "icrstt" : ("The input Coordinate Reference System (CRS) detected for the geometry.  Specifies "
            + "how the geometry related to the real world.  The code 'epsg:4326' indicates that "
            + "the data is in Longitude/Latitude format; this is also often the default choice."),
    "icrstt1" : "No input projection information found, so we assumed the coordinates are longitude / latitude.",
    "newcrs" : "Transform to epsg:",
    "epsg_url" : "http://spatialreference.org/ref/epsg/",
    "epsg_url_text" : "(Click for a list)",
    "newcrstt" : ("The EPSG code to transform the geometry to.  "
            + "This should be the same as the projection used for the input data.  "
            + "If left blank, we make a best guess, which is likely the best choice if the events are given as Longitude/Latitude."),
    "preview" : "Preview of the geometry",
    "fail" : "Failed to load: {}/{}",
    "guesscrs" : "Assumed Longitude/Latitude",
    "preview2" : "After transform and with input events",
    "previewtt" : "A visualisation of the loaded geometry, with the detected projection",
    "previewnonett" : "No preview, as either no geometry selected, or geometry could not be loaded",
    "previewtt2" : ("A visualisation with the location of the events.  "
            + "The geometry is now projected using the given EPSG code or the best guess.  "
            + "Check that the points correctly align with the geometry!"),
    
}


class CropToGeometry(comparitor.Comparitor):
    def __init__(self, model):
        super().__init__(model)
        self._filename = None
        self._epsg = None

        self._frame = None
        self._error = None
    
    @staticmethod
    def describe():
        return "Crop results to geometry"

    @staticmethod
    def order():
        return comparitor.TYPE_ADJUST

    def make_view(self, parent):
        self._view = CropToGeometryView(parent, self)
        return self._view

    @property
    def name(self):
        return "Crop to geometry"
        
    @property
    def settings_string(self):
        if self._filename is None:
            return "No geometry"
        out = "Geometry file: " + funcs.string_ellipse(self._filename, 40)
        if self._epsg is not None:
            out += " @ " + str(self._epsg)
        return out

    def config(self):
        return {"resize" : True}

    def to_dict(self):
        return {"filename" : self._filename,
            "epsg" : self._epsg}

    def from_dict(self, data):
        filename = data["filename"]
        if filename is not None:
            self.load(filename)
            if data["epsg"] is not None:
                self._epsg = int(data["epsg"])
            else:
                self._epsg = None

    def run(self, projector):
        """Returns a geometry object (typically from `shapely`).

        :param projector: Optional `predictors._TYPE_COORD_PROJ` task to use
          to project the geometry, if an explicit EPSG code has not been set.

        :return: None if no geometry.
        """
        if self.epsg is not None:
            return self.geometry(self.epsg)
        if projector is not None:
            return self._proj_geo(projector)
        return self.geometry()

    def make_tasks(self):
        return [self.Task(self)]
        
    class Task(comparitor.AdjustTask):
        def __init__(self, parent):
            self._parent = parent
        
        @staticmethod
        def assemble_sizes(grids):
            out = dict()
            for grid in grids:
                key = (grid.xsize, grid.ysize, grid.xoffset % grid.xsize,
                    grid.yoffset % grid.ysize)
                if key not in out:
                    out[key] = list()
                out[key].append(grid)
            return out

        @staticmethod
        def to_list(grids):
            try:
                return list(iter(grids))
            except:
                return [grids]

        def __call__(self, projector, grid_prediction):
            geo = self._parent.run(projector)
            if geo is None:
                return grid_prediction
            grid_prediction = self.to_list(grid_prediction)
            out = []
            for ((xsize, ysize, xoffset, yoffset), preds) in self.assemble_sizes(grid_prediction).items():
                grid = open_cp.data.Grid(xsize, ysize, xoffset, yoffset)
                masked_grid = open_cp.geometry.mask_grid_by_intersection(geo, grid)
                for pred in preds:
                    new_pred = pred.new_extent(
                        xoffset=masked_grid.xoffset, yoffset=masked_grid.yoffset,
                        xextent=masked_grid.xextent, yextent=masked_grid.yextent)
                    new_pred.mask_with(masked_grid)
                    out.append(new_pred)
            if len(grid_prediction) == 1:
                return out[0]
            return out

    def _proj_geo(self, proj):
        if self._frame is not None:
            try:
                # Project back to lon/lat
                frame = self._frame.to_crs({"init": "epsg:4326"})
                geo = frame.unary_union
                return shapely.ops.transform(lambda x,y,z=None : proj(x,y), geo)
            except:
                _logger.exception("While trying to return merged geometry")
        return None

    @property
    def filename(self):
        """The filename of the geometry."""
        return self._filename

    @property
    def epsg(self):
        """The selected output epsg code, or None."""
        return self._epsg

    @epsg.setter
    def epsg(self, value):
        self._epsg = value

    @property
    def crs(self):
        """The CRS dictionary, or `None`"""
        if self._frame is None:
            return None
        crs = self._frame.crs
        if crs is None or len(crs) == 0:
            return None
        return crs

    @property
    def guessed_crs(self):
        """True/False: Did we guess that the input was longitude / latitude?"""
        return self._guessed

    @property
    def error(self):
        """The last error message, or `None` is all is okay."""
        return self._error

    def load(self, filename):
        _logger.debug("Attempting to load geometry file '%s'", filename)
        self._filename = filename
        self._frame = None
        self._error = None
        try:
            self._frame = gpd.read_file(filename)
            _logger.debug("Loaded successfully.  crs is '%s'", self._frame.crs)
            if len(self._frame.crs) == 0:
                self._guessed = True
                self._frame.crs = {"init": "epsg:4326"}
            else:
                self._guessed = False
        except Exception as ex:
            self._error = _text["fail"].format(type(ex), ex)
            self._filename = None
            self._frame = None

    def geometry(self, epsg=None):
        """The geometry, or `None`
        
        :param epsg: If not `None` then first transform to this EPSG
          projection.
        """
        if self._frame is not None:
            try:
                if epsg is not None:
                    frame = self._frame.to_crs({"init": "epsg:{}".format(epsg)})
                else:
                    frame = self._frame
                return frame.unary_union
            except:
                _logger.exception("While trying to return merged geometry")
        return None

    def dataset_coords(self):
        """If possible, the x/y coordinates of the input data.  Or `None`"""
        return self._model.analysis_tools_model.projected_coords()

    def chosen_projector(self):
        """Using the main model, finds the 1st selected projector (if possible)
        and returns a function object which sends pairs `(x,y)` to their
        projection."""
        proj = self._model.analysis_tools_model.coordinate_projector()
        if proj is None:
            return None
        return proj.make_tasks()[0]

    def projected_geometry(self):
        """Return the geometry projected in the best way we can:
          - If an EPSG code is set, us it
          - If there is a "projector task" set in the main model, use it.
          - Return the unprojected geometry.
        """
        return self.run(self.chosen_projector())


class CropToGeometryView(tk.Frame):
    def __init__(self, parent, model):
        super().__init__(parent)
        self._model = model
        util.stretchy_rows_cols(self, [3], [0])
        self._text = richtext.RichText(self, height=12, scroll="v")
        self._text.grid(sticky=tk.NSEW, row=0, column=0)
        if gpd is None:
            self._error_case()
            return

        self._text.add_text(_text["main"])
        
        subframe = ttk.Frame(self)
        subframe.grid(row=1, column=0, padx=2, pady=3, sticky=tk.NW)
        ttk.Button(subframe, text=_text["load"], command=self._load).grid(row=0, column=0, padx=2)
        self._filename_label = ttk.Label(subframe)
        self._filename_label.grid(row=0, column=1, padx=2)
        tooltips.ToolTipYellow(self._filename_label, _text["filett"])

        subframe = ttk.Frame(self)
        subframe.grid(row=2, column=0, padx=2, pady=3, sticky=tk.NW)
        self._input_crs_label = ttk.Label(subframe, text=_text["icrs"].format(""))
        self._input_crs_label.grid(row=0, column=0, padx=2)
        self._input_crs_lavel_tt=tooltips.ToolTipYellow(self._input_crs_label, _text["icrstt"])
        label = ttk.Label(subframe, text=_text["newcrs"])
        label.grid(row=0, column=1, padx=2)
        self._output_crs_entry_var = tk.StringVar()
        self._output_crs_entry = ttk.Entry(subframe, textvariable=self._output_crs_entry_var)
        self._output_crs_entry.grid(row=0, column=2, padx=2)
        util.IntValidator(self._output_crs_entry, self._output_crs_entry_var, callback=self._new_epsg, allow_empty=True)
        tooltips.ToolTipYellow(label, _text["newcrstt"])
        href = util.HREF(subframe, text=_text["epsg_url_text"], url=_text["epsg_url"])
        href.grid(row=0, column=3, padx=2)
        tooltips.ToolTipYellow(href, _text["epsg_url"])

        subframe = ttk.Frame(self)
        subframe.grid(row=3, column=0, padx=2, pady=3, sticky=tk.NSEW)
        util.stretchy_rows_cols(subframe, [0], [0, 1])
        self._preview_frame = ttk.LabelFrame(subframe, text=_text["preview"])
        self._preview_frame.grid(row=0, column=0, sticky=tk.NSEW, padx=2)
        util.stretchy_rows_cols(self._preview_frame, [0], [0])
        self._preview_canvas = mtp.CanvasFigure(self._preview_frame)
        self._preview_canvas.grid(sticky=tk.NSEW, padx=1, pady=1)
        self._preview_frame_tt = tooltips.ToolTipYellow(self._preview_frame, _text["previewnonett"])

        with_coords_frame = ttk.LabelFrame(subframe, text=_text["preview2"])
        with_coords_frame.grid(row=0, column=1, sticky=tk.NSEW, padx=2)
        util.stretchy_rows_cols(with_coords_frame, [0], [0])
        self._with_coords_canvas = mtp.CanvasFigure(with_coords_frame)
        self._with_coords_canvas.grid(sticky=tk.NSEW, padx=1, pady=1)
        self._with_coords_frame_tt = tooltips.ToolTipYellow(with_coords_frame, _text["previewnonett"])

        self._update()

    def _update(self):
        if self._model.error is not None:
            self._filename_label["text"] = self._model.error
        elif self._model.filename is not None:
            self._filename_label["text"] = funcs.string_ellipse(self._model.filename, 80)
            self._loaded()
            return
        else:
            self._filename_label["text"] = ""

        self._set_input_crs("")
        self._preview_canvas.set_blank()
        self._with_coords_canvas.set_blank()
        self._input_crs_lavel_tt.text = _text["icrstt"]
        self._output_crs_entry_var.set("")
        self._preview_frame_tt.text = _text["previewnonett"]
        self._with_coords_frame_tt.text = _text["previewnonett"]

    def _loaded(self):
        self._input_crs_lavel_tt.text = _text["icrstt"]
        crs = self._model.crs
        if self._model.guessed_crs:
            crs = _text["guesscrs"]
            self._input_crs_lavel_tt.text = _text["icrstt1"]
        self._set_input_crs(crs)

        if self._model.epsg is None:
            self._output_crs_entry_var.set("")
        else:
            self._output_crs_entry_var.set(self._model.epsg)

        self._plot_preview()
        self._plot_with_points()

    def _plot_preview(self):
        geo = self._model.geometry()
        if geo is not None:
            def task():
                xmin, ymin, xmax, ymax = geo.bounds
                xdelta = (xmax - xmin) / 100 * 2
                ydelta = (ymax - ymin) / 100 * 2
                fig = mtp.new_figure(size=(8,8))
                ax = fig.add_subplot(1,1,1)
                self._try_add_patch(ax, geo)
                ax.set(xlim=[xmin-xdelta, xmax+xdelta], ylim=[ymin-ydelta, ymax+ydelta])
                ax.set_aspect(1)
                fig.set_tight_layout("tight")
                return fig
            self._preview_canvas.set_figure_task(task)
            self._preview_frame_tt.text = _text["previewtt"]
        else:
            self._preview_canvas.set_blank()
            self._preview_frame_tt.text = _text["previewnonett"]

    def _plot_with_points(self):
        geo = self._model.projected_geometry()
        coords = self._model.dataset_coords()
        if coords is not None and geo is not None:
            def task():
                fig = mtp.new_figure(size=(8,8))
                ax = fig.add_subplot(1,1,1)
                self._try_add_patch(ax, geo)
                ax.scatter(coords[0], coords[1], marker="+", color="black", alpha=0.5)
                ax.set_aspect(1)
                fig.set_tight_layout("tight")
                return fig
            self._with_coords_canvas.set_figure_task(task)
            self._with_coords_frame_tt.text = _text["previewtt2"]
        else:
            self._with_coords_canvas.set_blank()
            self._with_coords_frame_tt.text = _text["previewnonett"]

    def _try_add_patch(self, ax, geo):
        try:
            ax.add_patch(descartes.PolygonPatch(geo, fc="none", ec="Black"))
        except:
            _logger.exception("While invokling descartes to general polygon patch")

    def _load(self):
        filename = util.ask_open_filename(filetypes=[("GeoJSON", "*.geojson"),
            ("Shape file", "*.shp"), ("Any file", "*.*")])
        if filename is not None:
            self._model.load(filename)
            self._update()

    def _set_input_crs(self, value):
        self._input_crs_label["text"] = _text["icrs"].format(value)

    def _new_epsg(self):
        epsg = self._output_crs_entry_var.get()
        if epsg is None or epsg == "":
            self._model.epsg = None
        else:
            self._model.epsg = int(epsg)
        self._update()

    def _error_case(self):
        self._text.add_text(_text["nogpd"])


def test(root):
    from . import predictor
    ll = CropToGeometry(predictor.test_model())
    predictor.test_harness(ll, root)
    