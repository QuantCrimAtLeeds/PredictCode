"""
load_network_view
~~~~~~~~~~~~~~~~~

"""

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox 
from . import util
from .. import funcs
import open_cp.gui.resources as resources
import open_cp.gui.tk.projectors_view as projectors_view
import open_cp.gui.predictors.geo_clip as geo_clip
import PIL.ImageTk as ImageTk
from . import mtp
from . import tooltips
from .. import load_network_model
from .. import locator
NetworkType = load_network_model.NetworkModel.NetworkType

_text = {
    "title" : "Load Network Geometry",
    "what" : "Load a geometry file which specifies the road/street network.",
    "okay" : "Accept",
    "cancel" : "Cancel",
    "net" : "Network",
    "load_file" : "Load geometry file",
    "current_file" : "Current file: {}",
    "none" :  "None loaded",
    "preview" : "Network preview",
    "preview_tt" : "A preview of the graph as constructed directly from the loaded geometry.",
    "preview_proj" : "Projected Network preview",
    "preview_proj_tt" : ("A preview of the graph as constructed from the projected geometry, using either "
            + "the explicitly set EPSG code, or the (first) projection task chosen from the collection of "
            + "'Predictors'.  Overlayed with the actual data.  Check that these agree!"),
    "os_type" : "Ordnance Survey style",
    "os_type_tt" : ("Assume the network geometry is similar to the UK Ordnance Survey OpenRoads data.  "
            + "This assumes that paths intersect only at their end-points.  This allows for accurate "
            + "representation of over/under passes, for example."),
    "tl_type" : "TIGER/Lines (tm) style",
    "tl_type_tt" : ("Assume the network geometry is similar to the USA TIGER/Lines (tm) data.  "
            + "This assumes that every path is composed of line segements.  These line segments are "
            + "considered individually.  If the coordinates of different start/end points of the line "
            + "segment are very close, they are merged.  This might lead us to identify over/under "
            + "passes, for example, but it correctly detects road junctions etc. in this dataset." ),
    "loading" : "Loading network...",
    "remove_tt" : "Remove current network",
    "crop" : "Crop network to an area",
    
}

def alert(message):
    tkinter.messagebox.showerror("Error", message)

class FurtherWait(util.ModalWaitWindow):
    def __init__(self, parent):
        super().__init__(parent, _text["loading"])
    
    def run(self, task):
        def done(input=None):
            self.destroy()
        locator.get("pool").submit(task, done)
        self.wait_window(self)
    
    
class LoadNetworkView(util.ModalWindow):
    def __init__(self, parent, controller):
        self._parent = parent
        self.controller = controller
        super().__init__(parent, title=_text["title"], resize="hw")
        util.centre_window_percentage(self, 75, 66)
        util.stretchy_rows_cols(self, [10], [0])
        self.refresh()
        
    @property
    def model(self):
        return self.controller.model
        
    def add_widgets(self):
        self._close_icon = ImageTk.PhotoImage(resources.close_icon)

        ttk.Label(self, text=_text["what"]).grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        
        frame = ttk.LabelFrame(self, text=_text["net"])
        frame.grid(row=10, column=0, sticky=tk.NSEW, padx=2, pady=2)
        subframe = ttk.Frame(frame)
        subframe.grid(row=0, column=0, sticky=tk.NSEW)
        subsubframe = ttk.Frame(subframe)
        subsubframe.grid(row=0, column=0, sticky=tk.W)
        ttk.Button(subsubframe, text=_text["load_file"], command=self._load_file).grid(row=0, column=0, padx=2, pady=2)
        b = ttk.Button(subsubframe, image=self._close_icon, command=self._remove_file)
        b.grid(row=0, column=1, padx=2, pady=2)
        tooltips.ToolTipYellow(b, _text["remove_tt"])
        self._filename_label = ttk.Label(subsubframe)
        self._filename_label.grid(row=0, column=2, padx=2, pady=2)
        self._projector_widget = projectors_view.GeoFrameProjectorWidget(subframe, self.model.geoframe_projector, self._new_epsg)
        self._projector_widget.grid(row=1, column=0, sticky=tk.W)
        
        subframe = ttk.Frame(frame)
        subframe.grid(row=1, column=0, sticky=tk.NSEW)
        util.stretchy_rows_cols(frame, [1], [0])
        
        subsubframe = ttk.Frame(subframe)
        subsubframe.grid(row=0, column=0, sticky=tk.NSEW)
        self._network_type = tk.IntVar()
        rb = ttk.Radiobutton(subsubframe, text=_text["os_type"], value=NetworkType.ORDNANCE.value, variable=self._network_type, command=self._new_network_type)
        rb.grid(row=0, column=0, sticky=tk.W)
        tooltips.ToolTipYellow(rb, _text["os_type_tt"])
        rb = ttk.Radiobutton(subsubframe, text=_text["tl_type"], value=NetworkType.TIGER_LINES.value, variable=self._network_type, command=self._new_network_type)
        rb.grid(row=1, column=0, sticky=tk.W)
        tooltips.ToolTipYellow(rb, _text["tl_type_tt"])
        
        preview_frame = ttk.LabelFrame(subframe, text=_text["preview"])
        preview_frame.grid(row=0, column=1, sticky=tk.NSEW)
        util.stretchy_rows_cols(preview_frame, [0], [0])
        util.stretchy_rows_cols(subframe, [0], [1, 2])
        self._preview_canvas = mtp.CanvasFigure(preview_frame)
        self._preview_canvas.grid(sticky=tk.NSEW, padx=1, pady=1)
        tooltips.ToolTipYellow(self._preview_canvas, _text["preview_tt"])
        preview_frame = ttk.LabelFrame(subframe, text=_text["preview_proj"])
        preview_frame.grid(row=0, column=2, sticky=tk.NSEW)
        util.stretchy_rows_cols(preview_frame, [0], [0])
        self._preview_projected_canvas = mtp.CanvasFigure(preview_frame)
        self._preview_projected_canvas.grid(sticky=tk.NSEW, padx=1, pady=1)
        tooltips.ToolTipYellow(self._preview_projected_canvas, _text["preview_proj_tt"])

        #frame = ttk.LabelFrame(self, text=_text["crop"])
        #frame.grid(row=20, column=0, sticky=tk.NSEW, padx=2, pady=2)
        #self.model.crop_to_geometry.make_view(frame, "inline").grid(sticky=tk.NSEW)

        frame = ttk.Frame(self)
        frame.grid(row=100, column=0, sticky=tk.EW, pady=2)
        ttk.Button(frame, text=_text["okay"], command=self.okay_pressed).grid(row=0, column=0, padx=5)
        ttk.Button(frame, text=_text["cancel"], command=self.cancel).grid(row=0, column=1, padx=5)
        
    def refresh(self):
        if self.model.filename is not None:
            name = funcs.string_ellipse(self.model.filename, 80)
        else:
            name = _text["none"]
        self._filename_label["text"] = _text["current_file"].format(name)
        self._network_type.set(self.model.network_type.value)
        self._plot_preview()
        self._projector_widget.update()
        self.refresh_projected()
        
    def refresh_projected(self):
        coords = self.model.geoframe_projector.projected_dataset_coords()
        if self.model.projected_graph is None or coords is None:
            self._preview_projected_canvas.set_blank()
        else:
            def task():
                fig = mtp.new_figure(size=(8,8))
                ax = fig.add_subplot(1,1,1)
                lc = mtp.matplotlib.collections.LineCollection(self.model.projected_graph.as_lines(),
                        color="black", linewidth=0.5, zorder=1)
                ax.add_collection(lc)
                ax.scatter(coords[0], coords[1], marker="+", color="blue", alpha=0.5, zorder=2)
                ax.set_aspect(1)
                fig.set_tight_layout("tight")
                return fig
            self._preview_projected_canvas.set_figure_task(task)

    def _new_epsg(self):
        self.controller.new_epsg()
        
    def _load_file(self):
        filename = util.ask_open_filename(filetypes=[("Shape file", "*.shp"),
                ("GeoJSON", "*.geojson"), ("Any file", "*.*")])
        if filename is not None:
            self.controller.load(filename)
        
    def _remove_file(self):
        self.controller.remove()
        
    def _plot_preview(self):
        if self.model.graph is None:
            self._preview_canvas.set_blank()
        else:
            def task():
                fig = mtp.new_figure(size=(8,8))
                ax = fig.add_subplot(1,1,1)
                lc = mtp.matplotlib.collections.LineCollection(self.model.graph.as_lines(), color="black", linewidth=0.5)
                ax.add_collection(lc)
                xmin, ymin, xmax, ymax = self.model.graph.bounds
                xdelta = (xmax - xmin) / 100 * 2
                ydelta = (ymax - ymin) / 100 * 2
                ax.set(xlim=[xmin-xdelta, xmax+xdelta], ylim=[ymin-ydelta, ymax+ydelta])
                ax.set_aspect(1)
                fig.set_tight_layout("tight")
                return fig
            self._preview_canvas.set_figure_task(task)
        
    def _new_network_type(self):
        choice = NetworkType( int(self._network_type.get()) )
        self.model.network_type = choice
        self.controller.reload()
    
    def okay_pressed(self):
        self.okay = True
        super().cancel()
        
    def cancel(self):
        self.okay = False
        super().cancel()
        