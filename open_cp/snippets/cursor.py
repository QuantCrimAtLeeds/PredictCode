import tkinter as tk

root = tk.Tk()

windows_native = ["arrow", "center_ptr", "crosshair", "fleur", "ibeam", "icon",
    "sb_h_double_arrow", "sb_v_double_arrow", "watch", "xterm"]

windows_extra = ["no", "starting", "size", "size_ne_sw", "size_ns", "size_nw_se", "size_we", "uparrow", "wait"]

cross_plat = ["X_cursor", "arrow", "based_arrow_down", "based_arrow_up", "boat",
    "bogosity", "bottom_left_corner", "bottom_right_corner", "bottom_side",
    "bottom_tee", "box_spiral", "center_ptr", "circle", "clock", "coffee_mug",
    "cross", "cross_reverse", "crosshair", "diamond_cross", "dot", "dotbox",
    "double_arrow", "draft_large", "draft_small", "draped_box", "exchange", "fleur",
    "gobbler", "gumby", "hand1", "hand2", "heart", "icon", "iron_cross", "left_ptr",
    "left_side", "left_tee", "leftbutton", "ll_angle", "lr_angle", "man",
    "middlebutton", "mouse", "pencil", "pirate", "plus", "question_arrow",
    "right_ptr", "right_side", "right_tee", "rightbutton", "rtl_logo", "sailboat",
    "sb_down_arrow", "sb_h_double_arrow", "sb_left_arrow", "sb_right_arrow",
    "sb_up_arrow", "sb_v_double_arrow", "shuttle", "sizing", "spider", "spraycan",
    "star", "target", "tcross", "top_left_arrow", "top_left_corner",
    "top_right_corner", "top_side", "top_tee", "trek", "ul_angle", "umbrella",
    "ur_angle", "watch", "xterm"]

cursors = windows_native + windows_extra + cross_plat

cols = 10

row, col = 0, 0
for i, name in enumerate(cursors):
    la = tk.Label(root, text=name, cursor=name)
    la.grid(row=row, column=col)
    col += 1
    if col == cols:
        row, col = row+1, 0

root.mainloop()