# http://infohost.nmt.edu/tcc/help/pubs/tkinter/web/entry-validation.html
# Note that there are some typos in that...
#
# http://stupidpythonideas.blogspot.co.uk/2013/12/tkinter-validation.html
# http://www.tcl.tk/man/tcl8.4/TkCmd/entry.htm#M12

from tkinter import *

root = Tk()
sv = StringVar()

def callback(d, i, P, s, ss, v, vv, W):
    print("d='{}', i='{}', P='{}', s='{}', S='{}', v='{}', V='{}', W='{}'".format(d, i, P, s, ss, v, vv, W))
    return True

value = StringVar()
old_value = ""
def validate_float(val, why):
    if why == "focusin":
        global old_value
        old_value = value.get()
        print("Captured", old_value)
    elif why == "focusout":
        if val == "":
            print("Empty, so okay")
            return True
        try:
            float(val)
            print(val, "is okay, leaving")
        except:
            return False
    else:
        raise ValueError()
    return True

float_entry = None

def revalidate():
    global float_entry
    float_entry["validate"] = "focus"

def reset_float():
    global old_value
    value.set(old_value)
    print("not okay so reset to", old_value)
    global float_entry
    # Really?  really...
    float_entry.after_idle(revalidate)

cmd = root.register(callback)
Label(root, text="Entry text here:").grid(row=0, column=0)
e = Entry(root, textvariable=sv, validate="all", validatecommand=(cmd, "%d", "%i", "%P", "%s", "%S", "%v", "%V", "%W"))
e.grid(row=0, column=1)
Label(root, text="Other box:").grid(row=1, column=0)
cmd1 = root.register(validate_float)
cmd2 = root.register(reset_float)
e = Entry(root, textvariable=value, validate="focus", validatecommand=(cmd1, "%P", "%V"), invalidcommand=(cmd2,))
e.grid(row=1, column=1)
float_entry = e
root.mainloop()