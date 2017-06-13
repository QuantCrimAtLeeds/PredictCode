import base64 as _b64
import PIL.Image as _image
import io as _io

close_icon = 'iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAQAAAC1+jfqAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAAAmJLR0QAAKqNIzIAAAAJcEhZcwAADdcAAA3XAUIom3gAAAAHdElNRQfhBg0PLB09jQ+wAAAAnklEQVQoz42RuxHCMBBEnzAdkChSKIpxB3TgTB3gHi4g8zgjVEQzEJE5oQYR2DoxjH8v0Ehzq7m9PUPGc8FhgMSbO09+MAQEp2+HEDCl3FPzT02fJWGmPErC2FtYQvDQau+IBcAS1csVOtVbInY6M92xeGWg4QY0DGW+A5tstNhhsoxZ6b98E3zFhzMnXkBSQZqCMjx2Rb26rJLCwrq/pi4oZuXrgZ8AAAAldEVYdGRhdGU6Y3JlYXRlADIwMTctMDYtMTNUMTU6NDQ6MjkrMDI6MDBotB6VAAAAJXRFWHRkYXRlOm1vZGlmeQAyMDE3LTA2LTEzVDE1OjQ0OjI5KzAyOjAwGemmKQAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAAASUVORK5CYII='
edit_icon = 'iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAQAAAC1+jfqAAAABGdBTUEAALGPC/xhBQAAACBjSFJNAAB6JgAAgIQAAPoAAACA6AAAdTAAAOpgAAA6mAAAF3CculE8AAAAAmJLR0QAAKqNIzIAAAAJcEhZcwAADdcAAA3XAUIom3gAAAAHdElNRQfhBg0PLS/sQW9xAAAA5klEQVQoz23Ruy6DAQCG4ef/K44llEgMIhaJQdgExSASdRhrkLgXLkBMHSy6SURMDh260SBiaLgAC5FIdKgqo6GRUv+3Pu/2BaI3Zl23kkw0Jx3LatZnO4zkLWV5e8oSQQSn3es1Km/RUyyCi/aNqJrUJRP+408vyHozaNdD0MBVF3KWDJhwpEDTH/5QkJMyr63G9SApreLKuWWzdf4JanztzIoZ7XUmRNKGdzdOrZrW8ZsJtdqRUHFizZT4XyYwZ8irBSVxnY1MzKYTz6pSOGxkQuN6tOg37OA/E3h0p+jSra+oY78BqEZCMVdOKtgAAAAldEVYdGRhdGU6Y3JlYXRlADIwMTctMDYtMTNUMTU6NDU6NDcrMDI6MDARJgdxAAAAJXRFWHRkYXRlOm1vZGlmeQAyMDE3LTA2LTEzVDE1OjQ1OjQ3KzAyOjAwYHu/zQAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAAAASUVORK5CYII='

def to_image(string):
    b = _b64.b64decode(string)
    return _image.open(_io.BytesIO(b))

def alpha_to_bw(image):
    image = image.convert("RGBA")
    
    pixels = image.load()
    for y in range(image.height):
        for x in range(image.width):
            r,g,b,a = pixels[x,y]
            if a > 0:
                a = 255 - a
                pixels[x,y] = (a,a,a,255)
            else:
                pixels[x,y] = (255,255,255,255)

    return image.convert("L")

def invert_alpha(image):
    image = image.convert("RGBA")

    pixels = image.load()
    for y in range(image.height):
        for x in range(image.width):
            r,g,b,a = pixels[x,y]
            pixels[x,y] = r,g,b,255-a

    return image

close_icon = to_image(close_icon).convert("RGBA")
edit_icon = to_image(edit_icon).convert("RGBA")
