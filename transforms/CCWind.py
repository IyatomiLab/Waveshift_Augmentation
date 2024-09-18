# CENTER CROP WINDOW OF AN IMAGE
def CCWind(self, img, size):
    width, height = img.size
    left = (width - size[0]) / 2
    top = (height - size[1]) / 2
    right = (width + size[0]) / 2
    bottom = (height + size[1]) / 2
    return img.crop((left, top, right, bottom))
