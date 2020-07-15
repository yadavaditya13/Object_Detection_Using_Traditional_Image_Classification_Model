# importing required package for our package
import imutils


# lets create the sliding window generator
# we have three inputs to the function
# image for sliding which is output from image pyramid,
# step to determine the skip pixel value / step size
# and the final one is the window size

def sliding_window(image, step, ws):
    # we begin looping over the image just like any matrix
    # the ending pixel has '-' for window size x and y value
    # so that we won't exceed the boundary of the image pixels

    for y in range(0, image.shape[0] - ws[1], step):
        for x in range(0, image.shape[1] - ws[0], step):
            # yield keyword is used in place of return because our function
            # will be used as a generator
            yield x, y, image[y:y + ws[1], x:x + ws[0]]


# now it's time for image pyramid generator
def image_pyramid(image, scale=1.5, minSize=(224, 224)):
    # we will first yield the original image
    yield image

    # lets loop over the image pyramid
    while True:
        # calculate the dimensions of the next image
        size = int(image.shape[1] / scale)
        image = imutils.resize(image=image, width=size)

        # if the resized image does not meet minimum size then stop
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
        # yield the next image in the pyramid
        yield image