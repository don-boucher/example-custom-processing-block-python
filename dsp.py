import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import base64
import io


def smoothing(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode="same")
    return y_smooth


def generate_features(
    implementation_version,
    draw_graphs,
    raw_data,
    axes,
    sampling_freq,
    scale_axes,
    smooth,
):

    scale_axes = float(scale_axes)
    # features is a 1D array, reshape so we have a matrix with one raw per axis
    raw_data = raw_data.reshape(int(len(raw_data) / len(axes)), len(axes))

    features = []
    smoothed_graph = {}

    # split out the data from all axes
    for ax in range(0, len(axes)):
        X = []
        for ix in range(0, raw_data.shape[0]):
            X.append(raw_data[ix][ax])

        # X now contains only the current axis
        fx = np.array(X)

        # first scale the values
        fx = fx * scale_axes

        # if smoothing is enabled, do that
        if smooth:
            fx = smoothing(fx, 5)

        # we save bandwidth by only drawing graphs when needed
        if draw_graphs:
            smoothed_graph[axes[ax]] = list(fx)

        # we need to return a 1D array again, so flatten here again
        for f in fx:
            features.append(f)

    # draw the graph with time in the window on the Y axis, and the values on the X axes
    # note that the 'suggestedYMin/suggestedYMax' names are incorrect, they describe
    # the min/max of the X axis
    graphs = []
    if draw_graphs:
        graphs.append(
            {
                "name": "Smoothed",
                "X": smoothed_graph,
                "y": np.linspace(
                    0.0,
                    raw_data.shape[0] * (1 / sampling_freq) * 1000,
                    raw_data.shape[0] + 1,
                ).tolist(),
                "suggestedYMin": -20,
                "suggestedYMax": 20,
            }
        )

    # create a new image, and draw some text on it
    im = Image.new("RGB", (438, 146), (248, 86, 44))
    draw = ImageDraw.Draw(im)
    draw.text((10, 10), "Hello world!", fill=(255, 255, 255))

    # save the image to a buffer, and base64 encode the buffer
    with io.BytesIO() as buf:
        im.save(buf, format="png", bbox_inches="tight", pad_inches=0)
        buf.seek(0)
        image = base64.b64encode(buf.getvalue()).decode("ascii")

        # append as a new graph
        graphs.append(
            {
                "name": "Image from custom block",
                "image": image,
                "imageMimeType": "image/png",
                "type": "image",
            }
        )

    return {
        "features": features,
        "graphs": graphs,
        "output_config": {"type": "flat", "shape": {"width": len(features)}},
    }
