#!/usr/bin/python2

from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image
from google.protobuf import text_format

from os import path, mkdir
import caffe
import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
        "-a", "--animate", default=0, type=int,
        help="animate a zoom for this many frames (default 0)"
)
parser.add_argument(
        "-s", "--scale", default=0.05, type=float,
        help="zoom speed (use with --animate)"
)
parser.add_argument(
        "-l", "--layer", default="inception_4d/pool",
        help="the layer to output from"
)
parser.add_argument(
        "-m", "--model", default="bvlc_googlenet",
        help="path to folder, .caffenet or name of model"
)
parser.add_argument(
        "--force-backward", action="store_true",
        help="patch model file for gradient support (force_backward=True)"
)
parser.add_argument(
        "--show", action="store_true",
        help="open a window with the result"
)
parser.add_argument(
        "image", nargs="+",
        help="images to process"
)
args = parser.parse_args()

def showarray(a):
    a = np.uint8(np.clip(a, 0, 255))
    PIL.Image.fromarray(a).show()

if path.isdir(args.model):
    net_fn      = path.join(args.model, "deploy.prototxt")
    param_fn    = path.join(args.model, "{}.caffeemodel".format(path.basename(args.model)))
elif path.isfile(args.model):
    net_fn      = path.join(path.dirname(args.model), "deploy.prototxt")
    param_fn    = args.model
else:
    modeldir    = path.join("/opt/caffe/models", args.model)
    net_fn      = path.join(modeldir, "deploy.prototxt")
    param_fn    = path.join(modeldir, "{}.caffemodel".format(args.model))

if args.force_backward:
    model = caffe.io.caffe_pb2.NetParameter()
    text_format.Merge(open(net_fn).read(), model)
    model.force_backward = True
    open("tmp.prototxt", "w").write(str(model))
    net_fn = "tmp.prototxt"

net = caffe.Classifier(net_fn, param_fn,
                       mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
                       channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']
def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

def make_step(net, step_size=1.5, end='inception_4c/output', jitter=32, clip=True):
    '''Basic gradient ascent step.'''

    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    dst = net.blobs[end]

    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift

    net.forward(end=end)
    dst.diff[:] = dst.data  # specify the optimization objective
    net.backward(start=end)
    g = src.diff[0]
    # apply normalized ascent step to the input image
    src.data[:] += step_size/np.abs(g).mean() * g

    src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image

    if clip:
        bias = net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias)

def deepdream(net, base_img, iter_n=10, octave_n=4, octave_scale=1.4, end='inception_4c/output', clip=True, **step_params):
    # prepare base images for all octaves
    octaves = [preprocess(net, base_img)]
    for i in xrange(octave_n-1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))

    src = net.blobs['data']
    detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)

        src.reshape(1,3,h,w) # resize the network's input image size
        src.data[0] = octave_base+detail
        for i in xrange(iter_n):
            make_step(net, end=end, clip=clip, **step_params)

            # visualization
            vis = deprocess(net, src.data[0])
            if not clip: # adjust image contrast if clipping is disabled
                vis = vis*(255.0/np.percentile(vis, 99.98))
            #showarray(vis)
            print octave, i, end, vis.shape

        # extract details produced on the current octave
        detail = src.data[0]-octave_base
    # returning the resulting image
    return deprocess(net, src.data[0])

# 4a/1x1: square, edge   or 4a/3x3 ??
# 4a/1x1: circles ??
# 4d/pool: dogs, birds, weasels
# 5b/5x5_reduce: insects
for img in args.image:
    frame = np.float32(PIL.Image.open(img))

    print args.animate, args.layer

    if args.animate > 1:
        h, w = frame.shape[:2]
        s = args.scale # scale coefficient
        try: mkdir("%s-%s"%(img, args.layer.replace("/","-")))
        except OSError: pass
        for i in xrange(args.animate):
            frame = deepdream(net, frame, end=args.layer)
            res = PIL.Image.fromarray(np.uint8(frame))
            res.save("%s-%s/%04d.jpg"%(img, args.layer.replace("/","-"), i))
            if args.show: res.show()
            frame = nd.affine_transform(frame, [1-s,1-s,1], [h*s/2,w*s/2,0], order=1)
    else:
        frame = deepdream(net, frame, end=args.layer)
        res = PIL.Image.fromarray(np.uint8(frame))
        res.save("%s-%s.jpg"%(img, args.layer.replace("/","-")))
        if args.show:
            res.show()
