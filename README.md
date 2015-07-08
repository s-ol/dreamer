dreamer
=======

An improved Deep Dreaming script with a proper CLI.

Original script from [pastebin](http://pastebin.com/1ePNC89A), I mainly added an `argparse` CLI and removed the ipython parts.

    usage: dream.py [-h] [-a ANIMATE] [-s SCALE] [-l LAYER] [-m MODEL]
                    [--force-backward] [--show]
                    image [image ...]
    
    positional arguments:
      image                 images to process
    
    optional arguments:
      -h, --help            show this help message and exit
      -a ANIMATE, --animate ANIMATE
                            animate a zoom for this many frames (default 0)
      -s SCALE, --scale SCALE
                            zoom speed (use with --animate)
      -l LAYER, --layer LAYER
                            the layer to output from
      -m MODEL, --model MODEL
                            path to folder, .caffenet or name of model
      --force-backward      patch model file for gradient support
                            (force_backward=True)
      --show                open a window with the result

