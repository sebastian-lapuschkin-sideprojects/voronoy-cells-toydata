# Voronoy Cell Toy Data Generator
This is some code which has been written to illustrate an idea in some XAI research context.
The idea is to draw some simple images with ground truth masks for localization testing.
Voronoy regions seem like a simple enough yet at the same time interesting enough idea.
The provided implementations draws parameterizable images of colored regions and produces ground truth labels for region counts, each samples true class, and a binary mask for the true class.

## Requirements
This tool has been developed with Python 3.8 and the following packages:
```
tqdm==4.56.0
imageio==2.9.0
numpy==1.20.1
matplotlib==3.3.4
scipy==1.6.0
scikit-learn==0.24.1
opencv-python==4.5.2.52
```
Required packages can be installed via `pip install -r requirements.txt`

## How to Use
```
usage: main.py [-h] [--random_seed RANDOM_SEED] [--size SIZE] [--number NUMBER] [--min_centroids MIN_CENTROIDS] [--max_centroids MAX_CENTROIDS] [--distance_metric DISTANCE_METRIC] [--class_colors [CLASS_COLORS [CLASS_COLORS ...]]]
               [--class_color_deviation CLASS_COLOR_DEVIATION] [--bg_colors [BG_COLORS [BG_COLORS ...]]] [--bg_color_deviation BG_COLOR_DEVIATION] [--draw_markers] [--marker_color MARKER_COLOR] [--draw_borders DRAW_BORDERS]
               [--line_dilation_iterations LINE_DILATION_ITERATIONS] [--line_erosion_iterations LINE_EROSION_ITERATIONS] [--show] [--output OUTPUT]

Generate some toy images of colored voronoy cells. Produces ground truth class labels, ground truth true class region masks and region count labels. Also outputs its parameterization for reproducibility.

optional arguments:
  -h, --help            show this help message and exit
  --random_seed RANDOM_SEED, -rs RANDOM_SEED
                        random seed for random number generation. int or hex code.
  --size SIZE, -s SIZE  canvas size for the images to be generated. we are assuming square images and are working in pixel coordinates.
  --number NUMBER, -n NUMBER
                        number of samples to be generated.
  --min_centroids MIN_CENTROIDS, -mnc MIN_CENTROIDS
                        minimum number of centroids to be scattered on the canvas.
  --max_centroids MAX_CENTROIDS, -mxc MAX_CENTROIDS
                        maximum number of centroids to be scattered on the canvas.
  --distance_metric DISTANCE_METRIC, -d DISTANCE_METRIC
                        the distance measure to use for knn. supports naming choices compatible for (parameter-free) sklearn.neighbors.DistanceMetric.
  --class_colors [CLASS_COLORS [CLASS_COLORS ...]], -cc [CLASS_COLORS [CLASS_COLORS ...]]
                        the colors for labelled classes as rgb hex strings or valid paths to texture images. multiple namings possible. each color adds a class.
  --class_color_deviation CLASS_COLOR_DEVIATION, -ccd CLASS_COLOR_DEVIATION
                        the standard deviation (in rgb color steps) for possible deviations in class color.
  --bg_colors [BG_COLORS [BG_COLORS ...]], -bc [BG_COLORS [BG_COLORS ...]]
                        the colors for "background" tiles or valid paths to texture images.. each added color adds to the variation.
  --bg_color_deviation BG_COLOR_DEVIATION, -bcd BG_COLOR_DEVIATION
                        the standard deviation (in rgb color steps) for possible deviations in background color.
  --draw_markers, -dm   set to draw (single pixel) markers for centroids.
  --marker_color MARKER_COLOR, -mc MARKER_COLOR
                        the color of centroid markers. "class" is a darker version of the class color. otherwise, rgb hex codes specify special color choices, e.g. 0x000000 is black.
  --draw_borders DRAW_BORDERS, -db DRAW_BORDERS
                        how to draw draw dividing lines between regions? Options: "none", "color:<hexcode>:flat" (e.g. color:0x000000:flat for black lines), "color:<hexcode>:gaussian:stdev" to draw a gaussian-weighted blur around the line, "color:<hexcode>:linear:n_pixels" to draw a linearly-weighted blur around the line
  --line_dilation_iterations LINE_DILATION_ITERATIONS, -ldi LINE_DILATION_ITERATIONS
                        how often to binary dilate region boundaries? dilation is applied before erosion.
  --line_erosion_iterations LINE_EROSION_ITERATIONS, -lei LINE_EROSION_ITERATIONS
                        how often to binary erode region boundaries? erosion is applied after dilation.
  --show                show generated images?
  --output OUTPUT, -o OUTPUT
                        output directory for outputting data and labels.
```

## Example Data
The following call
```
python main.py -rs 0xc0ffee  -s 224  -n 6  -mnc 5  -mxc 10  -d chebyshev  -cc 0xff0000 0x00ff00 0x0000ff  -ccd 40  -bc 0xeeeeee  -bcd 7  -mc class  -db color:0xffffff:gaussian:1.5  -ldi 3 -lei 1  --show  -o ./output_224
```
will generate some data containing `6` images of size `244x244` with `5` to `10` centroids each which are placed according to a random process initialized with the random seed `0xc0ffee`.
The data will consist of three different classes, identified by the given class colors `0xff0000`, `0x00ff00`, `0x0000ff`, of which the color value (within a range of [0,255] per rgb color channel) might deviate with a standard deviation of `40` per class region.
One region will be picked at random to be the class region, while the others will be considered as background and colored with `0xeeeeee` +- some standard deviation of `7`.
The specification of the marker color as `class` via `-mc` would draw the centroid location in a slightly darker hue than the region color, but has no effect here, since `-dm` has not been set. Due to the setting of `-db color:0xffffff:gaussian:1.5` lines between colored regions will be drawn with a base line color of `0xffffff`. Note that `color` currently is the only base parameter for borders.
After finding the edges between voronoy regions using a sobel operator, line pixels are post processed with `3` iterations of binary dilation and `1` iteration of binary erosion (where dilation iterations are always applied before erosion iterations).
Te addendum `gaussian:1.5` specifies an application of gaussian blur with sigma `1.5` on the lines.
The blur effect will be restricted to regions in close proximity of the border pixels, with an approximate range of `3*sigma`, i.e. covering approx. 99.75% of a single gaussian's range.
This will maintain the original sharpness of the image near the center of the regions.
Alpha-blending avoids the preference to high-density border regions
`--show` specifies that each generated sample is shown to the user. Generated data is then written to the specified output (`-o`) location, which contains the generated images, with one corresponding ground truth localization mask each, a file `labels.txt` containing the class labels and number of regions per sample, and a file `args.txt` ensuring repeatability of the function call.

The following images, with ground truth masks below, have been generated:

![output_224/0.png](output_224/0.png) ![output_224/1.png](output_224/1.png) ![output_224/2.png](output_224/2.png)

![output_224/3.png](output_224/3.png) ![output_224/4.png](output_224/4.png) ![output_224/5.png](output_224/5.png)

![output_224/0_gt.png](output_224/0_gt.png) ![output_224/1_gt.png](output_224/1_gt.png) ![output_224/2_gt.png](output_224/2_gt.png)

![output_224/3_gt.png](output_224/3_gt.png) ![output_224/4_gt.png](output_224/4_gt.png) ![output_224/5_gt.png](output_224/5_gt.png)

The content of `labels.txt`, describing one sample per line, is

```
# image_id true_class num_regions
0 1 9
1 0 5
2 1 7
3 2 6
4 0 5
5 2 6

```

The final file generated by the script is `args.txt`, containing the complete configuration leading to the generation of above data:
```
--random_seed 0xc0ffee  --size 224  --number 6  --min_centroids 5  --max_centroids 10  --distance_metric chebyshev  --class_colors 0xff0000 0x00ff00 0x0000ff  --class_color_deviation 40  --bg_colors 0xeeeeee  --bg_color_deviation 7  --marker_color class  --draw_borders color:0xffffff:gaussian:1.5  --line_dilation_iterations 3  --line_erosion_iterations 1  --show   --output ./output_224
```

This allows for a replication of the previous results by simply calling
```
python main.py $(cat output_224/args.txt)
```

### Borders or no borders?

The parameterization in [`output_224_noborders`](output_224_noborders/args.txt) demonstrates how the `--draw_borders` with a gaussian standard deviation of `10` instead of `1.5` can be used to remove any high contrast edges. Alterantively, setting `--draw_borders color:0x222222:flat` would draw hard borders of almost black color, without any additional post processing.



### Region texturing.

As an alternative to painting regions in flat colors, texturing is also supported by specifying a (relative) path to an image, instead of a hex code for colorization. Additionally the parmeterization in [`output_224_textured`](output_224_textured/args.txt) demonstrates a further alternative for soft border parameterization via linear regression from the border center with a certain range, here `15`, by setting `--draw_borders with color:0x000000:linear:15`, based on euclidean distances to region centroids.

The following images, with ground truth masks below, have been generated (**TODO: random transformation support**):

![output_224_textured/0.png](output_224_textured/0.png) ![output_224_textured/1.png](output_224_textured/1.png) ![output_224_textured/2.png](output_224_textured/2.png)

![output_224_textured/3.png](output_224_textured/3.png) ![output_224_textured/4.png](output_224_textured/4.png) ![output_224_textured/5.png](output_224_textured/5.png)

![output_224_textured/0_gt.png](output_224_textured/0_gt.png) ![output_224_textured/1_gt.png](output_224_textured/1_gt.png) ![output_224_textured/2_gt.png](output_224_textured/2_gt.png)

![output_224_textured/3_gt.png](output_224_textured/3_gt.png) ![output_224_textured/4_gt.png](output_224_textured/4_gt.png) ![output_224_textured/5_gt.png](output_224_textured/5_gt.png)
