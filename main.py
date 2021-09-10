import os
import tqdm
import imageio
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import sobel
from scipy.ndimage import binary_erosion
from scipy.ndimage import binary_dilation
from sklearn.neighbors import DistanceMetric


def hex2rgb_arr(hexcode):
    rgb_int = int(hexcode,0)
    return np.array([(rgb_int >> 16) & 255, (rgb_int >> 8) & 255, rgb_int & 255], dtype=np.uint8)

parser = argparse.ArgumentParser(description='Generate some toy images of colored voronoy cells. Produces ground truth class labels, ground truth true class region masks and region count labels. Also outputs its parameterization for reproducibility.')
# data generation
parser.add_argument('--random_seed', '-rs', type=str, default='0xc0ffee', help='random seed for random number generation. int or hex code.')
parser.add_argument('--size', '-s', type=int, default=50, help='canvas size for the images to be generated. we are assuming square images and are working in pixel coordinates.')
parser.add_argument('--number','-n', type=int, default=10, help='number of samples to be generated.')
parser.add_argument('--min_centroids', '-mnc', type=int, default=5, help='minimum number of centroids to be scattered on the canvas.')
parser.add_argument('--max_centroids', '-mxc', type=int, default=10, help='maximum number of centroids to be scattered on the canvas.')
parser.add_argument('--distance_metric', '-d', type=str, default='euclidean', help='the distance measure to use for knn. supports naming choices compatible for (parameter-free) sklearn.neighbors.DistanceMetric.')
# drawing parameters
parser.add_argument('--class_colors', '-cc', type=str, nargs='*', default='0xff0000', help='the colors for labelled classes as rgb hex strings. multiple namings possible. each color adds a class.')
parser.add_argument('--class_color_deviation', '-ccd', type=int, default=10, help='the standard deviation (in rgb color steps) for possible deviations in class color.')
parser.add_argument('--bg_colors', '-bc', type=str, nargs='*', default='0xffffff', help='the colors for "background" tiles. each added color adds to the variation.')
parser.add_argument('--bg_color_deviation', '-bcd', type=int, default=10, help='the standard deviation (in rgb color steps) for possible deviations in background color.')
parser.add_argument('--draw_markers', '-dm', action='store_true', help='set to draw (single pixel) markers for centroids.')
parser.add_argument('--marker_color', '-mc', type=str, default='class', help='the color of centroid markers. "class" is a darker version of the class color. otherwise, rgb hex codes specify special color choices, e.g. 0x000000 is black.')
parser.add_argument('--draw_lines', '-dl', action='store_true', help='draw dividing lines between regions?')
parser.add_argument('--line_color', '-lc', type=str, default='0x000000', help='color of lines dividing voronoy cells.')
parser.add_argument('--line_dilation_iterations', '-ldi', type=int, default=0, help='how often to binary dilate region boundaries? dilation is applied before erosion.')
parser.add_argument('--line_erosion_iterations', '-lei', type=int, default=1, help='how often to binary erode region boundaries? erosion is applied after dilation.')
# visualize while generating?
parser.add_argument('--show', action='store_true', help='show generated images?')
#output location.
parser.add_argument('--output', '-o', type=str, default='./output', help='output directory for outputting data and labels.')

#TODO: enforce min distance between centroids?
#TODO: pick colors from some nice palettes?
#TODO: draw some more sensible markers?
#TODO: refine line drawing, ie line thickness (scipy.ndimage.binary_erosion, scipy.ndimage.binary_dilation)
    # TODO: explore dilation and erosion parameters

args = parser.parse_args()
if not isinstance(args.bg_colors, list): args.bg_colors = [args.bg_colors]
if not isinstance(args.class_colors, list): args.class_colors = [args.class_colors]
#print(args)

np.random.seed(int(args.random_seed,0))
data = []



for i in tqdm.tqdm(range(args.number), desc='generating samples'):
    #generate some random centroids.
    centroids = np.random.randint(  low=0,
                                    high=args.size,
                                    size=(np.random.randint(low=args.min_centroids, high=args.max_centroids), 2)
                                 )

    #compute 1-nn assignments on a canvas
    XX, YY = np.meshgrid(np.arange(0,args.size), np.arange(0,args.size))
    COORDS = np.concatenate([YY[None,...], XX[None,...]], axis=0).reshape(2,args.size**2).T

    #compute distances and centroid assignments
    distances = DistanceMetric.get_metric(args.distance_metric).pairwise(COORDS, centroids)
    assignments = np.argmin(distances, axis=1)

    #draw image for some (randomly picked) true class and centroid
    k = np.random.randint(low=0, high=centroids.shape[0])
    clazz = np.random.randint(low=0, high=len(args.class_colors))

    #paint foreground and background colored areas, optionally markers.
    #drawing could be done differently, eg with matplotlib itself. for a start, I chose to remain pixel-accurate and thus directly draw images.
    canvas = np.zeros((args.size, args.size, 3), dtype=np.uint8)        # the image to be painted
    regions = np.zeros((args.size, args.size), dtype=np.int32)          # for cataloguing region labels for border computation
    class_area_ground_truth = np.zeros((args.size, args.size, 1), dtype=np.uint8)  # for cataloguing the ground truth region for the true class.
    count_ground_truth = centroids.shape[0]
    for c in np.arange(centroids.shape[0]):
        if c == k:
            rgb = hex2rgb_arr(args.class_colors[clazz])
            rgb = np.clip(rgb + np.random.normal(0, args.class_color_deviation, rgb.shape), 0, 255).astype(np.uint8)
        else:
            rgb = hex2rgb_arr(args.bg_colors[np.random.randint(low=0, high=len(args.bg_colors))])
            rgb = np.clip(rgb + np.random.normal(0, args.bg_color_deviation, rgb.shape), 0, 255).astype(np.uint8)

        I = np.where(assignments == c)[0]
        canvas[COORDS[I,0], COORDS[I,1],:] = rgb
        regions[COORDS[I,0], COORDS[I,1]] = c
        if c == k:
            class_area_ground_truth[COORDS[I,0], COORDS[I,1]] = 255 # max rgb grayscale value

        #paint markers?
        if args.draw_markers:
            if args.marker_color == 'class':
                canvas[centroids[c,0],centroids[c,1]] = (rgb*0.65 + np.zeros_like(rgb)*0.35).astype(np.uint8)
            else:
                canvas[centroids[c,0],centroids[c,1]] = hex2rgb_arr(args.marker_color)


    if args.draw_lines:
        borders = (np.abs(sobel(regions, axis=0)) + np.abs(sobel(regions, axis=1)))  > 0
        npad = args.line_erosion_iterations
        borders = np.pad(borders, ((npad,npad),(npad,npad)), mode='edge') #extend image to avoid information loss during dilation/erosion
        if args.line_dilation_iterations > 0: borders = binary_dilation(borders, iterations=args.line_dilation_iterations)
        if args.line_erosion_iterations > 0: borders = binary_erosion(borders, iterations=args.line_erosion_iterations, structure=np.ones((2,2)).astype(borders.dtype)) #structure here prevents in a 1-iteratin setting the complete removal of vertical and horizontal lines
        borders = borders[npad:-npad,npad:-npad]# undo padding
        canvas[borders] = hex2rgb_arr(args.line_color)



    #store for later
    data.append((canvas, class_area_ground_truth, clazz, count_ground_truth))

    if args.show:
        plt.imshow(canvas)
        plt.show()



# write data
if not os.path.isdir(args.output): os.makedirs(args.output)
with open('{}/labels.txt'.format(args.output), 'wt') as f_labels:
    f_labels.write('# image_id true_class num_regions\n')
    iname_template = "{:0"+str(len(str(args.number-1)))+"d}"
    for i in tqdm.tqdm(range(len(data)), desc='writing data'):
        iname = iname_template.format(i)
        imageio.imwrite('{}/{}.png'.format(args.output, iname), data[i][0])
        imageio.imwrite('{}/{}_gt.png'.format(args.output, iname), data[i][1])
        f_labels.write('{} {} {}\n'.format(iname, data[i][2], data[i][3]))


#dump arguments for reproducibility #TODO convert store_True action things for proper output
with open('{}/args.txt'.format(args.output), 'wt') as f_args:
    arg_line = ''
    for a in vars(args):
        vals = getattr(args, a)
        if isinstance(vals, bool):
            if vals:vals = '' #assume store_true action type variables. convert to produce directly usable arg lines.
            else: continue
        if isinstance(vals, (tuple, list)): vals = ' '.join(vals)
        arg_line += '--{} {}  '.format(a, vals)
    f_args.write(arg_line)

