'''
3D atom logo
'''

import pythreejs
import numpy as np
import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from datetime import datetime
from ipywebrtc import ImageRecorder, WidgetStream
from ipywidgets import VBox
import os
import copy
import matplotlib.image as mpimg
import matplotlib.patches as mpatches

fp = FontProperties(family="monospace", weight="bold")
globscale = 1.2

list_atoms = ['C', 'O', 'N', 'S']
atom_letters = dict([(letter, TextPath((-0.30, 0), letter, size=1, prop=fp)) for letter in list_atoms])
atom_colors = {
    'C': [210 / 256, 180 / 256, 140 / 256, 1.0],
    'O': 'red',
    'N': 'blue',
    'S': 'yellow'
}

list_aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y','0']

aa_letters = dict([(letter, TextPath((-0.30, 0), letter, size=1, prop=fp)) for letter in list_aa])
aa_colors = {
    'A': 'gray',
    'C': 'green',
    'D': 'red',
    'E': 'red',
    'F': [199 / 256., 182 / 256., 0., 1.],
    'G': 'gray',
    'H': 'blue',
    'I': 'black',
    'K': 'blue',
    'L': 'black',
    'M': 'green',
    'N': 'purple',
    'P': 'gray',
    'Q': 'purple',
    'R': 'blue',
    'S': 'purple',
    'T': 'purple',
    'V': 'black',
    'W': [199 / 256., 182 / 256., 0., 1.],
    'Y': [199 / 256., 182 / 256., 0., 1.],
    '0': 'black',
}


list_atom_valencies = [
    r'$\; \mathbf{C}$',
    r'$\mathbf{CH}$',
    r'$\mathbf{CH}_2$',
    r'$\mathbf{CH}_3$',
    r'$\mathbf{C\pi}$',
    r'$\; \mathbf{O}$',
   r'$\mathbf{OH}$',
    r'$\mathbf{ N}$',
   r'$\mathbf{NH}$',
    r'$\mathbf{NH}_2$',
    r'$\mathbf{ S}$',
    r'$\mathbf{SH}$',
    r'$\mathbf{Any}$'
]

valency_letters = dict([(letter, TextPath((-0.30, 0), letter, size=1, prop=fp,usetex=True)) for letter in list_atom_valencies])

valency_colors = {
    r'$\; \mathbf{C}$': [210 / 256, 180 / 256, 140 / 256, 1.0],
    r'$\mathbf{CH}$': [210 / 256, 180 / 256, 140 / 256, 1.0],
    r'$\mathbf{CH}_2$': [210 / 256, 180 / 256, 140 / 256, 1.0],
    r'$\mathbf{CH}_3$': [210 / 256, 180 / 256, 140 / 256, 1.0],
    r'$\mathbf{C\pi}$': [203/256,109/256,81/256,1.0],#[210 / 256, 180 / 256, 140 / 256, 1.0],
    r'$\; \mathbf{O}$':'red',
    r'$\mathbf{OH}$':'red',
    r'$\mathbf{ N}$':'blue',
    r'$\mathbf{NH}$':'blue',
    r'$\mathbf{NH}_2$':'blue',
    r'$\mathbf{ S}$':'yellow',
    r'$\mathbf{SH}$':'yellow',
    r'$\mathbf{Any}$':'black'
}


list_hb = ['0','D','A','B']
hb_letters = dict([(letter, TextPath((-0.30, 0), letter, size=1, prop=fp)) for letter in list_hb])
hb_colors = {'0':'black',
            'A':'red',
             'D':'blue',
             'B':'purple'
            }

list_ss = ['H','B','E','G','I','T','S','-']

ss_letters = dict([(letter, TextPath((-0.30, 0), letter, size=1, prop=fp)) for letter in list_ss])
ss_colors = {'H':'red',
             'B':'blue',
             'E':'blue',
             'G':'red',
             'I':'red',
             'T':[210 / 256, 180 / 256, 140 / 256, 1.0],
             'S':[210 / 256, 180 / 256, 140 / 256, 1.0],
             '-':[210 / 256, 180 / 256, 140 / 256, 1.0],
            }




def letterAt(index, x, y, yscale=1, ax=None, type='atom'):
    if type == 'atom':
        categories = list_atoms
        letters = atom_letters
        colors = atom_colors
    elif type == 'valency':
        categories = list_atom_valencies
        letters = valency_letters
        colors = valency_colors
    elif type == 'ss':
        categories = list_ss
        letters = ss_letters
        colors = ss_colors
    elif type == 'hb':
        categories = list_hb
        letters = hb_letters
        colors = hb_colors
    elif type == 'aa':
        categories = list_aa
        letters = aa_letters
        colors = aa_colors
    else:
        print('unsupported type')
        return

    sign = np.sign(yscale)
    if type =='valency':
        offset_negatives = (1 - sign) / 2 * globscale * 1.0
    else:
        offset_negatives = 0
    letter = categories[index]
    text = letters[letter]
    t = mpl.transforms.Affine2D().scale(sign * globscale, yscale * globscale) + \
        mpl.transforms.Affine2D().translate(x + offset_negatives, y) + ax.transData
    p = PathPatch(text, lw=0, fc=colors[letter], transform=t)
    if ax != None:
        ax.add_artist(p)
    return p


def pieAt(index,x,y,yscale=1.,ax=None):
    fraction = [0,1/3,2/3,1.0][index]
    sign = np.sign(yscale)

    t = mpl.transforms.Affine2D().scale(sign, sign*yscale) \
    + mpl.transforms.Affine2D().translate(x,y) + ax.transData
    artists = []
    if fraction != 0.0:
        w1 = mpatches.Wedge((0, 0.5), 0.5, 0,
                       360. * fraction,color='dodgerblue',
                       clip_on=False,transform=t)
        if ax is not None:
            ax.add_artist(w1)
        artists.append(w1)

    if fraction != 1.0:
        w2 = mpatches.Wedge((0, 0.5), 0.5, fraction*360,
                       360.,color='gray',
                       clip_on=False,transform=t)
        if ax is not None:
            ax.add_artist(w2)
        artists.append(w2)
    return artists

def weight_logo_atom(W, threshold=0.5, ymax=3):
    pos_atoms = np.nonzero(W > threshold)[0]
    neg_atoms = np.nonzero(W < -threshold)[0]
    pos_order = np.argsort(W[pos_atoms])
    neg_order = np.argsort(W[neg_atoms])[::-1]

    fig, ax = plt.subplots(figsize=(4, 8))
    ypos = 0.01
    yneg = -0.01
    for atom in pos_atoms[pos_order]:
        weight = W[atom]
        letterAt(atom, 0, ypos, yscale=weight, ax=ax, type='atom')
        ypos += weight

    for atom in neg_atoms[neg_order]:
        weight = W[atom]
        letterAt(atom, 0, yneg, yscale=weight, ax=ax, type='atom')
        yneg += weight
    plt.plot([-0.3, 0.3], [0, 0], c='black', linewidth=2.0)
    plt.xlim([-ymax / 4, ymax / 4])
    plt.ylim([-ymax, ymax])
    plt.axis('off')
    return fig


def weight_logo_valency(W, threshold=0.5, ymax=3,bar=True):
    pos_atoms = np.nonzero(W > threshold)[0]
    neg_atoms = np.nonzero(W < -threshold)[0]
    pos_order = np.argsort(W[pos_atoms])
    neg_order = np.argsort(W[neg_atoms])[::-1]

    if len(neg_atoms) >= 10: # Negative activation whenever any atom is present.
        neg_atoms = np.array([-1],dtype=np.int)
        neg_order = np.array([0],dtype=np.int)

    fig, ax = plt.subplots(figsize=(4, 8))
    ypos = 0.01
    yneg = -0.01
    if bar:
        x = -2/3
    else:
        x = -0.9
    for atom in pos_atoms[pos_order]:
        weight = W[atom]
        letterAt(atom, x, ypos, yscale=weight, ax=ax, type='valency')
        ypos += weight

    for atom in neg_atoms[neg_order]:
        weight = W[atom]
        letterAt(atom, x, yneg, yscale=weight, ax=ax, type='valency')
        yneg += weight
    if bar:
        plt.plot([-1.0, 1.0], [0, 0], c='black', linewidth=2.0)
    plt.xlim([-ymax / 2, ymax / 2])
    plt.ylim([-ymax, ymax])
    plt.axis('off')
    return fig

def weight_logo_aa(W, ymax=2,threshold=0.05):
    pos_aa = np.nonzero(W>threshold)[0]
    pos_order = np.argsort(W[pos_aa])

    neg_aa = np.nonzero(W<-threshold)[0]
    neg_order = np.argsort(W[neg_aa])[::-1]
    fig, ax = plt.subplots(figsize=(4, 8))
    ypos = 0.04
    yneg = -0.04
    for aa in pos_aa[pos_order]:
        weight = W[aa]
        letterAt(list_aa[aa], 0, ypos, yscale=weight, ax=ax, type='aa')
        ypos += weight

    for aa in neg_aa[neg_order]:
        weight = W[aa]
        letterAt(list_aa[aa], 0, yneg, yscale=weight, ax=ax, type='aa')
        yneg += weight
    plt.plot([-0.4, 0.4], [0, 0], c='black', linewidth=4.0)
    plt.xlim([-ymax / 2, ymax / 2])
    plt.ylim([-ymax, ymax])
    plt.axis('off')

    return fig


def weight_logo_aa(PWM_pos, value_pos, PWM_neg=None, value_neg=None, threshold=0.05, ymax=2):
    pos_aa = np.nonzero(PWM_pos > threshold)[0]
    pos_order = np.argsort(PWM_pos[pos_aa])

    if PWM_neg is not None:
        neg_aa = np.nonzero(PWM_neg > threshold)[0]
        neg_order = np.argsort(PWM_neg[neg_aa])

    fig, ax = plt.subplots(figsize=(2, 8))
    ypos = 0.04
    yneg = -0.04
    for aa in pos_aa[pos_order]:
        weight = PWM_pos[aa] * value_pos
        letterAt(aa, 0, ypos, yscale=weight, ax=ax, type='aa')
        ypos += weight

    if PWM_neg is not None:
        for aa in neg_aa[neg_order]:
            weight = PWM_neg[aa] * value_neg
            letterAt(aa, 0, yneg, yscale=weight, ax=ax, type='aa')
            yneg += weight
    if PWM_neg is not None:
        plt.plot([-0.4, 0.4], [0, 0], c='black', linewidth=4.0)
    plt.xlim([-ymax / 2, ymax / 2])
    plt.ylim([-ymax, ymax])
    plt.axis('off')
    return fig


def categories_logo(probability, categories, ax=None, threshold=None, scaling='conservation', multiplier=1.0,
                    orientation='+'):
    ncategories = len(probability)
    if threshold is None:
        threshold = min(1 / ncategories, 0.1)
    assert categories in ['aa', 'atom', 'atom_valency', 'asa', 'ss', 'hb']
    relevant = np.nonzero(probability > threshold)[0]
    order = np.argsort(probability[relevant])

    if ax is None:
        return_fig = True
        fig, ax = plt.subplots(figsize=(2, 8))
    else:
        return_fig = False

    if scaling == 'conservation':
        value = np.log2(ncategories) + (np.log2(probability + 1e-8) * (probability + 1e-8)).sum()
        ymax = np.log2(ncategories)
    else:
        value = 1
        ymax = 1
    value *= multiplier


    y = 0.025 * ymax

    if categories == 'asa':
        xlims = [-0.5, 0.5]
    else:
        xlims = [-0.35, 0.35]

    if orientation == '-':
        value *= -1
        y *= -1
        ax.plot(xlims, [0, 0], c='black', linewidth=2.0)

    for category in relevant[order]:
        weight = probability[category] * value
        if categories == 'asa':
            pieAt(category, 0, y, yscale=weight, ax=ax)
        else:
            letterAt(category, 0, y, yscale=weight, ax=ax, type=categories)
        y += weight

    ax.set_xlim(xlims)
    if orientation == '+':
        ax.set_ylim([0, ymax])
    else:
        ax.set_ylim([-ymax, ymax])
    ax.axis('off')
    if return_fig:
        return fig
    else:
        return


def complex_filter_logo(aa_probability,
                        aa_probability_neg=None,
                        hb_probability=None,
                        hb_probability_neg=None,
                        ss_probability=None,
                        ss_probability_neg=None,
                        asa_probability=None,
                        asa_probability_neg = None,
                        scaling_ = 'conservation',
                        scaling=1.0,
                        scaling_neg=1.0,
                        height=8,width = 1):
    nplots = 1
    if hb_probability is not None:
        nplots +=1
    if ss_probability is not None:
        nplots +=1
    if asa_probability is not None:
        nplots +=1

    figsize = (nplots*width,height)
    fig, ax = plt.subplots(1,nplots,figsize=figsize)
    fig.subplots_adjust(left  = 0.,  # the left side of the subplots of the figure
                        right = 1.0,    # the right side of the subplots of the figure
                        bottom = 0.0,   # the bottom of the subplots of the figure
                        top = 1.0,      # the top of the subplots of the figure
                        wspace = 0.00,   # the amount of width reserved for blank space between subplots
                        hspace = 0.2)   # the amount of height reserved for white space between subplots
    if nplots == 1:
        ax = [ax]

    count = 0
    categories_logo(aa_probability, 'aa', ax=ax[count], scaling=scaling_, multiplier=scaling, orientation='+')
    if aa_probability_neg is not None:
        categories_logo(aa_probability_neg, 'aa', ax=ax[count], scaling=scaling_, multiplier=scaling_neg, orientation='-')
    count +=1
    if asa_probability is not None:
        categories_logo(asa_probability, 'asa', ax=ax[count], scaling=scaling_, multiplier=scaling,
                        orientation='+')
        if asa_probability_neg is not None:
            categories_logo(asa_probability_neg, 'asa', ax=ax[count], scaling=scaling_, multiplier=scaling_neg,
                            orientation='-')
        count +=1
    if hb_probability is not None:
        categories_logo(hb_probability, 'hb', ax=ax[count], scaling=scaling_, multiplier=scaling,
                        orientation='+')
        if hb_probability_neg is not None:
            categories_logo(hb_probability_neg, 'hb', ax=ax[count], scaling=scaling_, multiplier=scaling_neg,
                            orientation='-')
        count +=1
    if ss_probability is not None:
        categories_logo(ss_probability, 'ss', ax=ax[count], scaling=scaling_, multiplier=scaling,
                        orientation='+')
        if ss_probability_neg is not None:
            categories_logo(ss_probability_neg, 'ss', ax=ax[count], scaling=scaling_, multiplier=scaling_neg,
                            orientation='-')
        count +=1

    return fig

def text_to_sprite(msg, position, resolution=64, color="red", fs=1.):
    if not isinstance(position, list):
        position = list(position)
    text = pythreejs.TextTexture(string=msg, size=resolution)
    mat = pythreejs.SpriteMaterial(map=text, transparent=True, color=color)
    return pythreejs.Sprite(material=mat, position=position, scale=[fs, fs, 1])


def matplotlib_to_sprite(fig, position, scale=1.0, figname=None, tmp_folder='tmp/',clear=True,dpi=300,crop=False):
    if not os.path.exists(tmp_folder):
        os.mkdir(tmp_folder)
    timestamp = str(datetime.now()).replace(':', '_').replace(' ', '_')
    if figname is not None:
        figname = 'fig_%s.png' % timestamp
    fig.savefig(tmp_folder + figname, transparent=True,dpi=dpi)
    fig.clear()
    if crop:
        img = mpimg.imread(tmp_folder + figname)
        rows = (img.sum(-1) == 1.*3).min(1) # Remove white
        cols = (img.sum(-1) == 1.*3).min(0)
        img = np.asarray(img[~rows,:][:,~cols],order='c')
        mpimg.imsave(tmp_folder + figname,img,dpi=dpi)
    img_figure = pythreejs.ImageTexture(imageUri=tmp_folder + figname)
    # if clear:
    #     os.system('rm %s%s'%(tmp_folder,figname))
    material = pythreejs.SpriteMaterial(map=img_figure, transparent=True, opacity=1.0, depthWrite=False)
    if not isinstance(position, list):
        position = list(position)
    return pythreejs.Sprite(material=material, position=position, scale=[scale, scale, scale])


def rgb_to_hex(rgb):
    if isinstance(rgb, str):
        return rgb
    else:
        rgb = np.array(rgb)[:3]
        if rgb.max() < 1:
            rgb *= 256
        rgb = np.floor(rgb).astype(np.int)
        return '#%02x%02x%02x' % (rgb[0], rgb[1], rgb[2])


def make_sphere_geometry(npoints):
    # WARNING EXECUTE THIS CELL AND PREVIOUS ONE INDEPENDENTLY
    sg = pythreejs.SphereGeometry(widthSegments=npoints, heightSegments=npoints)
    sg_ = pythreejs.Geometry.from_geometry(sg, store_ref=True)
    return sg_


def downloadable(renderer):
    webgl_stream = WidgetStream(widget=renderer)
    image_recorder = ImageRecorder(stream=webgl_stream)
    return VBox([renderer, image_recorder])

def make_screenshot(renderer,output_name):
    webgl_stream = WidgetStream(widget=renderer)  # renderer = la fenetre 3D
    image_recorder = ImageRecorder(stream=webgl_stream,filename=output_name,format='png')
    image_recorder.autosave = True
    image_recorder.recording = True
    return image_recorder


def show_ellipsoids(list_ellipsoids=[(np.zeros(3), np.eye(3))],
                    list_colors=None,
                    list_figures=None,
                    list_texts=None,
                    list_segments = None,
                    list_additional_objects = [],
                    level=1.0, sg=None,
                    wireframe=True,
                    show_frame=True,
                    fs=1.,
                    scale=2.5,
                    offset=None,
                    camera_position=None,
                    key_light_position=None,
                    opacity=0.2,
                    maxi=10,
                    xlims=None,
                    ylims=None,
                    zlims=None,
                    download=False,
                    crop=False,
                    render = True,
                    tmp_folder = 'tmp/',
                    dpi=300
                    ):
    nellipsoids = len(list_ellipsoids)
    os.system('rm %s/*'%tmp_folder)
    if list_colors is None:
        list_colors = [['red', 'green', 'blue'][u % 3] for u in range(nellipsoids)]
    if list_figures is None:
        list_figures = [None for _ in list_ellipsoids]
    if list_texts is None:
        list_texts = [None for _ in list_ellipsoids]
    if xlims is None:
        xlims = [-maxi, maxi]
    if ylims is None:
        ylims = [-maxi, maxi]
    if zlims is None:
        zlims = [-maxi, maxi]
    if camera_position is None:
        camera_position =  [0.8, 0.5, 0.8]
    if offset is None:
        offset = [0,0,0]
    if key_light_position is None:
        key_light_position = [0.5,1,0.0]
    offset = np.array(offset)

    sphere_V = np.array(sg.vertices)
    sphere_F = sg.faces

    ## Default camera and light positions suited for visualizing data in [-0.5, 0.5]^3. Otherwise, redo.
    key_light_position = copy.deepcopy(key_light_position)
    key_light_position[0] = key_light_position[0] * (xlims[1] - xlims[0]) + (xlims[0] + xlims[1]) / 2
    key_light_position[1] = key_light_position[1] * (ylims[1] - ylims[0]) + (ylims[0] + ylims[1]) / 2
    key_light_position[2] = key_light_position[2] * (zlims[1] - zlims[0]) + (zlims[0] + zlims[1]) / 2


    camera_position = copy.deepcopy(camera_position)
    camera_position[0] = camera_position[0] * (xlims[1] - xlims[0]) + (xlims[0] + xlims[1]) / 2
    camera_position[1] = camera_position[1] * (ylims[1] - ylims[0]) + (ylims[0] + ylims[1]) / 2
    camera_position[2] = camera_position[2] * (zlims[1] - zlims[0]) + (zlims[0] + zlims[1]) / 2

    key_light = pythreejs.DirectionalLight(position=key_light_position, intensity=.3)
    ambient_light = pythreejs.AmbientLight(intensity=.8)
    camera = pythreejs.PerspectiveCamera(position=camera_position)
    controller = pythreejs.OrbitControls(controlling=camera)

    if render:
        children = [camera, key_light, ambient_light]
    else:
        children = []
    children2 = []

    d2camera = np.array([((np.array(camera_position) - list_ellipsoids[n][0]) ** 2).sum() for n in range(nellipsoids)])
    order = np.argsort(d2camera)[::-1]

    for n in order:
        center, inertia = list_ellipsoids[n]
        color = list_colors[n]
        figure = list_figures[n]
        text = list_texts[n]

        lam, U = np.linalg.eigh(inertia)
        sqrt_inertia = np.dot(U, np.sqrt(lam)[:, np.newaxis] * U.T)
        ellipsoids_V = level * np.dot(sphere_V, sqrt_inertia)
        ellipsoids_V = ellipsoids_V.tolist()
        sphereG = pythreejs.Geometry(vertices=ellipsoids_V, faces=sphere_F)

        if figure is not None:
            sprite = matplotlib_to_sprite(figure, center + offset, scale=scale, figname='fig_%s' % n,tmp_folder=tmp_folder,dpi=dpi,crop=crop)
            figure.clear()
            children2.append(sprite)

        if text is not None:
            sprite = text_to_sprite(text, center + offset, color=rgb_to_hex(color), fs=fs)
            children2.append(sprite)

        if wireframe:
            sg_wireframe = pythreejs.WireframeGeometry(geometry=sphereG)
            ellipsoid_wireframe = pythreejs.LineSegments(sg_wireframe,
                                                         material=pythreejs.MeshLambertMaterial(color=rgb_to_hex(color),
                                                                                                transparent=True,
                                                                                                opacity=opacity+0.1),
                                                         position=center.tolist())
            children.append(ellipsoid_wireframe)

        ellipsoid_surface = pythreejs.Mesh(geometry=sphereG,
                                           material=pythreejs.MeshLambertMaterial(color=rgb_to_hex(color),
                                                                                  transparent=True, opacity=opacity),
                                           position=center.tolist())
        children.append(ellipsoid_surface)
    children = children[:3]+children2+children[3:]

    if show_frame:
        g = pythreejs.LineSegmentsGeometry(
            positions=[
                [[0, 0, 0], [xlims[1] * 0.75, 0, 0]],
                [[0, 0, 0], [0, ylims[1] * 0.75, 0]],
                [[0, 0, 0], [0, 0, zlims[1] * 0.75]]
            ],
            colors=[
                [[150/256, 150/256, 150/256], [150/256, 150/256, 150/256]],
                [[150/256, 150/256, 150/256], [150/256, 150/256, 150/256]],
                [[150/256, 150/256, 150/256], [150/256, 150/256, 150/256]]]
        )
        m = pythreejs.LineMaterial(linewidth=2.5, vertexColors='VertexColors')
        frame = pythreejs.LineSegments2(g, m)
        children.append(frame)
        children.append(text_to_sprite('X', np.array([xlims[1] * 0.75 + 0.5, 0, 0]), color=rgb_to_hex([150/256, 150/256, 150/256]), fs=0.5 * fs))
        children.append(text_to_sprite('Y', np.array([0, ylims[1] * 0.75 + 0.5, 0]),color=rgb_to_hex([150/256, 150/256, 150/256]), fs=0.5 * fs))
        children.append(text_to_sprite('Z', np.array([0, 0, zlims[1] * 0.75 + 0.5]),color=rgb_to_hex([150/256, 150/256, 150/256]),fs=0.5 * fs))

    if list_segments is not None:
        g = pythreejs.LineSegmentsGeometry(
            positions=list_segments,
            colors=[ [[150 / 256, 150 / 256, 150 / 256], [150 / 256, 150 / 256, 150 / 256]] for _ in list_segments])
        m = pythreejs.LineMaterial(linewidth=2.5, vertexColors='VertexColors')
        segments = pythreejs.LineSegments2(g, m)
        children.append(segments)

    # children += list_additional_objects
    children = list_additional_objects + children
    if not render:
        return children
    else:
        scene = pythreejs.Scene(children=children)
        renderer = pythreejs.Renderer(camera=camera, scene=scene, controls=[controller],
                                      width=1000, height=1000, antialias=True, sortObjects=False,
                                      clearOpacity=0, alpha=True, autoClear=True)
        if download:
            return downloadable(renderer)
        else:
            return renderer


def make_example(nellipsoids=10, K=10, sg=None):
    list_ellipsoids = []
    list_colors = []
    list_texts = []
    for i in range(nellipsoids):
        center = np.random.randn(3)
        sqrt_cov = np.random.randn(K, 3)
        covariance = np.dot(sqrt_cov.T, sqrt_cov) / K
        color = np.random.rand(3).tolist()

        letter1p = chr(ord('A') + np.random.randint(1, 26))
        letter2p = chr(ord('A') + np.random.randint(1, 26))
        letter1m = chr(ord('A') + np.random.randint(1, 26))
        letter2m = chr(ord('A') + np.random.randint(1, 26))

        msg = '%s%s//%s%s' % (letter1p, letter2p, letter1m, letter2m)
        list_ellipsoids.append((center, covariance))
        list_colors.append(color)
        list_texts.append(msg)

    return show_ellipsoids(list_ellipsoids=list_ellipsoids,
                           list_texts=list_texts,
                           colors=list_colors,
                           sg=sg,
                           xlims=[-5, 5],
                           ylims=[-5, 5],
                           zlims=[-5, 5]
                           )


if __name__ == '__main__':
    # %%
    W = np.random.randn(4)
    fig = weight_logo_atom(W, threshold=0.5, ymax=3)
    fig.show()
    # %%
    alpha = 20
    PWM_pos = np.random.rand(21)
    PWM_pos = (PWM_pos ** alpha) / (PWM_pos ** alpha).sum()
    value_pos = np.random.rand() + 1

    PWM_neg = np.random.rand(21)
    PWM_neg = (PWM_neg ** alpha) / (PWM_neg ** alpha).sum()
    value_neg = - (np.random.rand())
    fig = weight_logo_aa(PWM_pos, value_pos, PWM_neg=PWM_neg, value_neg=value_neg, threshold=0.05, ymax=2.0)
    fig.show()
    # %%
    W = np.random.randn(12)
    W[np.argsort(np.random.randn(12))[:5]] *=0
    fig = weight_logo_valency(W, threshold=0.5, ymax=5)
    fig.show()



    #%%

    W = np.random.randn(21)
    W[np.argsort(np.random.randn(21))[:15]] *=0
    fig = weight_logo_aa2(W, threshold=0.5, ymax=2)
    fig.show()

