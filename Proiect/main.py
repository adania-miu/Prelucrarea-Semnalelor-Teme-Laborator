import os
import scipy
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io, color


def main():
    # exemplul 1: ascent image din scipy
    image = scipy.datasets.ascent().astype('int32')
    template_output_dir = './outputs/example_1'

    # # exemplul 2: imagine cu monede
    # image = io.imread('input\\50 bani.png')
    # image = image[:, :, :3]
    # image = color.rgb2gray(image)
    #
    # template_output_dir = './outputs/example_2'

    # # exemplul 3: imagine cu monede
    # image = io.imread('input\\10 bani.png')
    # image = image[:, :, :3]
    # image = color.rgb2gray(image)
    #
    # template_output_dir = './outputs/example_3'


    # pasul 1: gaussian blurring
    output_dir = os.path.join(template_output_dir, 'gaussian_blurring')
    kernel_size = (5, 5)
    sigma = 1.0
    blurred_image_with_convolution = gaussian_blur_with_convolution(image, kernel_size, sigma, output_dir)
    blurred_image_in_frequency = gaussian_blur_in_frequency(image, 100.0, output_dir)
    blurred_image_with_opencv = gaussian_blur_with_opencv(image.astype('uint8'), kernel_size, sigma, output_dir)
    # ne uitam la diferentele intre cele 3 variante de gaussian blurring
    data = {'original image': image,
            'blurring with convolution\nkernel_size={} sigma={}'.format(kernel_size,
                                                                        sigma): blurred_image_with_convolution,
            'blurring in frequency': blurred_image_in_frequency,
            'blurring with opencv\nkernel_size={} sigma={}'.format(kernel_size, sigma): blurred_image_with_opencv}
    plot_images(data, 'gaussian_blurring.png', output_dir)

    # pasul 2: calculul gradientului
    output_dir = os.path.join(template_output_dir, 'gradients')

    # utilizam operatorii lui Sobel
    gradient_method = 'sobel'
    sobel_gradient_magnitude, sobel_gradient_orientation = compute_gradients(blurred_image_with_convolution,
                                                                             gradient_method,
                                                                             output_dir)
    # utilizam operatorii lui Prewitt
    gradient_method = 'prewitt'
    prewitt_gradient_magnitude, prewitt_gradient_orientation = compute_gradients(blurred_image_with_convolution,
                                                                                 gradient_method,
                                                                                 output_dir)

    # utilizam operatorii lui Scharr
    gradient_method = 'scharr'
    scharr_gradient_magnitude, scharr_gradient_orientation = compute_gradients(blurred_image_with_convolution,
                                                                               gradient_method,
                                                                               output_dir)

    # pasul 3: non maxima supression
    output_dir = os.path.join(template_output_dir, 'non_maxima_suppression')

    # utilizand operatorii lui Sobel
    gradient_method = 'sobel'
    sobel_suppressed_gradient_magnitude = non_maxima_suppression(sobel_gradient_magnitude, sobel_gradient_orientation,
                                                                 gradient_method, output_dir)

    # utilizand operatorii lui Prewitt
    gradient_method = 'prewitt'
    prewitt_suppressed_gradient_magnitude = non_maxima_suppression(prewitt_gradient_magnitude,
                                                                   prewitt_gradient_orientation,
                                                                   gradient_method, output_dir)

    # utilizand operatorii lui Scharr
    gradient_method = 'scharr'
    scharr_suppressed_gradient_magnitude = non_maxima_suppression(scharr_gradient_magnitude,
                                                                  scharr_gradient_orientation,
                                                                  gradient_method, output_dir)

    # pasul 4: edge detection (thresholding si unirea muchiilor)
    output_dir = os.path.join(template_output_dir, 'edge_detection')

    min_threshold = 5.0
    max_threshold = 15.0

    # utilizand operatorii lui Sobel
    gradient_method = 'sobel'
    edges_sobel = edge_detection(sobel_suppressed_gradient_magnitude, min_threshold, max_threshold, gradient_method,
                                 output_dir)

    # utilizand operatorii lui Prewitt
    gradient_method = 'prewitt'
    edges_prewitt = edge_detection(prewitt_suppressed_gradient_magnitude, min_threshold, max_threshold, gradient_method,
                                   output_dir)

    # utilizand operatorii lui Scharr
    gradient_method = 'scharr'
    edges_scharr = edge_detection(scharr_suppressed_gradient_magnitude, min_threshold, max_threshold, gradient_method,
                                  output_dir)

    data = {'original image': image,
            'Sobel operators': edges_sobel,
            'Prewitt operators': edges_prewitt,
            'Scharr operator': edges_scharr}
    file_name = 'my_implementation_vs_opencv.png'
    output_dir = os.path.join(template_output_dir, 'original_and_edges')
    plot_images(data, file_name, output_dir)


def edge_detection(gradient_magnitude, min_threshold, max_threshold, gradient_method, output_dir=None):
    if not (isinstance(gradient_magnitude, np.ndarray) and gradient_magnitude.ndim == 2):
        raise ValueError('parameter gradient_magnitude should be an 2D numpy array!')
    if not (isinstance(min_threshold, float) and isinstance(max_threshold, float) and min_threshold < max_threshold):
        raise ValueError(
            'parameters min_threshold and max_threshold must be of type float and min_threshold<max_threshold!')

    '''
    Explicatii:
        * pe baza celor 2 threshold-uri stabilim ce pixeli fac parte din muchii
        * avem 3 tipuri de pixeli:
            * non muchie: cu pixeli mai mici decat min_threshold si nu sunt luati in considerare
            * strong: cu pixelii mai mari ca max_threshold (ei fac parte sigur din muchie)
            * weak: cu pixeli intre min_threshold si max_threshold (pot sau nu face parte din muchie)
    '''

    # stabilim si marcam edge-urile care sunt strong si weak
    strong_edges = np.zeros_like(gradient_magnitude)
    weak_edges = np.zeros_like(gradient_magnitude)

    strong_edges_indices = gradient_magnitude >= max_threshold
    weak_edges_indices = ((gradient_magnitude >= min_threshold) & (gradient_magnitude < max_threshold))

    edge_mark = 255
    strong_edges[strong_edges_indices] = edge_mark
    weak_edges[weak_edges_indices] = edge_mark

    edges = strong_edges.copy()

    # daca un pixel weak este conectat (prin unul din cei 8 pixeli vecini) la pixelii strong
    # atunci si acest pixel va face parte din muchie
    no_rows, no_cols = weak_edges.shape
    for row_index in range(no_rows):
        for col_index in range(no_cols):
            if row_index == 0 or col_index == 0:
                continue
            if row_index == no_rows - 1 or col_index == no_cols - 1:
                continue

            if weak_edges[row_index, col_index] != 0:
                neighbours_pixels = []
                neighbours_pixels.append(strong_edges[row_index - 1, col_index])
                neighbours_pixels.append(strong_edges[row_index + 1, col_index])
                neighbours_pixels.append(strong_edges[row_index, col_index - 1])
                neighbours_pixels.append(strong_edges[row_index, col_index + 1])
                neighbours_pixels.append(strong_edges[row_index - 1, col_index + 1])
                neighbours_pixels.append(strong_edges[row_index + 1, col_index - 1])
                neighbours_pixels.append(strong_edges[row_index - 1, col_index - 1])
                neighbours_pixels.append(strong_edges[row_index + 1, col_index + 1])

                if np.sum(neighbours_pixels) > 0:
                    edges[row_index, col_index] = edge_mark

    if output_dir is not None:
        data = {'non maxima suppression': gradient_magnitude,
                'edge detection:\nmin_threshold={}\nmax_threshold={}'.format(min_threshold, max_threshold): edges}
        file_name = 'edge_detection_{}.png'.format(gradient_method)
        plot_images(data, file_name, output_dir)

    return edges


def non_maxima_suppression(gradient_magnitude, gradient_orientation, gradient_method, output_dir=None):
    if not (isinstance(gradient_magnitude, np.ndarray) and gradient_magnitude.ndim == 2):
        raise ValueError('parameter gradient_magnitude should be an 2D numpy array!')
    if not (isinstance(gradient_orientation, np.ndarray) and gradient_orientation.ndim == 2):
        raise ValueError('parameter gradient_orientation should be an 2D numpy array!')

    '''
    Descrierea algoritmului din aceasta functie:
        * matricea cu orientarea gradientului contine unghiuri masurate in radiani
        * putem transforma in grade sau putem lucra direct cu radiani: in continuare lucrez in radiani
        * fiecare pixel care nu e pe margine este inconjurat de alti 4 pixeli: la E, W, S si N
        * deci avem 4 directii: orizontala, verticala, si cele 2 diagonale
        * nu ne intereseaza daca orientarea gradientului este de la nord la sus sau de la sud la nord deoarece
          in ambele cazuri la aceiasi vecini pixeli ne vom uita
        * astfel pe cercul trigonometric putem ignora axa OY negativa: adica adunam pi=3.14159 la unghiurile negative
        * va trebui sa discretizam cercul trigonometric in regiuni in functie de orientarea gradientului intrucat
          matricea cu orientarea gradientului contine foarte multe valori care nu corespund cu cele 4 directii din matricea cu magnitudinea gradientului
        * astfel vom avea urmatoarele regiuni: 
            * o regiune centrata in 0pi radiani
            * o regiune centrata in 0.25pi radiani
            * o regiune centrata in 0.5pi radiani
            * o regiune centrata in 0.75pi radiani 
            * o regiune centrata in pi radiani
        * echivalent intervalele pentru cele 4 regiuni se pot obtine utilizand pasi de incrementare pentru 0.125pi radiani:
            * intre 0pi radiani si 0.125pi radiani -> regiunea centrata in 0pi radiani
            * intre 0.125pi radiani si 0.375pi radiani -> regiunea centrata in 0.25pi radiani
            * intre 0.375pi radiani si 0.625pi radiani -> regiunea centrata in 0.5pi radiani
            * intre 0.625pi radiani si 0.875pi radiani -> regiunea centrata in 0.75pi radiani
            * intre 0.875pi radiani si pi radiani -> regiunea centrata in pi radiani
        * stiu ca implementarea canny din opencv face si o interpolare, dar eu am decis sa utilizez doar o discretizare
        * pentru fiecare pixel din matricea cu magnitudinea gradientului ne uitam la magnitudinea celor doi vecini, iar
          daca acesta este mai mare decat magnitudinea celor doi vecini il pastram, altfel il inlocuim cu 0 
    '''

    gradient_orientation[gradient_orientation < 0] += np.pi

    suppressed_gradient_magnitude = np.zeros_like(gradient_magnitude)
    for row_index in range(gradient_magnitude.shape[0]):
        for col_index in range(gradient_magnitude.shape[1]):
            # nu ne uitam la vecinii pixelilor aflati pe margine
            if row_index == 0 or col_index == 0:
                continue
            if row_index == gradient_magnitude.shape[0] - 1 or col_index == gradient_magnitude.shape[1] - 1:
                continue

            orientation_angle = gradient_orientation[row_index, col_index]
            # ne uitam la vecinii de la vest si est
            if orientation_angle >= 0 and orientation_angle < 0.125 * np.pi:
                neighbours_indeces = ((row_index, col_index - 1), (row_index, col_index + 1))
            # ne uitam la vecinii de la NE si SW
            elif orientation_angle >= 0.125 * np.pi and orientation_angle < 0.375 * np.pi:
                neighbours_indeces = ((row_index - 1, col_index + 1), (row_index + 1, col_index - 1))
            # ne uitam la vecinii de la N si S
            elif orientation_angle >= 0.375 * np.pi and orientation_angle < 0.625 * np.pi:
                neighbours_indeces = ((row_index - 1, col_index), (row_index + 1, col_index))
            # ne uitam la vecinii de la NW si SE
            elif orientation_angle >= 0.625 * np.pi and orientation_angle < 0.875 * np.pi:
                neighbours_indeces = ((row_index - 1, col_index - 1), (row_index + 1, col_index + 1))
            # ne uitam la vecinii de la vest si est
            elif orientation_angle >= 0.875 * np.pi and orientation_angle <= np.pi:
                neighbours_indeces = ((row_index, col_index - 1), (row_index, col_index + 1))
            else:
                raise RuntimeError(
                    'there was a problem when traversing the trigonometric circle: please check the implemented algorithm!')

            first_neighbour_indices = neighbours_indeces[0]
            second_neighbour_indices = neighbours_indeces[1]

            first_neighbour_pixel = gradient_magnitude[first_neighbour_indices]
            second_neighbour_pixel = gradient_magnitude[second_neighbour_indices]
            current_pixel = gradient_magnitude[row_index, col_index]

            if current_pixel >= first_neighbour_pixel and current_pixel >= second_neighbour_pixel:
                suppressed_gradient_magnitude[row_index, col_index] = current_pixel

    if output_dir is not None:
        data = {'gradient magnitude': gradient_magnitude,
                'suppressed gradient magnitude': suppressed_gradient_magnitude}
        file_name = '{}_non_maxima_suppression.png'.format(gradient_method)
        plot_images(data, file_name, output_dir)

    return suppressed_gradient_magnitude


def compute_gradients(image, gradient_method, output_dir=None):
    if not (isinstance(image, np.ndarray) and image.ndim == 2):
        raise ValueError('parameter image should be an 2D numpy array!')

    if gradient_method == 'sobel':
        h_x, h_y = sobel_operators()
    elif gradient_method == 'prewitt':
        h_x, h_y = prewitt_operators()
    elif gradient_method == 'scharr':
        h_x, h_y = scharr_operators()
    else:
        raise ValueError('algorithm not implemented for gradient method={}!'.format(gradient_method))

    G_x = scipy.ndimage.convolve(image, h_x)
    G_y = scipy.ndimage.convolve(image, h_y)
    gradient_magnitude = np.sqrt(G_x ** 2 + G_y ** 2)
    gradient_magnitude *= 255.0 / np.max(gradient_magnitude)
    gradient_orientation = np.arctan2(G_y, G_x)

    if output_dir is not None:
        data = {'blurred image': image,
                'horizontal gradient': G_x,
                'vertical gradient': G_y,
                'gradient magnitude': gradient_magnitude,
                'gradient orientation': gradient_orientation}
        file_name = '{}_gradients.png'.format(gradient_method)
        plot_images(data, file_name, output_dir)

    return gradient_magnitude, gradient_orientation


def sobel_operators():
    h_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
    h_y = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])

    return h_x, h_y


def scharr_operators():
    h_x = np.array([[3, 0, -3],
                    [10, 0, -10],
                    [3, 0, -3]])
    h_y = np.array([[3, 10, 3],
                    [0, 0, 0],
                    [-3, -10, -3]])

    return h_x, h_y


def prewitt_operators():
    h_x = np.array([[-1, 0, 1],
                    [-1, 0, 1],
                    [-1, 0, 1]])
    h_y = np.array([[-1, -1, -1],
                    [0, 0, 0],
                    [1, 1, 1]])

    return h_x, h_y


def plot_images(data, file_name, output_dir, plot_grayscale=True):
    if not isinstance(data, dict):
        raise ValueError('parameter data should be a dictionary!')

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    fig, axes = plt.subplots(nrows=1, ncols=len(data), figsize=(15, 10))
    for ax, (plot_title, img) in zip(axes, data.items()):
        ax.imshow(img, cmap='gray') if plot_grayscale else ax.imshow(img)
        ax.set_title(plot_title.capitalize())
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, file_name))


def gaussian_blur_with_opencv(image, kernel_size, sigma, output_dir):
    if not (isinstance(image, np.ndarray) and image.ndim == 2):
        raise ValueError('parameter image should be an 2D numpy array!')
    # aici putem avea si sigma=0 si lasam opencv sa determine valoarea lui in functie de kernel_size
    if not (isinstance(sigma, float) and sigma >= 0):
        raise ValueError('parameter sigma should be a strictly positive number!')
    if not (isinstance(kernel_size, tuple) and len(kernel_size) == 2):
        raise ValueError('parameter kernel_size should be a tuple with two elements!')
    no_rows, no_cols = kernel_size
    if not (isinstance(no_rows, int) and isinstance(no_cols, int) and no_rows % 2 == 1 and no_cols % 2 == 1):
        raise ValueError('the dimensions of the kernel should be odd integers!')

    blurred_image = cv2.GaussianBlur(image, kernel_size, sigma)

    if output_dir is not None:
        data = {'original image': image,
                'blurred image: kernel_size={} sigma={}'.format(kernel_size, sigma): blurred_image}
        file_name = 'gaussian_blurring_with_opencv.png'
        plot_images(data, file_name, output_dir)

    return blurred_image


def gaussian_blur_in_frequency(image, sigma, output_dir):
    if not (isinstance(image, np.ndarray) and image.ndim == 2):
        raise ValueError('parameter image should be an 2D numpy array!')
    if not (isinstance(sigma, float) and sigma > 0):
        raise ValueError('parameter sigma should be a strictly positive number!')

    image_spectrum = np.fft.fft2(image)
    image_centered_spectrum = np.fft.fftshift(image_spectrum)

    no_rows, no_cols = image.shape
    kernel_center = np.array([no_rows // 2, no_cols // 2])
    gaussian_kernel = np.array([[np.exp(
        -squared_distance_between_points(np.array([row_index, col_index]), kernel_center) / (2 * sigma ** 2)) for
        col_index in range(no_cols)] for row_index in range(no_rows)])
    normalization_constant = 1.0 / (2 * np.pi * sigma ** 2)
    gaussian_kernel *= normalization_constant

    filtered_image_centered_spectrum = image_centered_spectrum * gaussian_kernel
    blurred_image = np.fft.ifft2(np.fft.ifftshift(filtered_image_centered_spectrum)).real

    if output_dir is not None:
        data = {'original image': image,
                'spectrum of image': 20 * np.log10(np.abs(image_spectrum)),
                'centered spectrum of image': 20 * np.log10(np.abs(image_centered_spectrum)),
                'gaussian kernel': gaussian_kernel,
                'filtered centered spectrum of image': 20 * np.log10(np.abs(filtered_image_centered_spectrum)),
                'blurred image': blurred_image}
        file_name = 'gaussian_blurring_in_frequency.png'
        plot_images(data, file_name, output_dir)

    return blurred_image


def squared_distance_between_points(first_point, second_point):
    if not (isinstance(first_point, np.ndarray) and first_point.ndim == 1 and first_point.size == 2):
        raise ValueError('parameter first_point should be a 1D numpy array with exact two elements!')
    if not (isinstance(second_point, np.ndarray) and second_point.ndim == 1 and second_point.size == 2):
        raise ValueError('parameter second_point should be a 1D numpy array with exact two elements!')

    squared_distance = np.sum((first_point - second_point) ** 2)

    return squared_distance


def gaussian_blur_with_convolution(image, kernel_size, sigma, output_dir):
    if not (isinstance(image, np.ndarray) and image.ndim == 2):
        raise ValueError('parameter image should be an 2D numpy array!')
    if not (isinstance(sigma, float) and sigma > 0):
        raise ValueError('parameter sigma should be a strictly positive number!')
    if not (isinstance(kernel_size, tuple) and len(kernel_size) == 2):
        raise ValueError('parameter kernel_size should be a tuple with two elements!')
    no_rows, no_cols = kernel_size
    if not (isinstance(no_rows, int) and isinstance(no_cols, int) and no_rows % 2 == 1 and no_cols % 2 == 1):
        raise ValueError('the dimensions of the kernel should be odd integers!')

    # aici practic iau in calcul distanta fata de centrul kernelului
    kernel_center = np.array([no_rows // 2, no_cols // 2])
    gaussian_kernel = np.array([[np.exp(
        -squared_distance_between_points(np.array([row_index, col_index]), kernel_center) / (2 * sigma ** 2)) for
        col_index in range(no_cols)] for row_index in range(no_rows)])
    normalization_constant = 1.0 / (2 * np.pi * sigma ** 2)
    gaussian_kernel *= normalization_constant

    blurred_image = scipy.ndimage.convolve(image, gaussian_kernel)

    if output_dir is not None:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        data = {'original image': image,
                'gaussian kernel: size={} sigma={}'.format(kernel_size, sigma): gaussian_kernel,
                'blurred image': blurred_image}
        file_name = 'gaussian_blurring_with_convolution.png'
        plot_images(data, file_name, output_dir)

    return blurred_image


if __name__ == '__main__':
    main()
