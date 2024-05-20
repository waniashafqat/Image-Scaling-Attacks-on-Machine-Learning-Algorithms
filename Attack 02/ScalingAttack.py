import numpy as np
from PIL import Image
import cvxpy as cp
import cv2

def get_coefficients(m, n, mprime, nprime, scale_func=Image.NEAREST):
    In_max = 255  # Maximum pixel value in an image (white)

    # Create an identity matrix of size m*m, scaled to the maximum pixel value (white box)
    matrix_of_white_box = np.identity(m) * In_max
    white_box = Image.fromarray(matrix_of_white_box.T)

    # Scale the white box down to the size m' * m
    resized_white_box = white_box.resize((mprime, m), scale_func)

    # Calculate the vertical coefficient matrix CL (size m' * m)
    CL = np.array(resized_white_box).T / In_max

    # Repeat the process for the width to calculate the horizontal coefficient matrix CR (size n' * n)
    matrix_of_white_box = np.identity(n) * In_max
    white_box = Image.fromarray(matrix_of_white_box.T)
    resized_white_box = white_box.resize((n, nprime), scale_func)
    CR = np.array(resized_white_box).T / In_max

    return CL, CR

def generate_attack_image(source_image, target_image, scale_func=Image.NEAREST):
    source_image_array = np.array(source_image)
    target_image_array = np.array(target_image)

    m, n = source_image_array.shape
    mprime, nprime = target_image_array.shape
    CL, CR = get_coefficients(m, n, mprime, nprime, scale_func)

    # Initialize delta_one_vertical matrix (size m * n') with zeros
    delta_one_vertical = np.zeros((m, nprime))

    # Resize the source image to intermediate size m * n'
    intermediate_source_image = source_image.resize((m, nprime), scale_func)
    intermediate_source_image_array = np.array(intermediate_source_image).T

    # Vertical scaling attack
    for col in range(nprime):
        delta_one_vertical[:, col] = get_perturbation(
            intermediate_source_image_array[:, col],
            target_image_array[:, col],
            CL,
            obj='min'
        )

    # Create the intermediate attack image array (size m * n')
    intermediate_attack_image_array = (intermediate_source_image_array + delta_one_vertical).astype('uint8')

    # Initialize delta_one_horizontal matrix (size m * n) with zeros
    delta_one_horizontal = np.zeros((m, n))

    # Horizontal scaling attack
    for row in range(m):
        delta_one_horizontal[row, :] = get_perturbation(
            source_image_array[row, :],
            intermediate_attack_image_array[row, :],
            CR,
            obj='min'
        )

    attack_image_array = (source_image_array + delta_one_horizontal).astype('uint8')
    attack_image = Image.fromarray(attack_image_array)

    return attack_image

def get_perturbation(source_vector, target_vector, convert_matrix, obj):
    n = source_vector.size
    perturb = cp.Variable(n)

    # Define the function to be maximized or minimized (L2 norm of the perturbation vector)
    function = cp.norm(perturb)
    if obj == 'max':
        objective = cp.Maximize(function)
    else:
        objective = cp.Minimize(function)

    constraints = [
        source_vector + perturb >= 0,  # Pixel values must be non-negative
        source_vector + perturb <= 255  # Pixel values must not exceed 255
    ]

    # Constraint for the difference between the transformed source vector and the target vector
    if convert_matrix[0].size == source_vector.size:
        constraints.append(cp.norm_inf((convert_matrix @ (source_vector + perturb)) - target_vector) <= (0.01 * 255))
    else:
        constraints.append(cp.norm_inf((convert_matrix.T @ (source_vector + perturb)) - target_vector) <= (0.01 * 255))

    prob = cp.Problem(objective, constraints)
    prob.solve()

    if prob.status != cp.OPTIMAL:
        print(f"Optimization problem status: {prob.status}")

    return perturb.value

def implement_attack(source_image, target_image):
    source_image = source_image.convert('RGB').split()
    target_image = target_image.convert('RGB').split()

    scale_func = Image.NEAREST

    # Generate the attack image for each channel (R, G, B)
    R = generate_attack_image(source_image[0], target_image[0], scale_func)
    G = generate_attack_image(source_image[1], target_image[1], scale_func)
    B = generate_attack_image(source_image[2], target_image[2], scale_func)

    attack_image = Image.merge('RGB', (R, G, B))

    attack_image.show()
    attack_image.resize(target_image[0].size, scale_func).show()

    attack_image.save("attack_image.jpg")
