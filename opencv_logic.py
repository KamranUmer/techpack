import cv2
import numpy as np
import streamlit as st


def apply_logo_realistic(cap_path, logo_path, dest_points, out_path):
    """
    Applies a logo to a cap image with perspective warping.
    """
    try:
        cap_img = cv2.imread(cap_path)
        # Load the logo, preserving its channels (3 for JPG, 4 for PNG)
        logo_img = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)

        if cap_img is None or logo_img is None:
            st.error("Error: Could not read one of the images. Check paths.")
            return None

        h, w = logo_img.shape[:2]
        src_points = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        dest_points_np = np.array(dest_points, dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(src_points, dest_points_np)

        # Check if the logo has an alpha (transparency) channel
        if logo_img.shape[2] == 4:
            logo_rgb = logo_img[:, :, :3]
            alpha_channel = logo_img[:, :, 3]
        else:
            logo_rgb = logo_img
            alpha_channel = np.ones((h, w), dtype=np.uint8) * 255

        warped_logo_rgb = cv2.warpPerspective(logo_rgb, matrix, (cap_img.shape[1], cap_img.shape[0]))

        warped_alpha_mask = cv2.warpPerspective(alpha_channel, matrix, (cap_img.shape[1], cap_img.shape[0]))

        mask = cv2.cvtColor(warped_alpha_mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0

        cap_float = cap_img.astype(np.float32) / 255.0
        logo_float = warped_logo_rgb.astype(np.float32) / 255.0

        blended_float = (logo_float * mask) + (cap_float * (1.0 - mask))

        blended_img = (blended_float * 255).astype(np.uint8)

        cv2.imwrite(out_path, blended_img)
        return out_path

    except Exception as e:
        st.error(f"An error occurred during image processing: {e}")
        return None


# def generate_bent_grid(four_corners, bend_factor=0.0, grid_size=(5, 5)):
#     """
#     Generates a 25-point (5x5) grid based on 4 corner points and a bend factor.
#     The bend is applied as a vertical parabolic curve.
#     """
#     tl, tr, br, bl = four_corners

#     # Create arrays for easy interpolation
#     top_edge = np.linspace(tl, tr, grid_size[0])
#     bottom_edge = np.linspace(bl, br, grid_size[0])

#     # Create the initial flat, perspective-corrected grid
#     flat_grid = np.linspace(top_edge, bottom_edge, grid_size[1])

#     # Calculate the direction and magnitude of the bend
#     # The "down" vector is perpendicular to the top edge
#     top_mid = (np.array(tl) + np.array(tr)) / 2
#     bottom_mid = (np.array(bl) + np.array(br)) / 2
#     down_vector = bottom_mid - top_mid

#     # The max bend amount is a fraction of the height of the shape
#     max_bend = np.linalg.norm(down_vector) * 0.5 * bend_factor

#     bent_grid = []
#     for i, row in enumerate(flat_grid):
#         new_row = []
#         for point in row:
#             # v_ratio is how far down the grid we are (0.0 at top, 1.0 at bottom)
#             v_ratio = i / (grid_size[1] - 1)

#             # Use a parabolic function: 4 * (x - x^2)
#             # This is 0 at the ends (v_ratio=0, 1) and 1 in the middle (v_ratio=0.5)
#             parabolic_factor = 4 * (v_ratio - v_ratio**2)

#             # Apply the offset
#             offset = down_vector * (max_bend / np.linalg.norm(down_vector)) * parabolic_factor
#             new_point = point + offset
#             new_row.append(new_point)
#         bent_grid.append(new_row)

#     # Flatten the grid to a list of points for the next function
#     return [tuple(p) for row in bent_grid for p in row]


# def apply_logo_with_bend(cap_path, logo_path, dest_points_grid, out_path, grid_size=(5, 5)):
#     """
#     Applies a logo to a cap image with a realistic bend using Thin Plate Spline.
#     (This function remains mostly the same as the one I provided before)
#     """
#     try:
#         cap_img = cv2.imread(cap_path)
#         logo_img = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)

#         if cap_img is None or logo_img is None:
#             st.error("Error: Could not read one of the images.")
#             return None

#         h, w = logo_img.shape[:2]

#         # Create the source grid on the flat logo image
#         src_points = []
#         for i in range(grid_size[1]):
#             for j in range(grid_size[0]):
#                 x = j * (w / (grid_size[0] - 1))
#                 y = i * (h / (grid_size[1] - 1))
#                 src_points.append([x, y])

#         src_points = np.array(src_points, dtype=np.float32).reshape((-1, 1, 2))
#         dest_points = np.array(dest_points_grid, dtype=np.float32).reshape((-1, 1, 2))

#         tps = cv2.createThinPlateSplineShapeTransformer()
#         matches = [cv2.DMatch(i, i, 0) for i in range(len(src_points))]
#         tps.estimateTransformation(src_points, dest_points, matches)

#         warped_img = tps.warpImage(logo_img)

#         # Blend the warped logo onto the cap
#         if warped_img.shape[2] == 4:
#             logo_rgb = warped_img[:, :, :3]
#             alpha_channel = warped_img[:, :, 3]
#         else:
#             logo_rgb = warped_img
#             alpha_channel = np.ones(warped_img.shape[:2], dtype=np.uint8) * 255

#         mask = alpha_channel > 0
#         rows, cols = np.where(mask)
#         if len(rows) == 0 or len(cols) == 0:  # Handle cases where the warp is empty
#             cv2.imwrite(out_path, cap_img)
#             return out_path

#         min_row, max_row = np.min(rows), np.max(rows)
#         min_col, max_col = np.min(cols), np.max(cols)

#         cap_roi = cap_img[min_row:max_row, min_col:max_col]
#         logo_roi = logo_rgb[min_row:max_row, min_col:max_col]
#         mask_roi = mask[min_row:max_row, min_col:max_col]

#         blended_roi = np.where(mask_roi[..., None], logo_roi, cap_roi)

#         final_img = cap_img.copy()
#         final_img[min_row:max_row, min_col:max_col] = blended_roi

#         cv2.imwrite(out_path, final_img)
#         return out_path

#     except Exception as e:
#         st.error(f"An error occurred during bending process: {e}")
#         return None
