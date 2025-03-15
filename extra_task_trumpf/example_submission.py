from typing import Union

import numpy as np
import PIL


def add_watermark(file_path: str) -> Union[PIL.Image, None]:
    """
    FUNCTION SIGNATURE: add_watermark(file_path: str) -> PIL.Image

    Adds random noise as and example 'watermak' to an image.

    Args:
        file_path (str): Valid path to image file.
    Returns:
        PIL.Image: Image with added watermark or None if an error occurred.

    WARNING: 
    Only basic dependencies (like opencv, numpy, PIL, torch) are available in the environment.  
    If you need to use different, specifically 'weird' libraries pleas include them in the submission, inside the function body.
            
    SUBMITTED BY TEAM: <team_name>

    README: <describe your though process and possible improvements / further work>
    """

    try:
        img = PIL.Image.open(file_path).convert(
            "RGB"
        )  # convert to RGB for numpy compatibility.
        img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to 0-1 range

        noise = np.random.normal(0, 0.5, img_array.shape)
        noisy_img_array = np.clip(img_array + noise, 0, 1)  # Clip to 0-1 range

        noisy_img = PIL.Image.fromarray(
            np.uint8(noisy_img_array * 255)
        )  # convert back to uint8 for PIL.

        return noisy_img
    
    except FileNotFoundError:
        print(f"Error: Image file '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None