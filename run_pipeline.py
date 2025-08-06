import argparse
import os
import cv2
from typing import Tuple
from pydantic import BaseModel, field_validator, model_validator
import cpp_module_edge_detector as cpp_module  # Assuming cpp_module is the C++ binding for the function

    
class PipelineParams(BaseModel):
    input_folder: str
    output_folder: str
    rect_tuple: Tuple[int, int, int, int]
    blur_kernel: int

    @field_validator('input_folder', 'output_folder')
    def folder_must_be_string(cls, v):
        if not isinstance(v, str):
            raise ValueError("Folder paths must be strings.")
        return v

    @field_validator('blur_kernel')
    def blur_kernel_must_be_odd(cls, v):
        if v <= 0 or v % 2 == 0:
            raise ValueError("Blur kernel size must be an odd positive integer.")
        return v

    @field_validator('rect_tuple')
    def rect_tuple_must_have_four_ints(cls, v):
        if len(v) != 4:
            raise ValueError("Rectangle must be a tuple of four integers (x, y, width, height).")
        if not all(isinstance(i, int) for i in v):
            raise ValueError("All elements of the rectangle tuple must be integers.")
        return v

    @model_validator(mode='after')
    def check_folders_exist(self):
        if not os.path.exists(self.input_folder):
            raise ValueError(f"Input folder '{self.input_folder}' does not exist.")
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        return self


def run_pipeline(input_folder: str, output_folder: str, rect_tuple: Tuple[int, int, int, int], blur_kernel: int = 15) -> None:
    # Validate parameters using Pydantic
    params = PipelineParams(
        input_folder=input_folder,
        output_folder=output_folder,
        rect_tuple=rect_tuple,
        blur_kernel=blur_kernel
    )
    
    # List files in input_folder with valid image extensions
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
    image_files = [f for f in os.listdir(params.input_folder) if f.lower().endswith(valid_extensions)]
    if not image_files:
        raise ValueError(f"No valid image files found in '{params.input_folder}'.")
    
    # Process each image file
    for image_file in image_files:
        image_path = os.path.join(params.input_folder, image_file)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Failed to read image '{image_file}'. Skipping.")
            continue
            
        try:
            # Call the C++ function to blur the largest shape in the specified rectangle
            result_img  = cpp_module.blur_largest_shape_in_rect(img, params.rect_tuple, params.blur_kernel)
            # Write result to output_folder
            output_path = os.path.join(params.output_folder, image_file)

            cv2.imwrite(output_path, result_img)

            print(f"Processed and saved: {output_path} with rectangle {params.rect_tuple} and blur kernel {params.blur_kernel}")
        except Exception as e:
            print(f"Error processing '{image_file}': {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image processing pipeline (Python + C++)")
    parser.add_argument("--input", required=True, help="Input images folder")
    parser.add_argument("--output", required=True, help="Output images folder")
    parser.add_argument("--rect", nargs=4, type=int, metavar=('X', 'Y', 'W', 'H'), required=True, help="Rectangle (x y w h) for ROI")
    parser.add_argument("--blur_kernel", type=int, default=15, help="Blur kernel size (odd integer)")
    args = parser.parse_args()


    try:
        run_pipeline(args.input, args.output, tuple(args.rect), args.blur_kernel)
    except ValueError as e:
        print(f"Input validation error: {e}")
        exit(1)