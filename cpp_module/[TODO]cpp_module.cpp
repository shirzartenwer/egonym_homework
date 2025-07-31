// cpp_module.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>

namespace py = pybind11;

py::array_t<uint8_t> blur_largest_shape_in_rect(
    py::array_t<uint8_t> input_array,
    py::tuple rect_tuple,
    int blur_kernel = 15,
    uint16_t img_width)
{
    // TODO: Implement this function
    // Construct the image using the input_arry and buffer info
    // Crop the data
    
    // Normalize the ROI 
    // Find the largest shape using contours - need to research the most effective algo

    // blur origianl image with the gaussian blur using blur kernel

    // return it


    // [candidate to fill]
    
    throw std::runtime_error("Function not yet implemented");
}

// Module registration – candidate does not need to change this
PYBIND11_MODULE(cpp_module_edge_detector, m) {
    m.def(
        "blur_largest_shape_in_rect", &blur_largest_shape_in_rect,
        py::arg("input_array"),
        py::arg("rect_tuple"),
        py::arg("blur_kernel") = 15,
        "Detect shapes in a rectangle and blur the largest shape."
    );
}
