// cpp_module.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv4/opencv2/opencv.hpp>

namespace py = pybind11;

py::object process_image_internal(
    py::array_t<uint8_t> input_array,
    py::tuple rect_tuple,
    int blur_kernel = 15)         // Only default parameter
{
#ifdef DEBUG
    std::cout << "=== DEBUG: Function called ===" << std::endl;
    std::cout << "blur_kernel: " << blur_kernel << std::endl;
#endif

    // Construct the image using the input_array and buffer info
    py::buffer_info buf_info = input_array.request();
    if (buf_info.ndim != 3 || buf_info.shape[2] != 3) {
        // handle the case for gray images, where input shape is (height, width)
        throw std::runtime_error("Input array must be a 3D array with 3 channels (RGB).");
    }

    
    size_t height = buf_info.shape[0];
    size_t width = buf_info.shape[1];   
    uint8_t* data_ptr = static_cast<uint8_t*>(buf_info.ptr);
#ifdef DEBUG    
    size_t channels = buf_info.shape[2];
    std::cout << "Array shape: [" << height << ", " << width << ", " << channels << "]" << std::endl;
#endif  
    // Create OpenCV Mat from the buffer
    cv::Mat input_image(height, width, CV_8UC3, data_ptr);

    //Extract the region of interest (ROI) using rect_tuple
    cv::Rect roi(
        rect_tuple[0].cast<int>(),
        rect_tuple[1].cast<int>(),
        rect_tuple[2].cast<int>(),
        rect_tuple[3].cast<int>()
    );

    // TODO: take the reference of the input image, instead of copying
    cv::Mat roi_image = input_image(roi);
#ifdef DEBUG
    std::cout << "ROI extracted: " << roi_image.rows << "x" << roi_image.cols << std::endl; 
#endif

    // Convert the ROI to grayscale for better edge detection
    std::vector<std::vector<cv::Point>> contours;
    cv::Mat gray, blurred, edges, mask;
    cv::cvtColor(roi_image, gray, cv::COLOR_BGR2GRAY);
#ifdef DEBUG
    std::cout << "Gray image created: " << gray.rows << "x" << gray.cols << std::endl;
#endif
    // Apply Gaussian blur and Canny edge detection
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
    cv::Canny(blurred, edges, 180, 500);
    cv::morphologyEx(edges, edges, cv::MORPH_CLOSE, cv::Mat(), cv::Point(-1, -1), 3);
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Apply Gaussian blur to the largest shape with the specified blur_kernel
    if (!contours.empty()) {
        auto largest_contour = *std::max_element(contours.begin(), contours.end(),
            [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                return cv::contourArea(a) < cv::contourArea(b);
            });
#ifdef DEBUG
        std::cout << "Largest contour found with area: " << cv::contourArea(largest_contour) << std::endl;
#endif
        mask = cv::Mat::zeros(roi_image.size(), CV_8UC1);
        cv::drawContours(mask, std::vector<std::vector<cv::Point>>{largest_contour}, -1, cv::Scalar(255), -1);

        cv::Mat blurred;
        cv::GaussianBlur(roi_image, blurred, cv::Size(blur_kernel, blur_kernel), 0);
        blurred.copyTo(input_image(roi), mask);
        
    }
#ifdef DEBUG
    py::array_t<uint8_t> roi_image_array(
        {roi_image.rows, roi_image.cols, roi_image.channels()},
        {static_cast<size_t>(roi_image.step[0]), static_cast<size_t>(roi_image.step[1]), static_cast<size_t>(roi_image.elemSize())},
        roi_image.data
    );


    // Return the edges detected by Canny for testing
    py::array_t<uint8_t> edges_array(
        {edges.rows, edges.cols, edges.channels()},
        {static_cast<size_t>(edges.step[0]), static_cast<size_t>(edges.step[1]), static_cast<size_t>(edges.elemSize())},
        edges.data
    );

    py::array_t<uint8_t> gray_array(
        {gray.rows, gray.cols},
        {static_cast<size_t>(gray.step), static_cast<size_t>(gray.elemSize())},
        gray.data
    );

    py::array_t<uint8_t> mask_array(
        {mask.rows, mask.cols},
        {static_cast<size_t>(mask.step), static_cast<size_t>(mask.elemSize())},
        mask.data
    );

    return py::make_tuple(input_array, gray_array, edges_array, roi_image_array, mask_array);
#else
    return input_array;
#endif
}

#ifdef DEBUG
py::tuple blur_largest_shape_in_rect(
    py::array_t<uint8_t> input_array,
    py::tuple rect_tuple,
    int blur_kernel = 15)
{
    return process_image_internal(input_array, rect_tuple, blur_kernel).cast<py::tuple>();
}
#else
py::array_t<uint8_t> blur_largest_shape_in_rect(
    py::array_t<uint8_t> input_array,
    py::tuple rect_tuple,
    int blur_kernel = 15)
{
    return process_image_internal(input_array, rect_tuple, blur_kernel).cast<py::array_t<uint8_t>>();
}
#endif


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
