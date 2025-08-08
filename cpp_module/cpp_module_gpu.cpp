// cpp_module_gpu.cpp - GPU-accelerated version
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/cudaimgproc.hpp>
#include <opencv4/opencv2/cudaarithm.hpp>

namespace py = pybind11;

py::object process_image_gpu(
    py::array_t<uint8_t> input_array,
    py::tuple rect_tuple,
    int blur_kernel = 15)
{
#ifdef DEBUG
    // Performance profiling setup
    cv::TickMeter total_timer, edge_timer, blur_timer, gpu_upload_timer, gpu_download_timer;
    total_timer.start();
    
    std::cout << "=== DEBUG: GPU ACCELERATED VERSION ===" << std::endl;
    std::cout << "blur_kernel: " << blur_kernel << std::endl;
#endif

    // Construct the image using the input_array and buffer info
    py::buffer_info buf_info = input_array.request();
    if (buf_info.ndim != 3 || buf_info.shape[2] != 3) {
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

    // Extract the region of interest (ROI) using rect_tuple
    cv::Rect roi(
        rect_tuple[0].cast<int>(),
        rect_tuple[1].cast<int>(),
        rect_tuple[2].cast<int>(),
        rect_tuple[3].cast<int>()
    );

    cv::Mat roi_image = input_image(roi);
#ifdef DEBUG
    std::cout << "ROI extracted: " << roi_image.rows << "x" << roi_image.cols << std::endl; 
    gpu_upload_timer.start();
#endif

    // GPU-accelerated processing
    cv::cuda::GpuMat gpu_roi, gpu_gray, gpu_blurred, gpu_edges;
    
    // Upload ROI to GPU
    gpu_roi.upload(roi_image);
    
#ifdef DEBUG
    gpu_upload_timer.stop();
    edge_timer.start();
#endif

    // Convert to grayscale on GPU
    cv::cuda::cvtColor(gpu_roi, gpu_gray, cv::COLOR_BGR2GRAY);
    
    // Gaussian blur on GPU  
    cv::cuda::GaussianBlur(gpu_gray, gpu_blurred, cv::Size(5, 5), 0);
    
    // Canny edge detection on GPU
    auto canny = cv::cuda::createCannyEdgeDetector(180, 500);
    canny->detect(gpu_blurred, gpu_edges);

#ifdef DEBUG
    edge_timer.stop();
    gpu_download_timer.start();
#endif

    // Download edges back to CPU for contour detection (OpenCV GPU doesn't have findContours)
    cv::Mat edges;
    gpu_edges.download(edges);

#ifdef DEBUG
    gpu_download_timer.stop();
#endif

    // Morphological operations and contour detection on CPU
    cv::morphologyEx(edges, edges, cv::MORPH_CLOSE, cv::Mat(), cv::Point(-1, -1), 3);
    
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

#ifdef DEBUG
    std::cout << "Edge detection (GPU) time: " << edge_timer.getTimeMilli() << " ms" << std::endl;
#endif

    // Apply GPU-accelerated Gaussian blur to the largest contour area
    if (!contours.empty()) {
        auto largest_contour = *std::max_element(contours.begin(), contours.end(),
            [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                return cv::contourArea(a) < cv::contourArea(b);
            });
            
#ifdef DEBUG
        std::cout << "Largest contour found with area: " << cv::contourArea(largest_contour) << std::endl;
        blur_timer.start();
#endif

        cv::Mat mask = cv::Mat::zeros(roi_image.size(), CV_8UC1);
        cv::drawContours(mask, std::vector<std::vector<cv::Point>>{largest_contour}, -1, cv::Scalar(255), -1);

        // GPU-accelerated blur
        cv::cuda::GpuMat gpu_roi_blur, gpu_mask;
        gpu_mask.upload(mask);
        
        cv::cuda::GaussianBlur(gpu_roi, gpu_roi_blur, cv::Size(blur_kernel, blur_kernel), 0);
        
        // Download blurred result
        cv::Mat blurred_roi;
        gpu_roi_blur.download(blurred_roi);
        
        // Apply mask and copy back to original image
        blurred_roi.copyTo(input_image(roi), mask);

#ifdef DEBUG
        blur_timer.stop();
        std::cout << "Blur processing (GPU) time: " << blur_timer.getTimeMilli() << " ms" << std::endl;
#endif
    }

#ifdef DEBUG
    py::array_t<uint8_t> roi_image_array(
        {roi_image.rows, roi_image.cols, roi_image.channels()},
        {static_cast<size_t>(roi_image.step[0]), static_cast<size_t>(roi_image.step[1]), static_cast<size_t>(roi_image.elemSize())},
        roi_image.data
    );

    // Return debug arrays
    py::array_t<uint8_t> edges_array(
        {edges.rows, edges.cols, edges.channels()},
        {static_cast<size_t>(edges.step[0]), static_cast<size_t>(edges.step[1]), static_cast<size_t>(edges.elemSize())},
        edges.data
    );

    cv::Mat gray;
    gpu_gray.download(gray);
    py::array_t<uint8_t> gray_array(
        {gray.rows, gray.cols},
        {static_cast<size_t>(gray.step), static_cast<size_t>(gray.elemSize())},
        gray.data
    );

    cv::Mat mask = cv::Mat::zeros(roi_image.size(), CV_8UC1);
    if (!contours.empty()) {
        auto largest_contour = *std::max_element(contours.begin(), contours.end(),
            [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                return cv::contourArea(a) < cv::contourArea(b);
            });
        cv::drawContours(mask, std::vector<std::vector<cv::Point>>{largest_contour}, -1, cv::Scalar(255), -1);
    }
    
    py::array_t<uint8_t> mask_array(
        {mask.rows, mask.cols},
        {static_cast<size_t>(mask.step), static_cast<size_t>(mask.elemSize())},
        mask.data
    );

    total_timer.stop();
    std::cout << "=== GPU ACCELERATED PERFORMANCE ===" << std::endl;
    std::cout << "Total processing time: " << total_timer.getTimeMilli() << " ms" << std::endl;
    std::cout << "- GPU upload time: " << gpu_upload_timer.getTimeMilli() << " ms" << std::endl;
    std::cout << "- Edge detection (GPU): " << edge_timer.getTimeMilli() << " ms" << std::endl;
    std::cout << "- Blur processing (GPU): " << blur_timer.getTimeMilli() << " ms" << std::endl;
    std::cout << "- GPU download time: " << gpu_download_timer.getTimeMilli() << " ms" << std::endl;
    std::cout << "=============================" << std::endl;
    
    return py::make_tuple(input_array, gray_array, edges_array, roi_image_array, mask_array);
#else
    return input_array;
#endif
}

#ifdef DEBUG
py::tuple blur_largest_shape_in_rect_gpu(
    py::array_t<uint8_t> input_array,
    py::tuple rect_tuple,
    int blur_kernel = 15)
{
    return process_image_gpu(input_array, rect_tuple, blur_kernel).cast<py::tuple>();
}
#else
py::array_t<uint8_t> blur_largest_shape_in_rect_gpu(
    py::array_t<uint8_t> input_array,
    py::tuple rect_tuple,
    int blur_kernel = 15)
{
    return process_image_gpu(input_array, rect_tuple, blur_kernel).cast<py::array_t<uint8_t>>();
}
#endif

// Module registration
PYBIND11_MODULE(cpp_module_gpu_edge_detector, m) {
    m.def(
        "blur_largest_shape_in_rect_gpu", &blur_largest_shape_in_rect_gpu,
        py::arg("input_array"),
        py::arg("rect_tuple"), 
        py::arg("blur_kernel") = 15,
        "GPU-accelerated detection and blur of largest shape in rectangle."
    );
}