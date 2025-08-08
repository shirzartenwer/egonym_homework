#!/usr/bin/env python3
"""
GPU vs CPU performance test for edge detection
"""

import cv2
import numpy as np
import time

def cpu_edge_detection(image, rect):
    """CPU-based edge detection matching our C++ implementation"""
    x, y, w, h = rect
    roi = image[y:y+h, x:x+w]
    
    start_time = time.perf_counter()
    
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection  
    edges = cv2.Canny(blurred, 180, 500)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    total_time = (time.perf_counter() - start_time) * 1000
    
    largest_area = 0
    if contours:
        largest_area = max(cv2.contourArea(c) for c in contours)
    
    return {
        'total_time_ms': total_time,
        'contours_found': len(contours),
        'largest_area': largest_area,
        'edges_shape': edges.shape
    }

def gpu_edge_detection(image, rect):
    """GPU-accelerated edge detection using available CUDA functions"""
    x, y, w, h = rect
    roi = image[y:y+h, x:x+w]
    
    start_time = time.perf_counter()
    
    # Upload to GPU
    gpu_roi = cv2.cuda_GpuMat()
    gpu_roi.upload(roi)
    
    # Convert to grayscale on GPU
    gpu_gray = cv2.cuda.cvtColor(gpu_roi, cv2.COLOR_BGR2GRAY)
    
    # Create Canny edge detector
    canny = cv2.cuda.createCannyEdgeDetector(180, 500)
    gpu_edges = canny.detect(gpu_gray)
    
    # Download edges for contour detection (CPU operation)
    edges = gpu_edges.download()
    
    # Morphological operations (CPU - not available in GPU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # Find contours (CPU operation)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    total_time = (time.perf_counter() - start_time) * 1000
    
    largest_area = 0
    if contours:
        largest_area = max(cv2.contourArea(c) for c in contours)
    
    return {
        'total_time_ms': total_time,
        'contours_found': len(contours),
        'largest_area': largest_area,
        'edges_shape': edges.shape
    }

def run_performance_test(image_path, rect):
    """Run both CPU and GPU tests on the same image"""
    print(f"\n=== Testing {image_path} ===")
    print(f"ROI: {rect}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load {image_path}")
        return
    
    print(f"Image shape: {image.shape}")
    
    # Warm up GPU
    _ = gpu_edge_detection(image, rect)
    
    # Run CPU test multiple times for better average
    cpu_times = []
    for i in range(3):
        result_cpu = cpu_edge_detection(image, rect)
        cpu_times.append(result_cpu['total_time_ms'])
    
    # Run GPU test multiple times for better average
    gpu_times = []
    for i in range(3):
        result_gpu = gpu_edge_detection(image, rect)
        gpu_times.append(result_gpu['total_time_ms'])
    
    cpu_avg = np.mean(cpu_times)
    gpu_avg = np.mean(gpu_times)
    speedup = cpu_avg / gpu_avg if gpu_avg > 0 else 0
    
    print("\n--- PERFORMANCE RESULTS ---")
    print(f"CPU Average Time: {cpu_avg:.2f}ms")
    print(f"GPU Average Time: {gpu_avg:.2f}ms")
    print(f"Speedup: {speedup:.2f}x")
    print(f"CPU Contours: {result_cpu['contours_found']}")
    print(f"GPU Contours: {result_gpu['contours_found']}")
    print(f"CPU Largest Area: {result_cpu['largest_area']:.1f}")
    print(f"GPU Largest Area: {result_gpu['largest_area']:.1f}")
    
    return {
        'cpu_time': cpu_avg,
        'gpu_time': gpu_avg,
        'speedup': speedup,
        'cpu_contours': result_cpu['contours_found'],
        'gpu_contours': result_gpu['contours_found']
    }

if __name__ == "__main__":
    print("GPU vs CPU Edge Detection Performance Test")
    print("==========================================")
    
    # Test cases matching our previous tests
    test_cases = [
        ("input_images_test/1/smiling_lady.jpg", (100, 1, 400, 300)),
        ("input_images_test/2/better_protrait_rect160_1_200_240.jpg", (160, 1, 200, 240)),
        ("input_images_test/3/Kai-Mende_rect200_1_550_650.jpg", (200, 1, 550, 650)),
        ("input_images_test/5/mens_gromming_rect100_1_400_300.jpg", (100, 1, 400, 300))
    ]
    
    results = []
    for image_path, rect in test_cases:
        result = run_performance_test(image_path, rect)
        if result:
            results.append({
                'image': image_path.split('/')[-1],
                **result
            })
    
    print("\n" + "="*60)
    print("SUMMARY RESULTS")
    print("="*60)
    for r in results:
        print(f"{r['image']:<40} CPU: {r['cpu_time']:.2f}ms  GPU: {r['gpu_time']:.2f}ms  Speedup: {r['speedup']:.2f}x")
    
    if results:
        avg_speedup = np.mean([r['speedup'] for r in results])
        print(f"\nAverage GPU Speedup: {avg_speedup:.2f}x")