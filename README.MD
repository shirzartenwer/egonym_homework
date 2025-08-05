# Thoughts behind solving this task

1. Research about the questions using ChatGPT
2. Looking into the code and understand what's asked. Checking the API's of each function and looking for which data structure is needed.
3. Writing the code step by step with AI
4. Looking at the default alogrithm it uses and researching into them to see if there is anything that can be improved
    4.1 Different Edge Detector Algo: https://opencv.org/blog/edge-detection-using-opencv/
        key take 

    4.2  



## 08-04
Reading up on edge detector to see if there is anyhting that can be improved upon. Meanwhile, there's some takeaway from the article

### take aways:
1. "Color images are often converted to grayscale before applying edge detection techniques like Canny, Sobel, or Laplacian. This simplifies the process by reducing the image to a single intensity channel, making it easier to detect edges based on intensity changes"

2. "Since the result is in floating-point format, it’s converted to an 8-bit unsigned format using convertScaleAbs() so it can be displayed or saved properly"


Testing: I should also inspect the biggest shape detected in the image to compare if any of the algorithm actually performs better.
- After trying to do this, I got the case where contour is not a convex polygon or not even a polygon.