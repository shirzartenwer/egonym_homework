### Documentation on how I solved this task

## Content

1. Logic of the problem and how to solve it -- through ChatGPT
2. DevOps and Linter Setup
3. Iterated impelmentation with Copilot
4. Got sample data and did paramter tuning
    4.0 Inspected the first blurring results, that it blurred to much area
    4.1 Changed sample image to a simpler one for easier tunning process
    4.2 Find usable rectangle paramter to extract ROI image
    4.3 Tunning the blur factor and didn't differ too much
    4.4 Introduced morphological extension to connect edges. Turn the first blur parameter again and the hystersis threshold of the Canny detection function 

5. Implemented another solution just using Claude Code CLI and compared the results against it for improvement
6. Performance improvement










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



## 08-05

Turns out tuning the mask was not as easy as it apepars. The detected edges through Canny directly influence how the Mask will look like. Getting a clearer picture as training data also makes a lot of differences. Considering this, I changed my training data from [A smiling laday with scarf](./input_images_test/1/smiling_lady.jpg) to a [standard linkedin protrait picture](./input_images_test/2/better_protrait_rect160_1_200_240.jpg). The edges and masks for the lady was returned as [](./output_images_test/report/smiling_lady_edges.jpg) and the mask was only one contour [](./output_images_test/report/smiling_lady_mask.jpg). 

In contrast, for the same parameter, without changing anything, the [edge picture for the standard linkedin protrait was ](./output_images_test/report/better_protrait_rect160_1_200_240_edges.jpg) and [the mask was ](./output_images_test/report/better_protrait_rect160_1_200_240_mask.jpg). At least, this time the mask was a contour that just missed one closing line.

After settling with the second picture, I tuned the following parameter:
1. The hysteries threshould of the Canny function.
2. The blur kernnel in the first Gaussina Blur function.

Tunining them lead to a little bit better results: [a more detailed edges](./output_images_test/report/tunning/better_protrait_rect160_1_200_240_edges.jpg) and at least [a closing mask](./output_images_test/report/tunning/better_protrait_rect160_1_200_240_mask.jpg).

From here, I then asked AI on how to improve it better. Go suggestions to apply dilataion, or erosion. In the end, after some trial and error, I got the best mask result so far using ```cv::morphologyEx(edges, edges, cv::MORPH_CLOSE...);``` to connect contours and. The end result I got was then [a good edge](./output_images_test/better_protrait_edges.jpg), [a mask with the shape of the head](./output_images_test/better_protrait_mask.jpg), which lead to a [good blur](./output_images_test/better_protrait_rect160_1_200_240.jpg).


After I got this result with this protrait, I went on and tunned the parameters a little bit and blured other 3 pictures. During this process, I gained more intuitive understanding of different parameters. For example, for the [5 th picture](./input_images_test/5/mens_gromming_rect100_1_400_300.jpg) where the the contrast between the face and cloth color is huge, having a very big paramter for the lower and upperbound of hysteresis threshold gives the best result [](./output_images_test/mens_gromming_mask.jpg). Also in this case, controlloing the lower threshold to 180 would be sufficient and upper threshold can be any number bigger than 180.




