# SFND_Radar_Project

<p align="center">
  <img src="https://github.com/englucrai/SFND_Radar_Project/blob/main/SFND_Radar.jpg"/>
</p>


Project for tracking a vehicle through radar. Fast Fourier Transform, Convolution, and Constant False Alarm Rate (CFAR) are implemented to acquire the Doppler velocity and range estimations as well as filtering the results for better results.

---

This READMe will walk you through the following :

- Implementation steps for the 2D CFAR process.
- Selection of Training, Guard cells and offset.
- How to suppress the non-thresholded cells at the edges.

## 2D CFAR process

The CFAR (Constant False Alarm Ratio) is a dynamic thresholding method that will be used to determine the presence of a target vehicle in the simulated environment. 

The method adapts thresholding based on target surroundings so it can track an object within the range. Therefore is a filtering method applied to extract better results from Range Doppler Map (RDM) generated by a 2D FFT (Fast Fourier Transform).

The Range Doppler Map is a matrix containing Range and Relative Doppler Velocity. For each element, a location window is applied. The window's size is defined by the number of Guard and Training cells used, and it's important to notice that only the Training cells will be used to estimate the noise ratio of the surroundings of the explored area.

Accordingly to the figure above, we can affirm that the location window dimensions are defined by `(2Tr+2Gr+1)` x `(2Td+2Gd+1)`

## Steps for 2D CFAR process

Its implementation can be done following these steps:

1. Determine the number of Training cells for each dimension. Similarly, pick the number of guard cells;

2. Slide the cell under test across the complete matrix. Make sure the CUT (Cell Under Training) has a margin for Training and Guard cells from the edges;

3. For every iteration sum the signal level within all the training cells. To sum convert the value from logarithmic to linear using the db2pow function;

4. Average the summed values for all of the training cells used;

5. After averaging convert it back to logarithmic using pow2db;

6. Further add the offset to it to determine the threshold;

7. Compare the signal under CUT against this threshold;

8. If the CUT level > threshold assigns it a value of 1, else equate it to 0;

The process above will generate a thresholded block, which is smaller than the Range Doppler Map as the CUTs cannot be located at the edges of the matrix due to the presence of Target and Guard cells. Hence, those cells will not be thresholded.

9. To keep the map size the same as it was before CFAR, equate all the non-thresholded cells to 0;

Any location window that can not generate a full matrix according to its defined dimension will be disregarded since it is a corner.
 
---

## RESULTS

The results are shown in the following Figures. The parameters used for CFAR implementation were:


#### Guard cells dimension (range,doppler)
- Gr = 4
- Gd = 4

#### Training cells (range, doppler)
- Tr = 10 
- Td = 8

#### Noise to Signal ratio offset
- offset = 6

<p align="center">
  <img src="https://github.com/englucrai/SFND_Radar_Project/blob/main/figure1_1d_fft.jpg" alt="Sublime's custom image"/>
</p>

<p align="center">
  <img src="https://github.com/englucrai/SFND_Radar_Project/blob/main/figure1_2d_fft.jpg" alt="Sublime's custom image"/>
</p>

<p align="center">
  <img src="https://github.com/englucrai/SFND_Radar_Project/blob/main/figure1_2d_cfar.jpg" alt="Sublime's custom image"/>
</p>


