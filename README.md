# TrackFlow
TrackFlow is a package to measure surface displacements using hillshades derived from high-resolution Digital Surface Models (DSMs). See "Measuring landslide-driven ground displacements with high-resolution DEMs and optical flow (Carle et al. 2022)" for more details.

# Requirements
- Python 3.9 with packages opencv (4.5.5), matplotlib (3.5.1), numpy (1.23.1), pandas (1.4.4), scipy (1.8.0), gdal (3.4.1), yaml (0.2.5) installed
- Windows 10 or Linux Machine

# Description of Files
- config.yml references the file paths to the data and is called by the python script. 
- 0_Code contains the code. runTrackFlow.py executes the program in the command line. The "functions" folder contains the helper functions. 
- 1_Data contain the data necessary to reproduce the results in the paper, including the DSMs, Imagery, outputs from CIAS, and the manually derived validation vectors. 
- 2_Optical_Flow_Results is where the optical flow outputs are stored for the whole area of interest (Geotiff) and validation points (csv).
- 3_CIAS_Results is where the averaged CIAS results are stored as well as the interpolations of the sparse grid ouputs from CIAS. 
- 4_Validation is where the summary tables for the validation vectors are stored. 
- 5_Plots is where the output plots are stored.

# Instructions 
1. Download TrackFlowPackage folder and put somewhere in your file system.
2. Ensure that required Python packages are installed
3. Edit the "config.yml" file inside of the trackflowpackage folder. Edit the windows and/or linux drivemaps to match your system. Edit the directory path to the location of the TrackFlowPackage. 
4. Run runTrackFlow.py in Python command line. The file takes one argument, "-c", which specifies the path to the config.yml file. E.g. "python path_to_code/runTrackFlow.py -c path_to_config/config.yml"

# Understanding the Results
- 2_Optical_Flow_Results has the results from the optical flow algorithm stored as a 9 or 11 band geotiff. Bands are as follows:
  1. θ - direction angle of the displacement vector for each pixel
  2. l - magnitude of the displacement vector
  3. U - x-direction component of the displacement vector
  4. V - y-direction component of the displacement vector
  5. σ_d - standard deviation of the angle of displacement
  6. σ_m - standard deviation of the displacement magnitude
  7. error band - error band constructed from σ_d, σ_m, and l
  8. σ_u - standard deviation of the u component 
  9. σ_v - standard deviation of the v component 
  10. u_filt - u component for masked vectors (unreliable vectors removed)
  11. v_filt - v component for masked vectors (unreliable vectors removed) 
- validation folder within 2_Optical_Flow_Results contains the optical flow predictions for the manually derived displacement vectors (csvs). These predictions are made for the different datasets (ortho, hillshade, 24 hillshade pairs, 168 hillshade pairs) with or without filtering schemes applied. These files have the starting X,Y coordinate and ending X,Y coordinate of each prediction vector as well as the u- and v- components.
- 3_CIAS_Results contains two folders, fullfield has the interpolated results from CIAS over the whole area of interest, validation has the predictions for the validation vectors for different datasets (avg of 24 hillshades, 1 hillshade, ortho, avg of 24 hillshade filtered) as csvs. 
- The fullfield geotiffs have 3 bands:
  1. U - x-direction component of the displacement vector
  2. V - y-direction component of the displacement vector
  3. Cross Correlation Coefficient - NCC coefficient corresponding to the prediction 
- The validation sets have the X,Y starting coordinates, dx, dy which are the U,V components, length and direction of the vectors, the NCC coefficient, and the error band & log(error band) (average hillshade sets only) for each vector. dx, dy = -9999 if vectors are filtered by the hillshade scheme. 
- 4_Validation contain the tables summarising the comparisons between the manually derived vectors and the predictions for various datasets for the NCC and the Optical Flow tracking methods
- 5_Plots contain the summary plots illustrating selected results that are included in the paper.

# Notes
- All imagery is georeferenced using New Zealand Transverse Mercator projection (NZTM, EPSG: 2193), all elevations are height above ellipsoid 
- Displacement measurements are in metres


