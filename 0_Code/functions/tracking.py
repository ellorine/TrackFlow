
## Optical Flow Functions for Te Horo ## 

import os 
import datetime 
from osgeo import gdal 
import numpy as np 
import cv2 
import pandas as pd
import math 
from scipy.interpolate import griddata 


import functions.utils as utils

### Feature Tracking ###

## Helper Fxns 
def calcMag(displacement_list):  
    '''
    Calculates displacement magnitudes for list of 3-D arrays containing the u- and v-
    components of the displacement

    Parameters
    ----------
    displacement_list : List of 3-D arrays
        List of arrays which contain the u and v components, e.g. [[[u1 v1], ...],
                                                                   [[u2 v2], ...], 
                                                                   ...]

    Returns
    -------
    magnitude_all : List of 2-D arrays
        List containing the 2-D arrays of displacement magnitude.

    '''
    magnitude_all = []
    for a in range(0,len(displacement_list)):
        current_disp = displacement_list[a]
        u_dir = current_disp[...,0]
        v_dir = current_disp[...,1]
        mag = np.hypot(u_dir, v_dir) 
        magnitude_all.append(mag)
    return magnitude_all

def calcMean(list_of_arrays):
    '''
    Calculate the mean array from a list of 2-D arrays

    Parameters
    ----------
    list_of_arrays : List of 2-D arrays
        List of 2-D arrays.

    Returns
    -------
    mean_array : 2-D array
        Mean array of the list_of_arrays.

    '''
    sum_array = np.zeros_like(list_of_arrays[0])
    for a in list_of_arrays:
        sum_array = np.add(sum_array, a)
    mean_array = (sum_array / len(list_of_arrays)).astype(np.float32) 
    return mean_array

def calcSumSquares(list_of_arrays):
    '''
    Calculates the sum of squares array and the mean array from a list of 2-D arrays

    Parameters
    ----------
    list_of_arrays : List of 2-D arrays
        List of 2-D arrays.

    Returns
    -------
    sum_of_squares : 2-D array
        Sum of squares array of the list_of_arrays.
    mean_array : 2-D array
        Mean array of the list_of_arrays.

    '''
    mean_array = calcMean(list_of_arrays)
    sum_of_squares = np.zeros_like(mean_array)
    for a in list_of_arrays:
        squares = np.multiply(np.subtract(a, mean_array), np.subtract(a, mean_array)).astype(np.float32) 
        sum_of_squares = np.add(sum_of_squares, squares)
    return sum_of_squares, mean_array

def calcSD(list_of_arrays):

    '''
    Calculates the standard deviation array and the mean array of a list of 2-D arrays

    Parameters
    ----------
    list_of_arrays : List of 2-D arrays
        List of 2-D arrays.

    Returns
    -------
    sigma_array : 2-D array
        Standard deviation array of the list_of_arrays.
    mean_array :  2-D array
        Mean array of the list_of_arrays.

    '''
    sum_of_squares, mean_array = calcSumSquares(list_of_arrays)
    if len(list_of_arrays) == 1:
        sigma_array = np.full_like(mean_array, np.nan)
    else:
        sigma_array = np.sqrt(sum_of_squares / (len(list_of_arrays)-1))
    
    return sigma_array, mean_array
    




## Optical Flow
def runFeatureDisplacement(data1, data2, output_file, list_of_azimuths=[], list_of_angles=[], dem_proc=True):

    '''
    Runs the optical flow routine for hillshade pairs generated from varying azimuths and angles or optical imagery
    
    Parameters
    ----------
    data1 : string
        Path to earlier DEM/ortho file
    data2 : string
        Path to later DEM/ortho file
    list_of_azimuths : 1-D array of ints
        1-D array containing all of the azimuth angles (in degrees) used to generate hillshades
    list_of_angles : 1-D array of ints
        1-D array containing all of the altitude angles (in degrees) used to generate hillshades
    output_file : string
        File path for output geotiff

    Returns
    -------
    9 band geotiff containing direction, magnitude, u, v, sigma_direction, sigma_magnitude,
    ellipse error, sigma_u, sigma_v.

    '''

    
    ## 1. IMPORT DATA 
    
    ref = utils.importGeotiffToRaster(data1,1) 
    test = utils.importGeotiffToRaster(data2,1) 


    metadatadict = utils.createMetadataDict(data1) #Extract Geographical Metadata
    res = metadatadict['GeoTransform'][1]
        
    ## 2. GENERATE HILLSHADES
    if dem_proc is True:
        print("Generating hillshades and computing optical flow...")
        all_displacement = [] #List of displacement arrays for each hillshade pair
        for i in list_of_azimuths:
            for j in list_of_angles:
                azimuth = i
                altitude = j                
                ref_hs_array = utils.createHillshade(ref, azimuth, altitude)
                test_hs_array = utils.createHillshade(test, azimuth, altitude)
                
                
                ## 3. RUN OPTICAL FLOW
                
                displacement = cv2.calcOpticalFlowFarneback(ref_hs_array, test_hs_array, False, pyr_scale = 0.5, levels = 4,
                                                            winsize = 10, iterations = 5, poly_n = 5, poly_sigma = 1.2, flags = 0)
                
                # ## Convert to meters
                displacement_meters = np.multiply(displacement,res) #1px represents how many metres
                
                all_displacement.append(displacement_meters) #Append flow to array of arrays
    else:
        print("Computing optical flow for single band orthoimages...")
        all_displacement = []
        ## Encode NaN as 0
        ref_array = ref 
        test_array = test 
        
        ref_array[np.isnan(ref_array)] = 0
        test_array[np.isnan(test_array)] = 0
        ## Convert to 8 bit
        ref_ortho_array = np.around(ref_array, decimals=0)
        test_ortho_array = np.around(test_array, decimals=0)
        ref_ortho_array = ref_ortho_array.astype(np.uint8)
        test_ortho_array = test_ortho_array.astype(np.uint8)
        
        displacement = cv2.calcOpticalFlowFarneback(ref_ortho_array, test_ortho_array, False, pyr_scale = 0.5, levels = 4,
                                                    winsize = 10, iterations = 5, poly_n = 5, poly_sigma = 1.2, flags = 0)
        
        # ## Convert to meters
        displacement_meters = np.multiply(displacement,res) #1px represents how many metres
        
        all_displacement.append(displacement_meters) #Append flow to array of arrays
        
    # 4. GET THE DISPLACEMENT MAGNITUDE, VARIANCE IN PREDICTIONS FOR EACH PIXEL, AND AVERAGE VALUES FOR EACH PIXEL
    
    print("Computing variance and average values for each pixel...")
    all_magnitude = calcMag(all_displacement) #Displacement Magnitude
    
    all_direction = [] 
    all_u = []
    all_v = []
    for a in all_displacement:
        u = a[...,0]
        v = a[...,1]
        direction = np.degrees(np.arctan2(v, u)).astype(np.float32) 
        all_direction.append(direction) #direction
        all_u.append(u)
        all_v.append(v)

    sigma_magnitude, average_magnitude = calcSD(all_magnitude) #Standard Deviation and Average Values
    sigma_direction, average_direction  = calcSD(all_direction)
    sigma_u, average_u = calcSD(all_u)
    sigma_v, average_v = calcSD(all_v)
    error_band = np.multiply(np.multiply(np.multiply((np.pi/45),sigma_direction),sigma_magnitude),average_magnitude).astype(np.float32) 

    ## Change predictions to NA if ref/test are NA
    average_magnitude[np.isnan(ref) | np.isnan(test)] = np.nan
    sigma_magnitude[np.isnan(ref) | np.isnan(test)] = np.nan
    sigma_direction[np.isnan(ref) | np.isnan(test)] = np.nan
    average_direction[np.isnan(ref) | np.isnan(test)] = np.nan
    sigma_u[np.isnan(ref) | np.isnan(test)] = np.nan
    sigma_v[np.isnan(ref) | np.isnan(test)] = np.nan
    average_v[np.isnan(ref) | np.isnan(test)] = np.nan
    average_u[np.isnan(ref) | np.isnan(test)] = np.nan
    error_band[np.isnan(ref) | np.isnan(test)] = np.nan
        
    #5. EXPORT GEOTIFF 
    print("Exporting Geotiff...")
    driver = gdal.GetDriverByName('GTiff') # Set up the GTiff driver

    bands = [average_direction, average_magnitude, average_u, average_v, sigma_direction, sigma_magnitude, error_band,
             sigma_u, sigma_v] #Bundle the bands 
  
    utils.createGeoTiff(output_file, 9, bands, driver, 
                    metadatadict["xsize"], metadatadict["ysize"], metadatadict["GeoTransform"],
                    metadatadict["Projection"], gdal.GDT_Float32, NDV = -9999)
    
    utils.buildOverviewFiles(output_file.with_suffix(".tif"))

    return(None)

def runPairwiseOpticalFlow(list_of_dsms, list_of_capture_dates, output_folder, list_of_azimuths=[], list_of_angles=[], tag="", dem_proc=True):
    '''
    Generates geotiffs containing results from optical flow algorithm run on set of hillshades/optical imagery for each valid pair in time series. 

    Parameters
    ----------
    list_of_dsms : list of strings
        List of file paths to input DSMs/single band optical imagery.
    list_of_capture_dates : list of datetime.dates 
        List of dates (in datetime.date format) corresponding to each DSM/single band optical imagery.
    list_of_azimuths : list of ints
        List of sun azimuths used to generate the hillshade pairs.
    list_of_angles : list of ints
        List of sun angles used to generate the hillshade pairs.
    parent_dir : string
        Folder path to the parent directory
    version : int/string
        Version ID for the workflow.

    Returns
    -------
    string
    
    Path to folder containing geotiffs for each pair. 
        Geotiffs have 9 bands: 
        1. direction
        2. magnitude
        3. u
        4. v
        5. sigma direction
        6. sigma magnitude
        7. ellipse error
        8. sigma u
        9. sigma v

    '''
    for i in range(len(list_of_dsms)): #Get a geotiff
        for j in range(len(list_of_dsms)): #Get a second geotiff
        
            days_between = (list_of_capture_dates[j] - list_of_capture_dates[i]).days
            
            if (days_between > 0): #Make sure date 2 occurs later than date 1
                print("Running optical flow on pair: " + str(list_of_capture_dates[i]) + " and " + str(list_of_capture_dates[j]))
                outfile = output_folder / str("Optical_Flow_Pair_" + str(list_of_capture_dates[i]) + "_" + str(list_of_capture_dates[j]) + "_" + tag)
                runFeatureDisplacement(list_of_dsms[i], list_of_dsms[j], outfile, list_of_azimuths, list_of_angles, dem_proc=dem_proc)
    return(None)

def filterOpticalFlowPredictions(predictions_list, aspect_raster, sigma_c, z_sig, small_angle): 


    metadatadict = utils.createMetadataDict(predictions_list[0])

    ##Statistical Thresholds
    sigma_stat = sigma_c*np.sqrt((4-np.pi)/2) #Expected standard dev of stationary pixel
    mag_stat = sigma_c*np.sqrt(np.pi/2) #Expected magnitude of stationary pixel
    ee_stat = 4*np.pi*z_sig*sigma_stat*mag_stat #Expected ellipse error of stationary pixel
    
    aspect = utils.importGeotiffToRaster(aspect_raster, 1)
    
    for processing in predictions_list:
        print("Processing " + str(processing))
        dir = utils.importGeotiffToRaster(processing, 1)
        mag = utils.importGeotiffToRaster(processing, 2)
        u = utils.importGeotiffToRaster(processing, 3)
        v = utils.importGeotiffToRaster(processing, 4)
        sig_dir = utils.importGeotiffToRaster(processing, 5)
        sig_mag = utils.importGeotiffToRaster(processing, 6)
        error_band = utils.importGeotiffToRaster(processing, 7)
        sig_u = utils.importGeotiffToRaster(processing, 8)
        sig_v = utils.importGeotiffToRaster(processing, 9)
        
        
        ##Statistical Thresholds
        sigma_stat = sigma_c*np.sqrt((4-np.pi)/2) #Expected standard dev of stationary pixel
        mag_stat = sigma_c*np.sqrt(np.pi/2) #Expected magnitude of stationary pixel
        ee_stat = 4*np.pi*z_sig*sigma_stat*mag_stat #Expected ellipse error of stationary pixel
        
        #Empty arrays to store filtered u and v results
        u_filt = np.empty_like(u)
        v_filt = np.empty_like(u)
        
        u_filt[:] = np.nan #initialise with nans
        v_filt[:] = np.nan 
        
        #no movement
        cond = (error_band < ee_stat) & (mag < mag_stat) & (sig_dir > small_angle)
        u_filt[cond] = 0
        v_filt[cond] = 0
        
        #movement
        cond = (error_band < ee_stat) & (sig_dir < small_angle)
        u_filt[cond] = u[cond]
        v_filt[cond] = v[cond]
        
        #check compass direction of vector relative to aspect
        vect_angle = np.arctan2(-v, u)
        compass_direction = np.rad2deg((np.arctan2(1, 0) - vect_angle) % (2 * np.pi)) % 360
        
        #if true, vector is pointed downhill...  
        angle_diff = abs(compass_direction - aspect)
        angle_diff[angle_diff > 180] = 360 - angle_diff[angle_diff > 180]
        
        vector_downhill = angle_diff < 90
        
        #no movement
        cond = (error_band < ee_stat) & (mag > mag_stat) & (vector_downhill==False) & (np.isfinite(aspect))
        u_filt[cond] = 0
        v_filt[cond] = 0
        
        #movement
        cond = (error_band < ee_stat) & (mag > mag_stat) & (vector_downhill==True)
        u_filt[cond] = u[cond]
        v_filt[cond] = v[cond]
        
        cond = (error_band < ee_stat) & (mag > mag_stat) & (np.isnan(aspect)) #if we don't have aspect information, then assume movement...
        u_filt[cond] = u[cond]
        v_filt[cond] = v[cond]
        
        ##calculate ee_moving 
        ee_mov = (np.pi/45)*z_sig*small_angle*sigma_c*mag
        
        cond = (error_band >= ee_stat) & (error_band < ee_mov)
        
        u_filt[cond] = u[cond]
        v_filt[cond] = v[cond]
        
        mag_filt = np.sqrt(u_filt**2 + v_filt**2)
        mag_filt[np.isnan(u_filt) | np.isnan(v_filt)] = np.nan
    
        ## Write to geotiff
        bands = [dir, mag, u, v, sig_dir, sig_mag, error_band, sig_u, sig_v, u_filt, v_filt]
    
        driver = gdal.GetDriverByName('GTiff') # Set up the GTiff driver
    
        ##Overwrite geotiff
        utils.createGeoTiff(processing.with_suffix(''), 11, bands, driver, 
                        metadatadict["xsize"], metadatadict["ysize"], metadatadict["GeoTransform"],
                        metadatadict["Projection"], gdal.GDT_Float32, NDV = -9999)
        utils.buildOverviewFiles(processing)
        
    return(None)


## NCC
def calcAvgHillshadeDisplacements(input_dir, output_dir, file_starts_with, sigma_c, z_sig, small_angle):
    
    print("Calculating Average Displacements from hillshades in " + str(input_dir) + "...")
    all_u= []
    all_v = []
    all_max_cc = []
    
    for file in os.listdir(input_dir):
        if file.startswith(file_starts_with): 
            data = pd.read_csv(input_dir / file)
            u = data['dx']
            v = data['dy']
            maxcc = data['max_corrcoeff']
            all_u.append(u)
            all_v.append(v)
            all_max_cc.append(maxcc)
            x = data[' X']
            y = data['Y']
    
    all_mag = np.hypot(all_u, all_v)
    all_dir = np.degrees(np.arctan2(all_v, all_u))
    
    avg_mag = np.mean(all_mag, 0)
    avg_dir = np.mean(all_dir, 0)
    avg_u = np.mean(all_u, 0)
    avg_v = np.mean(all_v, 0)
    avg_maxcc = np.mean(all_max_cc, 0)
    
    sigma_mag = np.std(all_mag, 0, ddof=1)
    sigma_dir = np.std(all_dir, 0, ddof=1)
    error_band = np.multiply(np.multiply(np.multiply((np.pi/45),sigma_dir),sigma_mag),avg_mag)
    error_band[error_band==0] = 0.01
    log_error_band = np.log(error_band)
    ## Filtered results
    
    ##Statistical Thresholds
    sigma_stat = sigma_c*np.sqrt((4-np.pi)/2) #Expected standard dev of stationary pixel
    mag_stat = sigma_c*np.sqrt(np.pi/2) #Expected magnitude of stationary pixel
    ee_stat = 4*np.pi*z_sig*sigma_stat*mag_stat #Expected ellipse error of stationary pixel
    u_filt = np.zeros_like(avg_mag)
    v_filt = np.zeros_like(avg_dir)
    
    #small movement
    u_filt[(error_band < ee_stat) & (avg_mag < mag_stat) & (sigma_dir < small_angle)] = avg_u[(error_band < ee_stat) & (avg_mag < mag_stat) & (sigma_dir < small_angle)]
    v_filt[(error_band < ee_stat) & (avg_mag < mag_stat) & (sigma_dir < small_angle)] = avg_v[(error_band < ee_stat) & (avg_mag < mag_stat) & (sigma_dir < small_angle)]
    #no movement
    u_filt[(error_band < ee_stat) & (avg_mag < mag_stat) & (sigma_dir > small_angle)] = 0
    v_filt[(error_band < ee_stat) & (avg_mag < mag_stat) & (sigma_dir > small_angle)] = 0
    #movement
    u_filt[(error_band < ee_stat) & (avg_mag > mag_stat)] = avg_u[(error_band < ee_stat) & (avg_mag > mag_stat)]
    v_filt[(error_band < ee_stat) & (avg_mag > mag_stat)] = avg_v[(error_band < ee_stat) & (avg_mag > mag_stat)]
    
    #movement
    sigma_mov = sigma_c #Expected standard deviation of moving pixel
    ee_mov = (np.pi/45)*z_sig*small_angle*sigma_mov*avg_mag #Expected ellipse error for moving pixel for each pixel
    u_filt[(error_band > ee_stat) & (error_band < ee_mov)] = avg_u[(error_band > ee_stat) & (error_band < ee_mov)]
    v_filt[(error_band > ee_stat) & (error_band < ee_mov)] = avg_v[(error_band > ee_stat) & (error_band < ee_mov)]
    
    #poor prediction
    u_filt[(error_band > ee_mov)] = np.nan
    v_filt[(error_band > ee_mov)] = np.nan
    
    #
    mag_filt = np.hypot(u_filt, v_filt)
    dir_filt = np.degrees(np.arctan2(v_filt, u_filt))
    
    #Encode NAs as -9999
    u_filt[np.isnan(u_filt)]  = -9999
    v_filt[np.isnan(v_filt)]  = -9999
    mag_filt[np.isnan(u_filt)]  = -9999
    dir_filt[np.isnan(u_filt)]  = -9999
    
    
    ### SAVE TO CSV ### 
    #unfiltered
    data = {' X': x,
            'Y': y,
            'dx': avg_u,
            'dy': avg_v,
            'length': avg_mag,
            'dir': avg_dir,
            'max_corrcoeff': avg_maxcc,
            'ellipse_err': error_band,
            'log_error_band': log_error_band}
    
    #filtered
    data_filt = {' X': x,
            'Y': y,
            'dx': u_filt,
            'dy': v_filt,
            'length': mag_filt,
            'dir': dir_filt,
            'max_corrcoeff': avg_maxcc}
    
    data_df = pd.DataFrame(data)
    data_filt_df = pd.DataFrame(data_filt)
    
    ## Export
    data_df.to_csv(output_dir / "ncc_average_hillshade.csv")
    data_filt_df.to_csv(output_dir / "ncc_average_hillshade_filtered.csv")
    
    return(None)

def interpolateSparseDisplacements(prediction_grid, interpolation_type, results_folder, file_name, ref_dsm, filter_on_correlation_coeff = False):
    
    print("Interpolating sparse grid of displacements...")
    
    ## import data 
    prediction_points = pd.read_csv(prediction_grid)
    
    ## Extract Geographical Metadata for Geotiff
    metadatadict = utils.createMetadataDict(ref_dsm) 
    geotransform = metadatadict.get("GeoTransform")
    upperl_n = geotransform[3]
    upperl_e = geotransform[0]
    scale = geotransform[1]

        
    x = prediction_points[' X']
    y = prediction_points['Y']
    
    ## Get array positions of x,y coordinates
    all_x_pos = []
    all_y_pos = []
    all_points = []
    
    for i in range(len(x)):
        x_pos = int(np.around((x[i] - upperl_e)/scale))
        y_pos = int(np.around((upperl_n - y[i])/scale))
        all_x_pos.append(x_pos)
        all_y_pos.append(y_pos)
        all_points.append([x_pos,y_pos])
    
    if filter_on_correlation_coeff == True:
        print("Filtering on correlation coefficient...")
        ##For filtering on correlation coefficient
        prediction_points.loc[prediction_points.max_corrcoeff < 0.6 , ['dx', 'dy']] = -9999, -9999
        
    ## Get reference grid to interpolate to (image coordinates)
    xrange = range(0, metadatadict.get('xsize'))
    yrange = range(0, metadatadict.get('ysize'))
    y_grid, x_grid = np.mgrid[yrange, xrange]
    
    print("Interpolating via " + interpolation_type)
    interpolated_u_grid = griddata(all_points, prediction_points['dx'], (x_grid, y_grid), method=interpolation_type) #method = nearest for filtered results
    interpolated_v_grid = griddata(all_points, prediction_points['dy'], (x_grid, y_grid), method=interpolation_type) #method = nearest for filtered results
    interpolated_cc_grid = griddata(all_points, prediction_points['max_corrcoeff'], (x_grid, y_grid), method=interpolation_type) #method = nearest for filtered results

    ## Write to geotiff 
    driver = gdal.GetDriverByName('GTiff')
    
    bands = [interpolated_u_grid, interpolated_v_grid, interpolated_cc_grid]
    utils.createGeoTiff(results_folder / file_name, 3, bands, driver, 
                    metadatadict["xsize"], metadatadict["ysize"], metadatadict["GeoTransform"],
                    metadatadict["Projection"], gdal.GDT_Float32, NDV = -9999)
    
    utils.buildOverviewFiles(results_folder / str(file_name + ".tif"))
    return(None)

def extractValidationSubset(prediction_gtiff, u_band, v_band, validation_points, x_start,y_start):
    
    print("importing data...")
    ## Get x and y displacements
    u = utils.importGeotiffToRaster(prediction_gtiff, u_band)
    v = utils.importGeotiffToRaster(prediction_gtiff, v_band)
    
    u[u==-9999] = np.nan
    v[v==-9999] = np.nan
    
    ## Extract metadata 
    metadatadict = utils.createMetadataDict(prediction_gtiff) #Extract Geographical Metadata
    geot = list(metadatadict.get('GeoTransform')) #Extract Geotransform
    upperl_n, upperl_e, scale = geot[3], geot[0], geot[1]

    
    ## Get geographic coordinates    
    print("getting geographic coordinates...")
    easting = np.empty_like(u)      
    northing = np.empty_like(u)  
    new_northing = np.empty_like(u)  
    new_easting = np.empty_like(u)  
    
    for row in range(0,len(u)):
        for col in range(0,len(u[0])):
            e, n = utils.calcCoordinateInArray(row, col,upperl_n,upperl_e,scale)          
            easting[row][col] = e
            northing[row][col] = n
            if np.isnan(u[row][col]):
                new_easting[row][col] = np.nan
            else:
                new_easting[row][col] = e + u[row][col] #easting + U
            if np.isnan(v[row][col]):
                new_northing[row][col] = np.nan   
            else:
                new_northing[row][col] = n - v[row][col] #northing + V

    coords = np.dstack((easting,northing,new_easting,new_northing,u,v))
    
    # ## Extract validation subset
    print("extracting validation subset...")
    validation = pd.read_csv(validation_points,header=0)
    
    subset = []
    
    for i in range(0,len(validation)):
        x = math.ceil(validation[x_start][i]/2.)*2 #x-coordinate rounded UP to nearest even
        y = math.ceil(validation[y_start][i]/2.)*2 #y-coordinate rounded UP to nearest even
        
        sub = coords[(coords[...,0] == x) & (coords[...,1]==y)]                
        subset.append(sub)
    

    subset = pd.DataFrame(np.concatenate(subset),columns=["X_2018","Y_2018",
                                                              "X_2020_PRED","Y_2020_PRED","u","v"])
    
    return subset, validation

def extractAllValidationSubsets(optical_flow_results_dir, validation_points):
    for file in os.listdir(optical_flow_results_dir):
        pred = optical_flow_results_dir / file
        if file.endswith(".tif"):
            if file.endswith("hs.tif"):
                
                ##get filtered data
                prediction_data, validation_data = extractValidationSubset(pred, 10, 11, validation_points, "X_2018", "Y_2018")
                
                file_name = file.split("2020-02-09_",1)[1]
                #save prediction data as a csv
                prediction_data.to_csv(optical_flow_results_dir / "validation" / str("Validation_Subset_" + file_name[:-4] + "_filtered.csv"))
    
                ##get unfiltered data
                prediction_data, validation_data = extractValidationSubset(pred, 3, 4, validation_points, "X_2018", "Y_2018")
                
                file_name = file.split("2020-02-09_",1)[1]
                #save prediction data as a csv
                prediction_data.to_csv(optical_flow_results_dir / "validation" / str("Validation_Subset_" + file_name[:-4] + ".csv"))
            else:
                ##get unfiltered data
                prediction_data, validation_data = extractValidationSubset(pred, 3, 4, validation_points, "X_2018", "Y_2018")
                
                file_name = file.split("2020-02-09_",1)[1]
                #save prediction data as a csv
                prediction_data.to_csv(optical_flow_results_dir / "validation" / str("Validation_Subset_" + file_name[:-4] + ".csv"))
    
    return(print("Done!"))
 
 
