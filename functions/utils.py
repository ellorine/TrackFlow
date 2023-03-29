

#### TOOLS USED ACROSS MODULES ##### 

## importGeotiffBoundBoxToRaster - import geotiff band to np array according to bounding box specifications
## importGeotiffToRaster - import geotiff band to np array
## createMetadataDict - create metadata dictionary from geotiff file
## createGeotiff - create geotiff from list of arrays
## buildOverviewFiles - create overview files for a geotiff
## mapSystemDrive - map system drive depending on whether using linux or windows machine
## calcPositionInArray - calculate r, c for a n, e coordinate in an array


from osgeo import gdal
from osgeo import osr
import numpy as np
import os 
import sys 
from pathlib import Path 
import datetime 


def checkTeHoroParameters(parameter_dictionary):
    if len(parameter_dictionary.get("list_of_dsms"))==2:
        if parameter_dictionary.get("list_of_dsms")[0].is_file() & parameter_dictionary.get("list_of_dsms")[1].is_file():
            list_of_dsms = parameter_dictionary.get("list_of_dsms")
        else:
            print("One or more DSM paths are invalid. Exiting.")
            sys.exit()
    else:
        print("Please supply two DSM paths. Exiting.")
        sys.exit()
    if len(parameter_dictionary.get("list_of_orthos"))==2:
        if parameter_dictionary.get("list_of_orthos")[0].is_file() & parameter_dictionary.get("list_of_orthos")[1].is_file():
            list_of_orthos = parameter_dictionary.get("list_of_orthos")
        else:
            print("One or more Ortho paths are invalid. Exiting.")
            sys.exit()
    else:
        print("Please supply two Ortho paths. Exiting.")
        sys.exit()
    if parameter_dictionary.get("aspect_raster").is_file():
        aspect_raster = parameter_dictionary.get("aspect_raster")
    else:
        print("Aspect path invalid. Exiting")
        sys.exit()
    if len(parameter_dictionary.get("list_of_capture_dates"))==2:
        if isinstance(parameter_dictionary.get("list_of_capture_dates")[0], datetime.date) & isinstance(parameter_dictionary.get("list_of_capture_dates")[1], datetime.date):
            list_of_capture_dates = parameter_dictionary.get("list_of_capture_dates")
        else:
            print("One or more DSM dates are invalid. Exiting.")
            sys.exit()
    else:
        print("Please supply DSM dates. Exiting.")
        sys.exit()
    if parameter_dictionary.get("cias_results_folder").is_dir():
        cias_results_folder = parameter_dictionary.get("cias_results_folder")
    else:
        print("CIAS results folder invalid or does not exist. Exiting")
        sys.exit()
    if parameter_dictionary.get("validation_vectors").is_file():
        validation_vectors = parameter_dictionary.get("validation_vectors")
    else:
        print("Validation vector file is invalid or does not exist. Exiting")
        sys.exit()
        
    if isinstance(parameter_dictionary.get("sigma_c"), float):
        sigma_c = parameter_dictionary.get("sigma_c")
    else:
        print("sigma_c must be a float. Exiting.")
        sys.exit()
    if isinstance(parameter_dictionary.get("z_sig"), float):
        z_sig = parameter_dictionary.get("z_sig")
    else:
        print("z_sig must be a float. Exiting.")
        sys.exit()
    if isinstance(parameter_dictionary.get("small_angle"), int):
        small_angle = parameter_dictionary.get("small_angle")
    else:
        print("small_angle must be an int. Exiting.")
        sys.exit()
    
    print("All parameters valid! :)")
    
    return(list_of_dsms, list_of_orthos, list_of_capture_dates, cias_results_folder, validation_vectors, sigma_c, 
           z_sig, small_angle, aspect_raster)






def importGeotiffToRaster(path_object, band=1, bbox=None):   
    
    ##bbox = [minx, miny, maxx, maxy]
    try:
        ds = gdal.Open(os.fspath(path_object), gdal.GA_ReadOnly) 
    except OSError as err:
        print("Oops unable to open " + str(path_object))
        print(err)

    try:
        # extract first band (assumed to be DEM)
        band = ds.GetRasterBand(band)    
        xsize = band.XSize
        ysize = band.YSize  
        
        if bbox is None: #open the whole thing
            bbox = [int(0), int(0), xsize, ysize]  
            array = band.ReadAsArray(bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]).astype(np.float32)   
        else:
            array = band.ReadAsArray(int(bbox[0]), int(bbox[1]), int(bbox[2]-bbox[0]), int(bbox[3]-bbox[1])).astype(np.float32)   
         
        nodata = band.GetNoDataValue()
        array[array == nodata] = np.nan #encode no data values as nans 
        
    except OSError as err:
        print("Oops unable to read from " + str(path_object))
        print(err)
        
    return array


def getGeoInfo(path_object):
    '''
    Returns projection info from a geotiff

    Parameters
    ----------
    FileName : String
        path to geotiff file.

    Returns
    -------
    xsize : int
        Number of columns.
    ysize : int
        Number of rows.
    GeoT : 1-D array
        Geotransform.
    Projection : string
        Projection.
    DataType : string
        Data Type.

    '''

    SourceDS = gdal.Open(os.fspath(path_object), gdal.GA_ReadOnly)

    xsize = SourceDS.RasterXSize
    ysize = SourceDS.RasterYSize
    GeoT = SourceDS.GetGeoTransform()
    Projection = osr.SpatialReference()
    Projection.ImportFromWkt(SourceDS.GetProjectionRef())
    DataType = gdal.GDT_Float32
    
    SourceDS = None 
    return xsize, ysize, GeoT, Projection, DataType

def createMetadataDict(path_object):
    '''
    Returns a dictionary with projection info from a geotiff

    Parameters
    ----------
    FileName : string
        Path to a geotiff file.

    Returns
    -------
    Dictionary : dictionary
        Dictionary with projection info.

    '''

    xsize, ysize, GeoT, Projection, DataType = getGeoInfo(path_object)
    Dictionary = {"xsize": xsize,
                  "ysize": ysize,
                  "GeoTransform": GeoT,
                  "Projection": Projection,
                  "Data Type": DataType}
    return Dictionary

## Build overview files
def buildOverviewFiles(infile):
    '''Builds overview for tiff.
    
    Args:
       infile (str):  Path to tiff file.
        
    Returns:
       None

    '''
    data = gdal.Open(os.fspath(infile), 0)  
    gdal.SetConfigOption('COMPRESS_OVERVIEW', 'DEFLATE')
    ##Check how big the array is
    a = np.array(data.GetRasterBand(1).ReadAsArray())
    h = len(a)
    w = len(a[0])
    if (h < 200) & (w < 200): #if small array...
        data.BuildOverviews('NEAREST', [4, 8, 16], gdal.TermProgress_nocb)
    else:
        data.BuildOverviews('NEAREST', [4, 8, 16, 32, 64, 128], gdal.TermProgress_nocb)
    del data  
    return(print("Overview file built."))
    
def createGeoTiff(Name, number_of_bands, band_list, driver,
                  xsize, ysize, GeoT, Projection, DataType, NDV, compress=False, ovr=False):
    '''
    Returns a multi-band  or single band Geotiff file from np arrays 

    Parameters
    ----------
    Name : str
        Path and file name of new geotiff without file extension.
    number_of_bands : int
        Integer # of bands to be created.
    band_list : list of 2-D arrays
        List of arrays to be converted into geotiff bands.
    driver : gdal Driver object
        GDAL Driver.
    xsize : int
        # cols of output raster.
    ysize : int
        # rows of output raster.
    GeoT : 1-D tuple
        Geotransform tuple.
    Projection : GDAL projection object
        Spatial Reference Object from getGeoInfo.
    DataType : gdal data type
        gdal datatype of output raster:  format as "gdal.GDT_Float32, etc.".
    NDV : int
        No Data Value.

    Returns
    -------
    NewFileName : string
        Name of new file created.

    '''

    NewFileName = os.fspath(Name) + '.tif'
    # on linux, compression hangs when done inside multithreading
    if compress:
        DataSet = driver.Create( NewFileName, xsize, ysize, number_of_bands, DataType, ['COMPRESS=DEFLATE', 'PREDICTOR=2', 'ZLEVEL=1', 'BIGTIFF=YES'])
    else:
        DataSet = driver.Create( NewFileName, xsize, ysize, number_of_bands, DataType)
    DataSet.SetGeoTransform(GeoT)
    DataSet.SetProjection( Projection.ExportToWkt() )
    # Write the array
    for i in range(0,number_of_bands): #For 
        band_list[i][np.isnan(band_list[i])] = NDV
        DataSet.GetRasterBand(i+1).WriteArray(band_list[i])   # write raster band to the GeoTiff
        DataSet.GetRasterBand(i+1).SetNoDataValue(NDV)

    DataSet = None
    if ovr:
        buildOverviewFiles(NewFileName)
     
    return NewFileName

def mapWindowsDrive(path_object, config_file):
    
    if path_object is not None:
        #Get the windows drive map
        drive_map = Path(config_file['project']['drivemap']['win32'])
        
        if sys.platform == 'win32':    
            if path_object.drive == "": #we have a linux path since drive is missing
                parts_to_keep = Path(*path_object.parts[5:]) #remove first 4 parts of the path (the Linux drive Map) 
                updated_path_object = Path(drive_map / parts_to_keep)
            elif (len(path_object.anchor) == 3) & (path_object.anchor[1]==":"): #we have a windows path 
                if path_object.anchor[0:2] == config_file['project']['drivemap']['win32'][0:2]: #check that drive map is the same as the windows drive map specified in the config
                        updated_path_object = Path(path_object)
                else:
                    print("Path: " + str(path_object) + " drive map does not match project drive map for " + sys.platform + " system...Exiting...")
                    sys.exit() 
            else:
                print("Path: " + str(path_object) + " is neither a valid linux or windows path...Exiting...")
                sys.exit()     
        
        elif sys.platform == 'linux':
            path_object_str = str(path_object)
            path_object = Path(path_object_str.replace('\\', '/'))
            
            if path_object.parts[0] == config_file['project']['drivemap']['win32'][0:2]: #check if drive map is the same as the windows drive map specified in the config
                updated_path_object = Path(path_object)
            elif path_object.parts[0] == "/": #a bit brute force, but we have a linux path
                parts_to_keep = Path(*path_object.parts[5:]) #remove first 4 parts of the path (the Linux drive Map) 
                updated_path_object = Path(drive_map / parts_to_keep)
            else:
                print("Path: " + str(path_object) + " is neither a valid linux or windows path...Exiting...")
                sys.exit()   
                
        return(updated_path_object)

def mapLinuxDrive(path_object, config_file):
    
    
    #Get the linux drive map 
    drive_map = Path(config_file['project']['drivemap']['linux'])   
    
    if sys.platform == "win32":
        # check the drive map of the path_object 
        if path_object.drive == "": #we have a linux path since drive is missing
            
            parts_to_keep = Path(*path_object.parts[5:]) #remove first 4 parts of the path (the Linux drive Map) 
            updated_path_object = Path(drive_map / parts_to_keep) #update the path with the linux path specified in the config file (so we can have different user folders)
            
    
        elif (len(path_object.anchor) == 3) & (path_object.anchor[1]==":"): #we have a windows path 
            parts_to_keep = Path(*path_object.parts[1:]) #remove windows drive from path
            
            updated_path_object = Path(drive_map / parts_to_keep)
            
        else:
            print("Path: " + str(path_object) + " is not a valid windows or linux path...Exiting...")
            sys.exit()
    
    elif sys.platform == "linux":
        path_object_str = str(path_object)
        path_object = Path(path_object_str.replace('\\', '/'))
        
        if path_object.parts[0] == config_file['project']['drivemap']['win32'][0:2]: #check if drive map is the same as the windows drive map specified in the config
            parts_to_keep = Path(*path_object.parts[1:]) #remove windows drive from path
            updated_path_object = Path(drive_map / parts_to_keep)
            
        elif path_object.parts[0] == "/": #a bit brute force, but we have a linux path
            parts_to_keep = Path(*path_object.parts[5:]) #remove first 4 parts of the path (the Linux drive Map) 
            updated_path_object = Path(drive_map / parts_to_keep) #update the path with the linux path specified in the config file (so we can have different user folders)
        else:
            print("Path: " + str(path_object) + " is not a valid windows or linux path...Exiting...")
            sys.exit() 
    else:
        print("Path: " + str(path_object) + " is not a valid windows or linux path...Exiting...")
        sys.exit()      
        
    return(updated_path_object)


## Append Correct Drive Map based on what system is running
def mapSystemDrive(path_object, config_file):
    
    #If we are on Windows
    if sys.platform == "win32":
        
        updated_path_object = mapWindowsDrive(path_object, config_file)
            
    #If we are on Linux
    elif sys.platform == "linux":
        
        updated_path_object = mapLinuxDrive(path_object, config_file)
            
    return(updated_path_object)

def createHillshade(dem_array,azimuth_deg,altitude_deg):

    '''
    Function to generate a hillshade from a DEM
    Parameters
    ----------
    dem_array : 2-D array
        2-D array of the elevation values
    azimuth_deg : int/float
        Azimuth, in degrees, of the light source
    altitude_deg : int/float
        Altitude, in degrees, of the light source

    Returns
    -------
    hillshade : 2-D array
        Hillshade array

    '''
    
    azimuth_deg = 360 - azimuth_deg
    
    azimuthrad = azimuth_deg*np.pi/180.
    altituderad = altitude_deg*np.pi/180.
    
    #Get slope and aspect
    y, x = np.gradient(dem_array)

    slope = np.pi/2. - np.arctan(np.hypot(x,y))
    aspect = np.arctan2(-y, x)

    #Get hillshade
    shaded = np.sin(altituderad)*np.sin(slope) + np.cos(altituderad)*np.cos(slope)*np.cos((azimuthrad - np.pi/2.) - aspect)
    hillshade = 255*(shaded + 1)/2
    
    #Convert array to 8bit integers
    hillshade = np.around(hillshade, decimals=0)
    
    hillshade[np.isnan(dem_array)] = 128 #Code nan values as avg value since nans dont transfer to int
    
    hillshade = hillshade.astype(np.uint8)
    return hillshade




def calcPositionInArray(northing, easting, geotransform):
    '''
    Get the index of a pixel in a geographic array (e.g. from a geotiff) from its northing, easting

    Parameters
    ----------
    northing : float/int
        Coordinate's northing.
    easting : float/int
        Coordinate's easting.
    geotransform : 1D array of float/int
        Geotransform of the array.

    Returns
    -------
    array_position_x : int
        Column index.
    array_position_y : int
        Row index.

    '''
    
    upperl_n = geotransform[3]
    upperl_e = geotransform[0]
    scale = geotransform[1]
    array_position_x = int(np.around((easting - upperl_e)/scale))
    array_position_y = int(np.around((upperl_n - northing)/scale))
    return array_position_x, array_position_y
            
def rmseDict(rmse_length, magnitude_error, signed_angular_error, filename, remove_stat_pixels = False, correlation_mask = False):
        
    mean_me = np.mean(magnitude_error)
    rmse_me = np.sqrt((magnitude_error ** 2).mean())

    mean_ae = np.mean(signed_angular_error)
    rmse_ae = np.sqrt((signed_angular_error ** 2).mean())

    n_obs = np.size(magnitude_error)
    
    if remove_stat_pixels == True:
        if correlation_mask ==True:
            fn = filename[:-4] + "corr_mask_remove_stat_px"
        else:
            fn = filename[:-4] + "_remove_stat_px"
    else:
        if correlation_mask == True:
            fn = filename[:-4] + "corr_mask"
        else:
            fn = filename[:-4]
    dict = {"dataset": fn, 
        "n_obs": n_obs,
            "magnitude_err_rmse": rmse_length,
            "angular_err_rmse": rmse_ae,
            "mean_magnitude_error" : mean_me,
            "mean_angular_error" : mean_ae
}
    return dict
    
    
def calcPositionInArray(northing, easting, geotransform):
    '''
    Get the index of a pixel in a geographic array (e.g. from a geotiff) from its northing, easting

    Parameters
    ----------
    northing : float/int
        Coordinate's northing.
    easting : float/int
        Coordinate's easting.
    geotransform : 1D array of float/int
        Geotransform of the array.

    Returns
    -------
    array_position_x : int
        Column index.
    array_position_y : int
        Row index.

    '''
    
    upperl_n = geotransform[3]
    upperl_e = geotransform[0]
    scale = geotransform[1]
    array_position_x = int(np.around((easting - upperl_e)/scale))
    array_position_y = int(np.around((upperl_n - northing)/scale))
    return array_position_x, array_position_y

def calcCoordinateInArray(row, col, upperl_n, upperl_e, scale):
    '''
    Get the coordinate of a pixel in a geographic array from its row/column indices

    Parameters
    ----------
    row : int
        Row index.
    col : int
        Column index.
    upperl_n : int/float
        Upper lefthand northing of array.
    upperl_e : int/float
        Upper lefthand easting of array.
    scale : int/float
        Scale (in metres) of array.

    Returns
    -------
    array_easting : float/int
        Pixel's easting.
    array_northing : float/int
        Pixel's northing.

    '''
    array_northing = -float(row)*scale + upperl_n
    array_easting = float(col)*scale + upperl_e
    return array_easting, array_northing


def clipArrayToIndices(array, indices):
    '''
    Clip an array to extent defined by indices of the array

    Parameters
    ----------
    array : 2-D array
        Input array.
    indices : 1-D array
        1-D array of [upper left x, upper left y, lower right x, lower right y] indices

    Returns
    -------
    array_clip: 2-D array.
        Clipped array

    '''
    keep_mask = np.zeros([len(array), len(array[0])], dtype=bool)
    keep_mask[indices[1]:indices[3],indices[0]:indices[2]] = True
    
    a_clip_1_d = array[keep_mask]
    array_clip = np.reshape(a_clip_1_d, ((indices[3] - indices[1]), 
                                         (indices[2] - indices[0])))
    
    return array_clip

def clipArraysToExtent(arrays, extent, geotransform):
    '''
    Clip multiple geographic arrays to geographic extent

    Parameters
    ----------
    arrays : List of 2-D arrays
        List containing the 2-D arrays of interest.
    extent : 2-D array
        Array containing the upper left and lower right coordinates as [[upper_left_x, upper_left_y],[lower_right_x, lower_right_y]].
    geotransform : 1D array of float/int
        Geotransform of the array.

    Returns
    -------
    clipped_arrays : List of 2-D arrays
        List containing the clipped 2-D arrays of interest.

    '''
    ul_x, ul_y = calcPositionInArray(extent[0][1], extent[0][0], geotransform)
    lr_x, lr_y = calcPositionInArray(extent[1][1], extent[1][0], geotransform)
    
    crop_index = [ul_x, ul_y, lr_x, lr_y]

    clipped_arrays = []
    for array in arrays:
        array_clipped = clipArrayToIndices(array, crop_index)
        clipped_arrays.append(array_clipped)
    
    return clipped_arrays 

