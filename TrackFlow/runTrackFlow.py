
import argparse
import sys 
import os 
from pathlib import Path 
import yaml 
import datetime as dt 


### Helper Functions
import functions.utils as utils
import functions.tracking as tr 
import functions.summary_stats as ss 
import functions.te_horo_plotting as thplt

def runTrackFlow(config_path, global_overwrite = False):    
    
    #### 0. Read Config File ####
    with open(Path(config_path), 'r') as file:
        config = yaml.safe_load(file)
    
    ### 1. Build Parameter Dictionary ### 
    
    parameter_dictionary = {"list_of_dsms": [utils.mapSystemDrive(Path(config['project']['directory']), config) / Path(config['dsm1']['path']), 
                                             utils.mapSystemDrive(Path(config['project']['directory']), config) / Path(config['dsm2']['path'])],
                            "list_of_orthos": [utils.mapSystemDrive(Path(config['project']['directory']), config) / Path(config['ortho1']['path']), 
                                             utils.mapSystemDrive(Path(config['project']['directory']), config) / Path(config['ortho2']['path'])],
                            "aspect_raster": utils.mapSystemDrive(Path(config['project']['directory']), config) / Path(config['dsm2']['aspectpath']),
                            "list_of_capture_dates": [config['dsm1']['date'], config['dsm2']['date']],
                            "cias_results_folder": utils.mapSystemDrive(Path(config['project']['directory']), config) / Path(config['cias']['folderpath']),
                            "validation_vectors": utils.mapSystemDrive(Path(config['project']['directory']), config) / Path(config['validation']['vectorpath']),
                            "sigma_c": config['parameters']['sigma_c'],
                            "z_sig": config['parameters']['z_sig'],
                            "small_angle": config['parameters']['small_angle'],
                            }
    
    ### 2. Check Parameters ###
    list_of_dsms, list_of_orthos, list_of_capture_dates, cias_results_folder, validation_vectors, sigma_c, z_sig, small_angle, aspect_raster = utils.checkTeHoroParameters(parameter_dictionary)
    
    #### 3. Create Folders for Results ####
    project_directory = utils.mapSystemDrive(Path(config['project']['directory']), config)
    os.makedirs(project_directory / "2_Optical_Flow_Results", exist_ok = True)
    os.makedirs(project_directory / "2_Optical_Flow_Results" / "validation", exist_ok = True)
    os.makedirs(project_directory / "3_CIAS_Results", exist_ok = True)
    os.makedirs(project_directory / "3_CIAS_Results"/ "fullfield", exist_ok=True)
    os.makedirs(project_directory / "3_CIAS_Results"/ "validation", exist_ok=True)
    os.makedirs(project_directory / "4_Validation", exist_ok = True)
    os.makedirs(project_directory / "5_Plots", exist_ok = True)
    
    ### 4. Run Optical Flow ####
    output_dir = project_directory / "2_Optical_Flow_Results"
    
    ## A. RUN OPTICAL FLOW FOR SINGLE HILLSHADE PAIR
    list_of_azimuths = [315]
    list_of_angles = [45]
    debut = dt.datetime.today()  
    print("\n Running Optical Flow on Single Hillshade Pair...")
    tr.runPairwiseOpticalFlow(list_of_dsms, list_of_capture_dates, output_dir, list_of_azimuths, list_of_angles, tag="hs_315_45", dem_proc=True)
    fin = dt.datetime.today()
    print("Done!")
    print("temps d'execution = %fs" % (fin-debut).total_seconds())
    
    ### B. RUN OPTICAL FLOW FOR 24 HILLSHADES
    list_of_azimuths = [0,45,90,135,180,225,270,315]
    list_of_angles = [30,45,60]
    debut = dt.datetime.today()  
    print("\n Running Optical Flow on 24 Hillshade Pairs...")
    tr.runPairwiseOpticalFlow(list_of_dsms, list_of_capture_dates, output_dir, list_of_azimuths, list_of_angles, tag="24_hs", dem_proc=True)
    fin = dt.datetime.today()
    print("Done!")
    print("temps d'execution = %fs" % (fin-debut).total_seconds())
    
    
    ### C. RUN OPTICAL FLOW FOR ORTHO PAIR
    debut = dt.datetime.today()  
    print("\n Running Optical Flow on Ortho Pair...")
    tr.runPairwiseOpticalFlow(list_of_orthos, list_of_capture_dates, output_dir, tag="ortho", dem_proc=False)
    fin = dt.datetime.today()
    print("Done!")
    print("temps d'execution = %fs" % (fin-debut).total_seconds())
    
    ### D. RUN OPTICAL FLOW FOR 168 HILLSHADE PAIRS
    list_of_azimuths = [0, 15, 30,45, 60,75, 90,105, 120,135, 150,165, 180,195, 210,225, 240,255, 270,285,300, 315,330,345]
    list_of_angles = [30,35,40,45,50,55,60]
    debut = dt.datetime.today()  
    print("\n Running Optical Flow on 168 Hillshade Pairs...")
    tr.runPairwiseOpticalFlow(list_of_dsms, list_of_capture_dates, output_dir, list_of_azimuths=list_of_azimuths, list_of_angles=list_of_angles, tag="168_hs", dem_proc=True)
    fin = dt.datetime.today()
    print("Done!")
    print("temps d'execution = %fs" % (fin-debut).total_seconds())
    
    ### 5. Filter optical flow predictions (24 HS, 168 HS) ####
    debut = dt.datetime.today()  
    print("\n Filtering Optical Flow Predictions...")
    pred_to_filter = []
    for file in os.listdir(output_dir):
        if file.endswith("24_hs.tif") | file.endswith("168_hs.tif"):
            pred_to_filter.append(output_dir / file)
    tr.filterOpticalFlowPredictions(pred_to_filter, aspect_raster, sigma_c, z_sig, small_angle)
    print("Done!")
    fin = dt.datetime.today()  
    print("temps d'execution = %fs" % (fin-debut).total_seconds())
    
    ### 6. Filter CIAS results using hillshading method ####
    input_dir = project_directory / "1_Data" / "3_CIAS"
    output_dir = project_directory / "3_CIAS_Results"
    
    # Full Field
    debut = dt.datetime.today()  
    print("\n Filtering CIAS Results on Full Field for 24 Hillshade Pairs...")
    tr.calcAvgHillshadeDisplacements(input_dir / "results_fullfield", output_dir / "fullfield",
                                       "ncc_hillshade_azi", sigma_c, z_sig, small_angle)
    print("Done!")
    fin = dt.datetime.today()  
    print("temps d'execution = %fs" % (fin-debut).total_seconds())
    
    ## Validation Set
    debut = dt.datetime.today()  
    print("\n Filtering CIAS Results on Validation Set for 24 Hillshade Pairs...")
    tr.calcAvgHillshadeDisplacements(input_dir / "results_validation", output_dir / "validation",
                                   "validation_ncc_hillshade", sigma_c, z_sig, small_angle)
    print("Done!")
    fin = dt.datetime.today()  
    print("temps d'execution = %fs" % (fin-debut).total_seconds())
    
    ### 7. Resample sparse CIAS results to full field #### 
    debut = dt.datetime.today()  
    print("\n Resampling Sparse Grids of CIAS predictions to full field...")
    output_dir = project_directory / "3_CIAS_Results"
    
    os.makedirs(output_dir / "fullfield" / "geotiff", exist_ok=True)
    
    
    ## Average Hillshade
    print("Working on Average Hillshade Grid...")
    tr.interpolateSparseDisplacements(output_dir / "fullfield" / "ncc_average_hillshade.csv", "cubic", 
                                       output_dir / "fullfield" / "geotiff", "ncc_average_hillshade", ref_dsm = list_of_dsms[1], filter_on_correlation_coeff = False)
    
    ## Single Hillshade
    print("Working on Single Hillshade Grid...")
    tr.interpolateSparseDisplacements(project_directory / "1_Data" / "3_CIAS" / "results_fullfield" / "ncc_hillshade_azi315_ang45.csv", "cubic", 
                                       output_dir / "fullfield" / "geotiff", "ncc_hillshade_azi315_ang45", ref_dsm = list_of_dsms[1], filter_on_correlation_coeff = False)
    
    ## Ortho
    print("Working on Ortho Grid...")
    tr.interpolateSparseDisplacements(project_directory / "1_Data" / "3_CIAS" / "results_fullfield" / "ncc_ortho_image.csv", "cubic", 
                                       output_dir / "fullfield" / "geotiff", "ncc_ortho_image", ref_dsm = list_of_dsms[1], filter_on_correlation_coeff = False)
    
    ## Average Hillshade Correlation Filter
    print("Working on Average Hillshade Grid with Correlation Filter...")
    tr.interpolateSparseDisplacements(output_dir / "fullfield" / "ncc_average_hillshade.csv", "nearest", 
                                       output_dir / "fullfield" / "geotiff", "ncc_average_hillshade_correlation_filter", ref_dsm = list_of_dsms[1], filter_on_correlation_coeff = True)
    
    ## Average Hillshade - Hillshade Filter
    print("Working on Average Hillshade Grid with Correlation Filter...")
    tr.interpolateSparseDisplacements(output_dir / "fullfield" / "ncc_average_hillshade_filtered.csv", "nearest", 
                                    output_dir / "fullfield" / "geotiff", "ncc_average_hillshade_filtered", ref_dsm = list_of_dsms[1], filter_on_correlation_coeff = False)
    print("Done!")
    fin = dt.datetime.today()  
    print("temps d'execution = %fs" % (fin-debut).total_seconds())

    ### 8. Validation Table Statistics #####
    ### A. OPTICAL FLOW ####
    #Extract validation subsets
    debut = dt.datetime.today()  
    print("\n Writing Validation Statistics to Summary Tables...")
    
    tr.extractAllValidationSubsets(project_directory / "2_Optical_Flow_Results", validation_vectors)
    
    
    ss.summaryTableOpticalFlow(project_directory / "2_Optical_Flow_Results" / "validation", validation_vectors, project_directory / "4_Validation")
    
    ### B. CROSS CORRELATION #### 
    
    ss.summaryTableNCC(project_directory / "3_CIAS_Results" / "validation", 
                       project_directory / "1_Data" / "3_CIAS" / "results_validation", validation_vectors, project_directory / "4_Validation")   
    
    print("Done!")
    fin = dt.datetime.today()
    print("temps d'execution = %fs" % (fin-debut).total_seconds())    
    
    #### 9.  Plots ####
    print("\n Plotting Results...")
    #folder to store results 
    outfolder = project_directory / "5_Plots"

    ### A. 3D VELOCITIES, UNFILTERED - CROPPED EXTENT 
    #Data
    folder = project_directory / "2_Optical_Flow_Results"
    cc_folder = project_directory / "3_CIAS_Results" / "fullfield" / "geotiff"
    
    debut = dt.datetime.today()  
    print("Plotting unmasked 3-D Velocities to clipped extent...")
    thplt.plot3DVelocitiesClippingExtent_3x2([folder / "Optical_Flow_Pair_2018-02-07_2020-02-09_ortho.tif", cc_folder / "ncc_ortho_image.tif",
                                              folder / "Optical_Flow_Pair_2018-02-07_2020-02-09_hs_315_45.tif", cc_folder / "ncc_hillshade_azi315_ang45.tif",
                                              folder / "Optical_Flow_Pair_2018-02-07_2020-02-09_24_hs.tif", cc_folder / "ncc_average_hillshade.tif"], 
                                              [3,1,3,1,3,1], [4,2,4,2,4,2], ["ortho", "ortho","dsm", "dsm", "dsm", "dsm"], ['opticalflow', "ncc", "opticalflow", "ncc", "opticalflow", "ncc"],
                                              [[1226680,5053460],[1229040,5052230]], 
                                              ["A.1 - Optical Flow, Ortho", "A.2 - NCC, Ortho", "B.1 - Optical Flow, Hillshade Pair", "B.2 - NCC, Hillshade Pair",
                                               "C.1 - Optical Flow, 24 Hillshades", "C.3 - NCC, 24 Hillshades" ], list_of_dsms, list_of_orthos, outfolder)
    print("Done!")
    fin = dt.datetime.today()
    print("temps d'execution = %fs" % (fin-debut).total_seconds())

    ### B. MASKED VELOCITIES ####
    debut = dt.datetime.today()  
    print("\n Plotting masked 3-D Velocities...")
    data = [folder / "Optical_Flow_Pair_2018-02-07_2020-02-09_24_hs.tif", 
            cc_folder / "ncc_average_hillshade_filtered.tif", 
            cc_folder / "ncc_average_hillshade_correlation_filter.tif"]
    
    thplt.plotMaskedVelocities_1x3(data, [10,1,1], [11,2,2], ["C.2 - Optical Flow, Error Band Mask", "C.4 - NCC, Error band Mask", "C.5 - NCC, Correlation Mask"], list_of_dsms, outfolder)
    print("Done!")
    fin = dt.datetime.today()
    print("temps d'execution = %fs" % (fin-debut).total_seconds())

    ### C. Error Band Correlation Coefficient Density

    thplt.plotErrorBandCorrCoeffDensity(project_directory / "3_CIAS_Results" / "fullfield" / "ncc_average_hillshade.csv", outfolder)

    ### D. VALIDATION PREDICTION SCATTER ####
    debut = dt.datetime.today()  
    print("\n Plotting validation prediction scatter...")
    thplt.plotValidationPredictionScatter_4x2(validation_vectors, project_directory, outfolder)
    print("Done!")
    fin = dt.datetime.today()  
    print("temps d'execution = %fs" % (fin-debut).total_seconds())



## Arguments ##


parser = argparse.ArgumentParser(description='Run TrackFlow.')
parser.add_argument('-c','--configfile', type=str, help='path to config yaml file')

args = parser.parse_args()

if not os.path.isfile(args.configfile):
    print("File: " + args.configfile + " is an invalid file path. Exiting...")
    sys.exit()


print("Processing Feature Tracking...")
runTrackFlow(args.configfile)


print("DONE. ALL PROCESSING COMPLETE")




