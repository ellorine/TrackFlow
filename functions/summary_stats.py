
import functions.utils as utils
import numpy as np
import pandas as pd 
import os
from sklearn.metrics import mean_squared_error

def summaryTableOpticalFlow(validation_prediction_folder, validation_points, outdir):
    
    all = []
    for file in os.listdir(validation_prediction_folder):
        ## Read in data
        prediction_data = pd.read_csv(validation_prediction_folder / file)
        validation_data = pd.read_csv(validation_points)
    
        #Remove observations with NA predictions
        validation_data = validation_data[prediction_data["X_2020_PRED"].notna() & prediction_data["Y_2020_PRED"].notna()]
        prediction_data = prediction_data[prediction_data["X_2020_PRED"].notna() & prediction_data["Y_2020_PRED"].notna()]
    
        #Get mean and angular error for each point    
        prediction_data["u"] = prediction_data['X_2020_PRED'] - prediction_data['X_2018']
        prediction_data["v"] = prediction_data['Y_2020_PRED'] - prediction_data['Y_2018']
        prediction_data["mag"] = np.sqrt(prediction_data["u"]**2 + prediction_data["v"]**2)
        prediction_data["unit_u"] = prediction_data["u"]/prediction_data["mag"]
        prediction_data["unit_v"] = prediction_data["v"]/prediction_data["mag"]
    
        validation_data["u"] = validation_data['X_2020_VAL'] - validation_data['X_2018']
        validation_data["v"] = validation_data['Y_2020_VAL'] - validation_data['Y_2018']
        validation_data["mag"] = np.sqrt(validation_data["u"]**2 + validation_data["v"]**2)
        validation_data["unit_u"] = validation_data["u"]/validation_data["mag"]
        validation_data["unit_v"] = validation_data["v"]/validation_data["mag"]
    
        dot_product = prediction_data["unit_u"]*validation_data["unit_u"] + prediction_data["unit_v"]*validation_data["unit_v"]
    
        ## Get angular error 
        angular_error = np.degrees(np.arccos(dot_product))
    
        all_unit_cross_prod = []
        ## Get sign of angular error
        for i in range(len(prediction_data['u'])):
            pred_vect = [prediction_data.iloc[i]['u'], prediction_data.iloc[i]['v']]
            val_vect = [validation_data.iloc[i]['u'], validation_data.iloc[i]['v']]
            cross_prod = np.cross(val_vect, pred_vect)
            unit_cross_prod = cross_prod / abs(cross_prod)
            all_unit_cross_prod.append(unit_cross_prod)
    
        signed_angular_error = all_unit_cross_prod * angular_error 
    
        ## Get magnitude rmse  
        magnitude_error = prediction_data["mag"] - validation_data['mag']     
        rmse_l = mean_squared_error(validation_data['mag'], prediction_data['mag'], squared=False)
        
        ## Get mean, standard deviation, and RMSE of angular/magnitude error
        dict = utils.rmseDict(rmse_l, magnitude_error, signed_angular_error, file, remove_stat_pixels = False)
        
        all.append(dict)
    
        ## Remove small movements/stationary pixels
        rmse_l = mean_squared_error(validation_data['mag'][validation_data['mag']>2.3], prediction_data['mag'][validation_data['mag']>2.3], squared=False)
        
        signed_angular_error = signed_angular_error[validation_data['mag']>2.3]
        magnitude_error = magnitude_error[validation_data['mag']>2.3]
        
        dict = utils.rmseDict(rmse_l, magnitude_error, signed_angular_error, file, remove_stat_pixels = True)
        
        all.append(dict)
    
    
    
    all_df = pd.DataFrame(all)
    
    ##Write to CSV
    all_df.to_csv(outdir / "Optical_Flow_Validation_Error_Metrics.csv")
    
    return(print("Done! Summary table for optical flow written to csv."))



def summaryTableNCC(validation_prediction_folder, input_data_folder, validation_points, outdir):
    
    all = []
    
    files = []
    for file in os.listdir(validation_prediction_folder): #Get predictions from average hillshade, avg filtered hillshade
        if file.endswith(".csv"):
            files.append(validation_prediction_folder / file)
    for file in os.listdir(input_data_folder): #Get predictions from single hillshade, orthoimage
        if file.endswith("azi315_ang45.csv"):
            files.append(input_data_folder / file)
        elif file.endswith("ortho_image.csv"):
            files.append(input_data_folder / file)
        elif file.endswith("ortho_image_50cm.csv"):
            files.append(input_data_folder / file)
        elif file.endswith("hillshade_multi.csv"):
            files.append(input_data_folder / file)
    
    for file in files:
            
        validation_data = pd.read_csv(validation_points)
        prediction_data = pd.read_csv(file)

        prediction_data['X_2020_PRED'] = prediction_data[' X'] + prediction_data['dx']
        prediction_data['Y_2020_PRED'] = prediction_data['Y'] + prediction_data['dy']

        ## All predictions    
        prediction_data["u"] = prediction_data['X_2020_PRED'] - prediction_data[' X']
        prediction_data["v"] = prediction_data['Y_2020_PRED'] - prediction_data['Y']
        prediction_data["mag"] = np.sqrt(prediction_data["u"]**2 + prediction_data["v"]**2)
        prediction_data["unit_u"] = prediction_data["u"]/prediction_data["mag"]
        prediction_data["unit_v"] = prediction_data["v"]/prediction_data["mag"]

        validation_data["u"] = validation_data['X_2020_VAL'] - validation_data['X_2018']
        validation_data["v"] = validation_data['Y_2020_VAL'] - validation_data['Y_2018']
        validation_data["mag"] = np.sqrt(validation_data["u"]**2 + validation_data["v"]**2)
        validation_data["unit_u"] = validation_data["u"]/validation_data["mag"]
        validation_data["unit_v"] = validation_data["v"]/validation_data["mag"]

        dot_product = prediction_data["unit_u"]*validation_data["unit_u"] + prediction_data["unit_v"]*validation_data["unit_v"]

        angular_error = np.degrees(np.arccos(dot_product))

        all_unit_cross_prod = []
        ## Get sign of angular error
        for i in range(len(prediction_data['u'])):
            pred_vect = [prediction_data.iloc[i]['u'], prediction_data.iloc[i]['v']]
            val_vect = [validation_data.iloc[i]['u'], validation_data.iloc[i]['v']]
            cross_prod = np.cross(val_vect, pred_vect)
            unit_cross_prod = cross_prod / abs(cross_prod)
            all_unit_cross_prod.append(unit_cross_prod)

        signed_angular_error = all_unit_cross_prod * angular_error 

        ## Get magnitude error
        magnitude_error = prediction_data["mag"] - validation_data['mag']
        
        
        ## Remove NAs
        magnitude_error = magnitude_error[prediction_data['dx']!=-9999]
        signed_angular_error = signed_angular_error[prediction_data['dx']!=-9999]
        
        rmse_l = mean_squared_error(validation_data['mag'][prediction_data['dx']!=-9999], prediction_data['mag'][prediction_data['dx']!=-9999],squared=False)
        
        ## Get mean, standard deviation, and RMSE of angular/magnitude error
        filename = file.name
        dict = utils.rmseDict(rmse_l, magnitude_error, signed_angular_error, filename, remove_stat_pixels = False)
        all.append(dict)

        ## Remove small movements/stationary pixels
        magnitude_error = magnitude_error[validation_data['mag']>2.3]
        signed_angular_error = signed_angular_error[validation_data['mag']>2.3]
        
        rmse_l = mean_squared_error(validation_data['mag'][(prediction_data['dx']!=-9999) & (validation_data['mag']>2.3)], 
                                    prediction_data['mag'][(prediction_data['dx']!=-9999) & (validation_data['mag']>2.3)],squared=False)
        
        
        dict = utils.rmseDict(rmse_l, magnitude_error, signed_angular_error, filename, remove_stat_pixels = True)
        all.append(dict)

        ## Correlation Mask (max_cc > 0.6 )
        validation_data = validation_data[prediction_data['max_corrcoeff']>=0.6]
        prediction_data = prediction_data[prediction_data['max_corrcoeff']>=0.6]

        prediction_data["u"] = prediction_data['X_2020_PRED'] - prediction_data[' X']
        prediction_data["v"] = prediction_data['Y_2020_PRED'] - prediction_data['Y']
        prediction_data["mag"] = np.sqrt(prediction_data["u"]**2 + prediction_data["v"]**2)
        prediction_data["unit_u"] = prediction_data["u"]/prediction_data["mag"]
        prediction_data["unit_v"] = prediction_data["v"]/prediction_data["mag"]

        validation_data["u"] = validation_data['X_2020_VAL'] - validation_data['X_2018']
        validation_data["v"] = validation_data['Y_2020_VAL'] - validation_data['Y_2018']
        validation_data["mag"] = np.sqrt(validation_data["u"]**2 + validation_data["v"]**2)
        validation_data["unit_u"] = validation_data["u"]/validation_data["mag"]
        validation_data["unit_v"] = validation_data["v"]/validation_data["mag"]

        dot_product = prediction_data["unit_u"]*validation_data["unit_u"] + prediction_data["unit_v"]*validation_data["unit_v"]

        angular_error = np.degrees(np.arccos(dot_product))

        all_unit_cross_prod = []
        ## Get sign of angular error
        for i in range(len(prediction_data['u'])):
            pred_vect = [prediction_data.iloc[i]['u'], prediction_data.iloc[i]['v']]
            val_vect = [validation_data.iloc[i]['u'], validation_data.iloc[i]['v']]
            cross_prod = np.cross(val_vect, pred_vect)
            unit_cross_prod = cross_prod / abs(cross_prod)
            all_unit_cross_prod.append(unit_cross_prod)

        signed_angular_error = all_unit_cross_prod * angular_error 

        ## Get magnitude error
        magnitude_error = prediction_data["mag"] - validation_data['mag']

        ## Remove NAs
        magnitude_error = magnitude_error[prediction_data['dx']!=-9999]
        signed_angular_error = signed_angular_error[prediction_data['dx']!=-9999]
        
        rmse_l = mean_squared_error(validation_data['mag'][(prediction_data['dx']!=-9999)], 
                                    prediction_data['mag'][(prediction_data['dx']!=-9999)],squared=False)
        
        dict = utils.rmseDict(rmse_l, magnitude_error, signed_angular_error, filename, remove_stat_pixels = False, correlation_mask = True)
        all.append(dict)

        ## Remove small movements/stationary pixels
        magnitude_error = magnitude_error[validation_data['mag']>2.3]
        signed_angular_error = signed_angular_error[validation_data['mag']>2.3]
        
        rmse_l = mean_squared_error(validation_data['mag'][(prediction_data['dx']!=-9999) & (validation_data['mag']>2.3)], 
                                    prediction_data['mag'][(prediction_data['dx']!=-9999) & (validation_data['mag']>2.3)],squared=False)
        
        dict = utils.rmseDict(rmse_l, magnitude_error, signed_angular_error, filename, remove_stat_pixels = True, correlation_mask = True)
        all.append(dict)
    
    
    all_df = pd.DataFrame(all)
    
    ##Write to CSV
    all_df.to_csv(outdir / "Cross_Correlation_Validation_Error_Metrics.csv")
    
    return(print("Done! Summary Table for NCC written to csv."))
    
    
 