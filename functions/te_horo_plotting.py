
import matplotlib.image as mpimg 
import matplotlib.patches as patches 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm

import numpy as np 
import matplotlib
import os 
import matplotlib as mpl 
import pandas as pd 
from scipy.stats import gaussian_kde
import functions.utils as utils
import copy 
from sklearn.metrics import mean_squared_error

#### Plotting Housekeeping ####
widthline = .5
plt.rcParams["font.family"] = "Palatino Linotype"
plt.rcParams["font.size"] = 10
plt.rcParams['axes.linewidth'] = widthline
plt.rcParams['grid.linewidth'] = widthline/2
plt.rcParams['xtick.major.width'] = widthline
#plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.major.width'] = widthline   
#plt.rcParams['ytick.direction'] = 'in'
legfsize = 5

## Get colourmaps for u/v plots
cmap = copy.copy(matplotlib.cm.get_cmap("YlOrBr"))
cmap.set_bad('dimgrey',0.)

## Get colourmaps for hillshade 
cmap_hs = copy.copy(matplotlib.cm.get_cmap("Greys_r"))
cmap_hs.set_bad('white',0.)

def calc3Dvelocity(displacement_data, u_band, v_band, dem_arrays, metadatadict, clipping_extent=None):
    u = utils.importGeotiffToRaster(displacement_data,u_band)
    v = utils.importGeotiffToRaster(displacement_data,v_band)

    ## Clip to Extent
    if clipping_extent: #if clipping extent exists
        clipped_arrays = utils.clipArraysToExtent([u,v],clipping_extent, metadatadict['GeoTransform'])
        u = clipped_arrays[0]
        v = clipped_arrays[1]
    
    elev_start = dem_arrays[0]

    ### Get row / column of displacement vector end
    grid = np.indices((len(elev_start),len(elev_start[0])))
    end_row = grid[0] + np.round((v/2))
    end_col = grid[1] + np.round((u/2))

    ## Convert out of bounds vectors to NA
    end_row[(end_row < 0) | (end_row >= len(end_row))] = np.nan
    end_col[(end_col < 0) | (end_col >= len(end_row[0]))] = np.nan

    elev_end = np.empty_like(elev_start)

    for row in range(len(elev_end)):
        for col in range(len(elev_end[0])):
            r = end_row[row][col]
            c = end_col[row][col]
            if (np.isnan(r) | np.isnan(c)):
                elev = np.nan
            else:
                elev = dem_arrays[1][int(r)][int(c)] #get elevation value for end of displacement vector
            elev_end[row][col] = elev

    dz = elev_end - elev_start

    ## Calculated 3D Magnitude Array
    mag = np.sqrt(u**2 + v**2 + dz**2)

    vel_3d_mm = (mag / 730)*1000 ## Convert to MM 

    vel_3d_mm[(np.isnan(u)) | (np.isnan(v)) | (np.isnan(dz))] = np.nan 
    
    return(vel_3d_mm, u, v)


def plot3DVelocitiesClippingExtent_3x2(data, u_bands, v_bands, data_types, algorithm_types, clipping_extent, subtitles, dsms, orthos, outfolder):
    
    print("Plotting 3-D velocities by clipping extent:" + str(clipping_extent) + "...")
    
    metadatadict = utils.createMetadataDict(dsms[0])
    
    ## 3x2 grid
    fig, axes = plt.subplots(3,2,constrained_layout=True)
    
    ## Dimensions
    fig.set_figheight(6)
    fig.set_figwidth(6)
    
    ## Get Elevations
    dsm1 = utils.importGeotiffToRaster(dsms[0], 1)
    dsm2 = utils.importGeotiffToRaster(dsms[1], 1)

    ## Deal with NaNs and 3.48e38 values
    dsm1[dsm1==-9999]=np.nan
    dsm2[dsm2==-9999]=np.nan
    dsm1[dsm1>4000]=np.nan
    dsm2[dsm2>4000]=np.nan
    dsm1[dsm1<=0]=np.nan
    dsm2[dsm2<=0]=np.nan

    ## Clip DEM Arrays to Extent
    clipped_dem_arrays = utils.clipArraysToExtent([dsm1, dsm2], clipping_extent, metadatadict['GeoTransform'])

    
    ## Plot data ##
    for i,data_to_plot in enumerate(data):
        
        print("Working on 3-D Displacement Plot " + str(i+1) + "...")
        print("Plotting 3-D Velocities...")
        
        #Calculate 3-D velocities mm/day
        vel_3d_mm, u, v = calc3Dvelocity(data_to_plot, u_bands[i], v_bands[i], clipped_dem_arrays, metadatadict, clipping_extent)
        
        elev_start = clipped_dem_arrays[0]
        
        #Create background image 
        hillshade = utils.createHillshade(elev_start, 315, 45)
    
        masked_vel_3d_mm = np.ma.array(vel_3d_mm, mask=np.isnan(vel_3d_mm))
        masked_hillshade = np.ma.array(hillshade, mask=np.isnan(hillshade))
        
        if i<2:
            axplot = axes[0][i]
        elif i<4:
            axplot = axes[1][i-2]
        else:
            axplot = axes[2][i-4]

        #If orthoimage predictions, use ortho as background image...
        if data_types[i]=="ortho":
            bg = utils.importGeotiffToRaster(orthos[0],1)
            bg[np.isnan(bg)] = 0
            bg = np.around(bg,decimals=0)
            bg= bg.astype(np.uint8)
            bg = utils.clipArraysToExtent([bg],clipping_extent, metadatadict['GeoTransform'])[0]
            masked_bg = np.ma.array(bg, mask=np.isnan(bg))   
            backim = axplot.imshow(masked_bg, cmap=cmap_hs)
        else:
            backim = axplot.imshow(masked_hillshade, cmap=cmap_hs)
        
        #Plot 3-D velocities
        im = axplot.pcolormesh(vel_3d_mm,vmin=0, vmax=200,cmap=cmap, alpha=0.7)
        im.set_array(masked_vel_3d_mm.ravel())
    
        ## Plot Sparse Quiver
        ## Subset vectors based on spacing
        print("Plotting Sparse Quiver...")
        
        h, w, = hillshade.shape
    
        nx = int((w) / 40) #spacing = 40
        ny = int((h) / 40)
    
        xsp = np.linspace(0, w - 1, nx, dtype=np.int64)
        ysp = np.linspace(0, h - 1, ny, dtype=np.int64)
        
        if algorithm_types[i]=="opticalflow":
            uplot = u[np.ix_(ysp, xsp)]
            vplot = v[np.ix_(ysp, xsp)]
        else:
            uplot = u[np.ix_(ysp, xsp)]
            vplot = -v[np.ix_(ysp, xsp)]
    
        xsp,ysp = np.meshgrid(xsp, ysp)
    
        ## Remove NaN vectors
        mask = np.logical_and(uplot!= np.nan,vplot!= np.nan)
    
        ## Plot Quiver
        kwargs = {**dict(angles="xy", scale_units="xy", scale = 1, color = 'black',linewidth=2)}    
        axplot.quiver(xsp[mask], ysp[mask], uplot[mask], vplot[mask],**kwargs) #width=0.001, headwidth=4,
        
        ## Axes labels
        if i == 0:
            x_axis = [int(clipping_extent[0][0]+2*len(hillshade[0])*0.2), int(clipping_extent[0][0]+2*len(hillshade[0])*.5), int(clipping_extent[0][0]+2*len(hillshade[0])*.8)]
            y_axis = [int(clipping_extent[1][1]-2*len(hillshade)*0.2), int(clipping_extent[1][1]-2*len(hillshade)*.8)]
            x = [round(len(hillshade[0])*0.2),round(len(hillshade[0])*0.5),round(len(hillshade[0])*0.8)]
            y = [round(len(hillshade)*0.2),round(len(hillshade)*0.8)]
        
            axplot.set_xticks(x)
            axplot.set_yticks(y)
            axplot.set_xticklabels(x_axis, fontsize=6)
            axplot.set_yticklabels(y_axis, fontsize=6, rotation=90)
            axplot.set_xlabel("Easting (m)", fontsize=6)
            axplot.set_ylabel("Northing (m)", fontsize=6)
            
        else:
            axplot.axes.xaxis.set_visible(False)
            axplot.axes.yaxis.set_visible(False)
    
        axplot.set_title(str(subtitles[i]), fontsize=8)
    
    # fig.subplots_adjust(top=0.8)
    # cbar_ax = fig.add_axes([0.15, 0.85, 0.4, 0.05])
    cbar = fig.colorbar(im, ax=axes[:,:],shrink=0.4,location="bottom")
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label(label='3-D Velocity (mm/day)', size=6)
    
    print("Saving figure...")
    plt.savefig(outfolder / str("3D_Velocities_Cropped_Extent" + str(clipping_extent[0][0]) + "_" + 
                str(clipping_extent[0][1]) + "_to_" + str(clipping_extent[1][0]) + "_" + str(clipping_extent[1][1]) + "_3x2.png"), bbox_inches='tight', dpi=300)
    
    return(None)



def plotMaskedVelocities_1x3(data_list, u_bands, v_bands, titles_list, dsm_list, outfolder):
    ## Initialise Plot
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize = (7,3))

    for i, d in enumerate(data_list):
        print("Rendering Map " + str(i+1) + "...")
        filt_u = utils.importGeotiffToRaster(d,u_bands[i])
        filt_v = utils.importGeotiffToRaster(d,v_bands[i])
    
        filt_u[filt_u==-9999]=np.nan
        filt_v[filt_v==-9999]=np.nan
    
        # Get Elevations
        dsm1 = utils.importGeotiffToRaster(dsm_list[0], 1)
        dsm2 = utils.importGeotiffToRaster(dsm_list[1], 1)
    
        ## Deal with NaNs and 3.48e38 values
        dsm1[dsm1==-9999]=np.nan
        dsm2[dsm2==-9999]=np.nan
        dsm1[dsm1>4000]=np.nan
        dsm2[dsm2>4000]=np.nan
        dsm1[dsm1<=0]=np.nan
        dsm2[dsm2<=0]=np.nan
        
        #Get 3-D velocities
        print("Calculating 3-D velocities...")
        metadatadict = utils.createMetadataDict(dsm_list[0])

        vel_3d_mm, u, v = calc3Dvelocity(d, u_bands[i], v_bands[i], [dsm1,dsm2],metadatadict)
        
        hillshade = utils.createHillshade(dsm1, 315, 45)
    
        masked_vel_3d_mm = np.ma.array(vel_3d_mm, mask=np.isnan(vel_3d_mm))
        masked_hillshade = np.ma.array(hillshade, mask=np.isnan(hillshade))
    
        ## Plot Velocity
        col_pos = i
    
        backim = axes[col_pos].imshow(masked_hillshade, cmap=cmap_hs)
        im = axes[col_pos].pcolormesh(vel_3d_mm,vmin=0, vmax=100,cmap=cmap, alpha=0.7)
        im.set_array(masked_vel_3d_mm.ravel())
    
        ##Add Axes Labels
        
        if i == 0 :
            x_axis = [int(metadatadict['GeoTransform'][0]+2*len(hillshade[0])*0.2), int(metadatadict['GeoTransform'][0]+2*len(hillshade[0])*.5), int(metadatadict['GeoTransform'][0]+2*len(hillshade[0])*.8)]
            y_axis = [int(metadatadict['GeoTransform'][3]-2*len(hillshade)*0.2), int(metadatadict['GeoTransform'][3]-2*len(hillshade)*.8)]
            x = [round(len(hillshade[0])*0.2),round(len(hillshade[0])*0.5),round(len(hillshade[0])*0.8)]
            y = [round(len(hillshade)*0.2),round(len(hillshade)*0.8)]
        
            axes[col_pos].set_xticks(x)
            axes[col_pos].set_yticks(y)
            axes[col_pos].set_xticklabels(x_axis, fontsize=6)
            axes[col_pos].set_yticklabels(y_axis, fontsize=6, rotation=90)
            axes[col_pos].set_xlabel("Easting (m)", fontsize=6)
            axes[col_pos].set_ylabel("Northing (m)", fontsize=6)
        else:
            axes[col_pos].xaxis.set_visible(False)
            axes[col_pos].yaxis.set_visible(False)

        axes[col_pos].set_title(str(titles_list[i]), fontsize=8)
    
    cbar = fig.colorbar(im, orientation='horizontal', shrink=0.4, ax=axes[0:3])
    cbar.ax.tick_params(labelsize=6)
    cbar.set_label(label='3-D Velocity (mm/day)', size=6)
    
    plt.savefig(outfolder / "MaskedVelocities_1x3.png", dpi=300, bbox_inches='tight')
    return(None)
        
        

#### VALIDATION PLOT ####

def plotyeqxline(ax):
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
        np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
    ]
    
    # now plot both limits against eachother
    ax.plot(lims, lims, 'k-', alpha=0.5, zorder=0)

    


def plotValidationPredictionScatter_4x2(validation_data, project_dir, outfolder):
    

    #ortho data
    ortho_ncc = pd.read_csv(project_dir / "1_Data" / "3_CIAS" / "results_validation" / "validation_ncc_ortho_image_50cm.csv")
    ortho_of = pd.read_csv(project_dir / "2_Optical_Flow_Results" / "validation"/ "Validation_Subset_ortho_50cm.csv")

    #hs data
    hs_ncc = pd.read_csv(project_dir / "1_Data" / "3_CIAS" / "results_validation" / "validation_ncc_hillshade_multi.csv")
    hs_of = pd.read_csv(project_dir / "2_Optical_Flow_Results" / "validation"/ "Validation_Subset_multi_hs.csv")

    #24hs 
    hs24_ncc = pd.read_csv(project_dir / "3_CIAS_Results" / "validation" / "ncc_average_hillshade.csv")
    hs24_ncc_filt = pd.read_csv(project_dir / "3_CIAS_Results" / "validation" / "ncc_average_hillshade_filtered.csv")
    hs24_of = pd.read_csv(project_dir / "2_Optical_Flow_Results" / "validation"/ "Validation_Subset_24_hs.csv")
    hs24_of_filt = pd.read_csv(project_dir / "2_Optical_Flow_Results" / "validation"/ "Validation_Subset_24_hs_filtered.csv")

    #168 hs
    hs168_of = pd.read_csv(project_dir / "2_Optical_Flow_Results" / "validation"/ "Validation_Subset_168_hs.csv")
    hs168_of_filt = pd.read_csv(project_dir / "2_Optical_Flow_Results" / "validation"/ "Validation_Subset_168_hs_filtered.csv")

    ## Create column demarkateing when points are filtered by hillshade/correlation
    ortho_ncc['CorrelationFilter'] = 0
    ortho_ncc.loc[ortho_ncc.max_corrcoeff <0.6, "CorrelationFilter"] = 1 
    
    hs_ncc['CorrelationFilter'] = 0
    hs_ncc.loc[hs_ncc.max_corrcoeff <0.6, "CorrelationFilter"] = 1 

    hs24_ncc['HillshadeFilter'] = 0
    hs24_ncc.loc[hs24_ncc_filt.dx == -9999, 'HillshadeFilter'] = 1
    
    hs24_ncc['CorrelationFilter'] = 0 
    hs24_ncc.loc[hs24_ncc.max_corrcoeff <0.6, "CorrelationFilter"] = 1 

    hs24_ncc['Filter'] = 0
    hs24_ncc.loc[hs24_ncc.CorrelationFilter ==1, 'Filter'] = 1
    hs24_ncc.loc[hs24_ncc.HillshadeFilter==1, 'Filter'] = 2
    hs24_ncc.loc[(hs24_ncc.CorrelationFilter==1) & (hs24_ncc.HillshadeFilter==1), 'Filter'] = 3

    hs24_of['HillshadeFilter'] = 0
    hs24_of.loc[hs24_of_filt.X_2020_PRED.isna(), 'HillshadeFilter'] = 1
    
    hs168_of['HillshadeFilter'] = 0
    hs168_of.loc[hs168_of_filt.X_2020_PRED.isna(), "HillshadeFilter"] = 1

    #read in validation data
    validation_dataset = pd.read_csv(validation_data)

    ## Calculate Displacement Magnitudes
    validation_dataset['magnitude'] = np.sqrt((validation_dataset['X_2020_VAL']-validation_dataset['X_2018'])**2 + (validation_dataset['Y_2020_VAL']- validation_dataset['Y_2018'])**2)
    ortho_of['length'] =  np.sqrt(ortho_of['u']**2 + ortho_of['v']**2)
    hs_of['length'] = np.sqrt(hs_of['u']**2 + hs_of['v']**2)
    hs24_of['length'] = np.sqrt(hs24_of['u']**2 + hs24_of['v']**2)
    hs168_of['length'] = np.sqrt(hs168_of['u']**2 + hs168_of['v']**2)
    
    hs24_ncc['length'] = np.sqrt(hs24_ncc['dx']**2 + hs24_ncc['dy']**2)
    ortho_ncc['length'] = np.sqrt(ortho_ncc['dx']**2 + ortho_ncc['dy']**2)
    hs_ncc['length'] = np.sqrt(hs_ncc['dx']**2 + hs24_ncc['dy']**2)

    
    fig, axes = plt.subplots(4,2,figsize = (6,7), constrained_layout=True)
    
    ## Plot 1 - ortho optical flow
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mpatches
    
    cmap = ListedColormap(['#1E88E5', '#D81B60', "#FFC107", '#004D40']) #no filter, correlation filter, hillshade filter, both filter
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
    
    axes[0][0].scatter(x=validation_dataset['magnitude'][validation_dataset['magnitude']<150], y=ortho_of['length'][validation_dataset['magnitude']<150], 
                       c='#1E88E5', marker='.', alpha=0.5)
            
    axes[0][0].text(0,100, r'$RMSE=18.46$',fontsize=8,bbox=props)
    
    axes[0][0].set_title("A.4 - Optical Flow, Ortho (50 cm)", fontsize = 10)
    plotyeqxline(axes[0][0])
    axes[0][0].set_xlabel("Validation Displacement (m)", fontsize=8)
    axes[0][0].set_ylabel("Predicted Displacement (m)", fontsize=8)
    axes[0][0].tick_params(labelsize=8)
    axes[0][0].tick_params(labelsize=8)
    
    
    axes[0][1].scatter(x=validation_dataset['magnitude'][validation_dataset['magnitude']<150], 
                       y=ortho_ncc['length'][validation_dataset['magnitude']<150], c=ortho_ncc['CorrelationFilter'][validation_dataset['magnitude']<150], 
                       alpha=0.5, cmap = ListedColormap(['#1E88E5', '#D81B60']), marker='.')
    axes[0][1].set_title("A.5-6 - NCC, Ortho (50 cm)", fontsize = 10)
    axes[0][1].text(0,100,r'$RMSE=18.85$' +
                    " (A.5), " + r"$6.96$" + " (A.6)", fontsize=8,bbox=props)
                   
                    
                    # "RMSE: " + str(np.round(rmse_all,2)) + (" (A.2), " + str(np.round(rmse_masked,2)) + " (A.3)"), fontsize=8)
    axes[0][1].xaxis.set_visible(False)
    axes[0][1].yaxis.set_visible(False)
    plotyeqxline(axes[0][1])
    
    axes[1][0].scatter(x=validation_dataset['magnitude'][validation_dataset['magnitude']<150], 
                       y=hs_of['length'][validation_dataset['magnitude']<150], c='#1E88E5', alpha=0.5, marker='.')
    axes[1][0].set_title("B.4 - Optical Flow, Multidirectional Hillshade", fontsize = 10)
    rmse_all = mean_squared_error(validation_dataset['magnitude'], hs_of['length'],squared=False)
    axes[1][0].text(0,100, r'$RMSE=16.08$',fontsize=8,bbox=props)
    axes[1][0].xaxis.set_visible(False)
    axes[1][0].yaxis.set_visible(False)
    plotyeqxline(axes[1][0])
    
    axes[1][1].scatter(x=validation_dataset['magnitude'][validation_dataset['magnitude']<150], y=hs_ncc['length'][validation_dataset['magnitude']<150], 
                       c=hs_ncc['CorrelationFilter'][validation_dataset['magnitude']<150], alpha=0.5, marker='.',cmap = ListedColormap(['#1E88E5', '#D81B60']))
    axes[1][1].set_title("B.5-6 - NCC, Multidirectional Hillshade", fontsize = 10)
    axes[1][1].text(0,100,r'$RMSE=16.69$' +
                    " (B.5), " + r'$5.47$' + " (B.6)", fontsize=8,bbox=props)            
    axes[1][1].xaxis.set_visible(False)
    axes[1][1].yaxis.set_visible(False)
    plotyeqxline(axes[1][1])
    
    axes[2][0].scatter(x=validation_dataset['magnitude'][validation_dataset['magnitude']<150], 
                       y=hs24_of['length'][validation_dataset['magnitude']<150], c=hs24_of['HillshadeFilter'][validation_dataset['magnitude']<150], alpha=0.5, marker='.',cmap = ListedColormap(['#1E88E5', "#FFC107"]))
    axes[2][0].set_title("C.1-2 - Optical Flow, 24 Hillshades", fontsize = 10)
    axes[2][0].text(0,100,r'$RMSE=16.90$' +
                    " (C.1), " + r'$4.54$' + " (C.2)", fontsize=8,bbox=props)           
    axes[2][0].xaxis.set_visible(False)
    axes[2][0].yaxis.set_visible(False)
    plotyeqxline(axes[2][0])
    
    
    axes[2][1].scatter(x=validation_dataset['magnitude'][validation_dataset['magnitude']<150], 
                       y=hs24_ncc['length'][validation_dataset['magnitude']<150], c=hs24_ncc['Filter'][validation_dataset['magnitude']<150], alpha=0.5, marker='.', cmap = cmap)
    axes[2][1].set_title("C.3-5 - NCC, 24 Hillshades", fontsize = 10)
    axes[2][1].text(0,100,r'$RMSE=16.81$' +
                    " (C.3), " + r'$3.92$' + " (C.4) " + 
                    r'$6.67$' + " (C.5)", fontsize=8,bbox=props)    

    axes[2][1].xaxis.set_visible(False)
    axes[2][1].yaxis.set_visible(False)
    plotyeqxline(axes[2][1])
    
    axes[3][0].scatter(x=validation_dataset['magnitude'][validation_dataset['magnitude']<150], 
                       y=hs168_of['length'][validation_dataset['magnitude']<150], c=hs168_of['HillshadeFilter'][validation_dataset['magnitude']<150], marker='.', alpha=0.5, cmap = ListedColormap(['#1E88E5', "#FFC107"]))
    axes[3][0].set_title("D.1-2 - Optical Flow, 168 Hillshades", fontsize = 10)
    axes[3][0].text(0,100,r'$RMSE=16.80$' +
                    " (D.1), " + r'$3.72$' + " (D.2)", fontsize=8,bbox=props) 
    axes[3][0].xaxis.set_visible(False)
    axes[3][0].yaxis.set_visible(False)
    plotyeqxline(axes[3][0])
    
    # Clear bottom-right ax
    bottom_right_ax = axes[-1][-1] 
    bottom_right_ax.set_axis_off()  # removes the XY axes
    
    # Manually create legend handles (patches)
    unmasked_patch = mpatches.Patch(color='#1E88E5', label='Unmasked')
    corr_patch = mpatches.Patch(color='#D81B60', label='Masked by Correlation')
    hs_patch = mpatches.Patch(color='#FFC107', label='Masked by Error Band')
    both_patch = mpatches.Patch(color='#004D40', label='Masked by Correlation & Error Band')
    
    # Add legend to bottom-right ax
    bottom_right_ax.legend(handles=[unmasked_patch, corr_patch, hs_patch, both_patch], loc='center', fontsize=10)
    
    
    plt.savefig(outfolder / "validation_prediction_scatter.png", dpi=300,bbox_inches='tight' )
    
    return(print("Done! Figure saved to plots folder"))


def plotErrorBandCorrCoeffDensity(file, outfolder):

    array = pd.read_csv(file)
    maxcc = array['max_corrcoeff']
    ee = array['ellipse_err']
    logee = array['log_error_band']


    # Calculate the point density
    xy = np.vstack([ee,maxcc])
    z = gaussian_kde(xy)(xy)
    
    x2y = np.vstack([logee,maxcc])
    z2 = gaussian_kde(x2y)(x2y)
    
    plt.rcParams["figure.figsize"] = (10,4)
    
    fig, ax = plt.subplots(nrows=1, ncols=2)
    
    density = ax[0].scatter(ee, maxcc, c=z, s=5, cmap=cm.plasma)
    ax[0].set_xlim([0, 5])
    text1 = r'$r=%.2f$' % (np.corrcoef(ee,maxcc)[0][1], )
    text2 = r'$r=%.2f$' % (np.corrcoef(logee,maxcc)[0][1], )
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
    
    ax[0].text(0.75, 0.95, text1, transform=ax[0].transAxes, fontsize=8,
            verticalalignment='top', bbox=props)
    
    logdensity = ax[1].scatter(logee, maxcc, c=z2, s=5, cmap=cm.plasma)
    ax[1].text(0.75, 0.95, text2, transform=ax[1].transAxes, fontsize=8,
            verticalalignment='top', bbox=props)
    ax[0].set_xlabel(r"$\varepsilon_{c}$")
    ax[0].set_ylabel(r"$\rho_{max}$")
    ax[1].set_xlabel(r"$log(\varepsilon_{c})$")
    ax[1].set_ylabel(r"$\rho_{max}$")
    
    cbar = fig.colorbar(density, ax=ax[0:2])
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(label='Point Density',fontsize=8)
    
    plt.savefig(outfolder /  "errorband_corrcoeff_density.png", dpi = 300, bbox_inches='tight')
    return(print("Done!"))


