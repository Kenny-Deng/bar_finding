#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 23 13:31:48 2025

@author: qadeng
"""

import pandas as pd
import numpy as np

from savitzky_golay_werrors import savgol_filter_werror as sav
from scipy.signal import savgol_filter as ssav

from scipy.signal import argrelmin as rmin
from scipy.signal import argrelmax as rmax
from astropy.cosmology import LambdaCDM


class Bar_finding(object):
    '''
    Finding bar from the ellipticity profile.
    If you want to find bar from the ellipticity profile of the disk-subtracted image,
    you should initialize this class feeding in the ellipticity profile from the image
    before disk subtraction, and input the disk-subtracted profile in the "find_peak_and_valley"
    function. 
    
    Dependencies:
        
        savitzky_golay_werrors: https://github.com/surhudm/savitzky_golay_with_errors
        scipy
        astropy
        
    Parameters:
        
        csv: The dataframe containing the ellipticity profile. It should be of the same format
        as the isophote object from photutils.Ellipse.
        
        galaxy_ID(int): Galaxy ID.
        
        RA(float): Right ascension of the galaxy (degree).
        
        DEC(float): Declination of the galaxy (degree).
        
        Z(float): Redshift of the galaxy.
    
        pix_scale(float): Pixel scale (arcsec/pixel), default: 0.262 arcsec/pixel.
        
        cosmo(astropy.cosmology.Cosmology object): The adopted cosmology, optional. Default: Flat LambdaCDM with h = 0.7, Omega_m0 = 0.3
    
    Usage:
        
        Bars = Bar_finding(csv_original, galaxy_ID = galaxy_ID, RA = RA, DEC = DEC, Z = Z,)
        peak_results, flags, min_slope = find_peak_and_valley() # This is for finding peaks in the original image
        peak_results_wo_disk, flags_wo_disk, min_slope_wo_disk = find_peak_and_valley(csv = csv_wo_disk, wo_disk = True) # This is for finding peaks in the original image
        
    
    '''
    def __init__(self, csv = None, galaxy_ID = None, RA = None, DEC = None, Z = None,
                 pix_scale = 0.262, cosmo = None, **kwarg):
        
        self.csv = csv
        self.galaxy_ID = galaxy_ID
        self.RA = RA
        self.DEC = DEC
        self.Z = Z
        self.pix_scale = pix_scale
        self.cosmo = cosmo
        
        if self.cosmo is None:
            self.cosmo =  LambdaCDM(70, 0.3, 0.7)
        
        _angular_dist = self.cosmo.angular_diameter_distance(self.Z).value * 1e3 # angular diameter distance in kpc
        self._scale = 1 / 3600 * np.pi / 180 * _angular_dist # kpc/arcsec at the galaxy's redshift
        
        self.csv = self.csv.sort_values(by='sma').reset_index(drop=True)
        flux = self.csv['tflux_e'].to_numpy()
        flux_ratio = flux/np.max(flux)
        R90_id = np.argmin(np.abs(flux_ratio - 0.9))
        self.R90 = self.csv.at[R90_id,'sma']
        self.e90 = self.csv.at[R90_id,'eps']
        self.PA90 = self.csv.at[R90_id,'pa']
        
    def find_peak_and_valley(self, csv = None, PA_thres = 20, e_thres = 0.25, sma_thres = 1.5, 
                             peak_from_gradient = True, valley_lim = True, wo_disk = False):
        '''
        Function that actually does the peak finding.
        The default is for finding peak in the original profile.
        You can input the disk-subtracted csv and set "wo_disk" to True to find peaks in disk-subtracted profile.
        
        Parameters:
            csv: The dataframe containing the ellipticity profile. It should be of the same format
                as the isophote object from photutils.Ellipse.
            
            PA_thres(float): The maximum allowed variation in PA within the bar region (degree), default: 20
            
            e_thres(float): The minimum ellipticity for a bar region, default: 0.25. Note that this threshold 
                is only used in the PA thresholding, it does not imply any cuts in ellipticity!
                
            sma_thres(float): The minimum semi-major axis for structure to be consider as a bar (kpc), default: 1.5
            
            peak_from_gradient(bool): Whether or not to find peak using the gradient profile, default: True
            
            valley_lim(bool): Whether or not to limit the minimum slope search between the peak and the nearby valley, default: True
            
            wo_disk(bool): Is this for disk-subtracted profile? Default: False
            
        Returns:
            
            first_peak: The peak ellipticity corresponding to the bar.
            first_peak_x: The position of the peak ellipticity in pixels.
            first_peak_pa: The position angle of the bar ellipse in degrees.
            
            peak_flag: Whether this profile has a peak (Yes:1,No:0)
            pa_badflag: Whether the potential bar region violates the PA requirement (Yes:1,No:0)
            back_flag: Whether we need to use the minimum slope from the ellipticity profile of the original image (Yes:1,No:0) 
            
            min_grd: Immediate minimum slope of the ellipticity profile beyond the peak.
            min_grd_x: The position of the immediate minimum slope of the ellipticity profile beyond the peak.
            
            
        '''
        if csv is None:
            csv = self.csv
            
        csv = csv.sort_values(by='sma').reset_index(drop=True)
        flux = csv['tflux_e'].to_numpy()
        flux_ratio = flux/np.max(flux)
        R90_id = np.argmin(np.abs(flux_ratio - 0.9))
        R90 = csv.at[R90_id,'sma']
        e90 = csv.at[R90_id,'eps']
        PA90 = csv.at[R90_id,'pa']
        
        # Define the semi-major axis limit within which we try to find ellipticity peaks.
        # Variables that start with 'f' indicate that this limit originates from the uncertainty in flux.
        flim = np.argmin(np.abs(csv['tflux_e']-0.90*csv['tflux_e'].max())) # The Dataframe index corresponding to R90
        flim_x = csv.at[flim,'sma']
        flim_p = csv.at[flim,'eps']
        flim_pa = csv.at[flim,'pa']
        
        # Add a SNR criteria so that we do not extend our search beyond the radius where 
        # the error in ellipticity exceed 0.05 for at least three consecutive ellipses for the first time.
        flagp = len(csv)-1 # The limiting radius
        fids = []
        for i in range(len(csv)-2):
            seg = csv.loc[i:i+3,:].reset_index(drop=True)
            # Restrict ourselves to apply the SNR threshold in the outer region of the galaxy(i.e. at least outside half r90).
            # This radius restriction is purely empirical, as in some disk-subtracted images, sharp spirals would have higher 
            # error in ellipticity. It is just difficult for an ellipse to capture sharp spiral arms embedded in an ambinet 
            # background due to the disjoint feature in the angular luminosity distribution).
            if seg.at[0,'sma'] > 0.5*flim_x:
                if len(seg[seg['ellip_err']>=0.05]) == 3:
                    fids.append(i)
        # Pick the smallest radius, i.e. the first time this SNR threshold is violated.
        if len(fids) > 0:
            flagp = fids[0]
            
        # Variables that start with 'e' indicate that this limit originates from the uncertainty in the shapes of the ellipses.
        elim = flagp
    
        elim_x = csv.at[elim,'sma']
        elim_p = csv.at[elim,'eps']
        elim_pa = csv.at[elim,'pa']
        
        # Set the radius limit to be the smallest one between the limits of "f" origins and that of "e" origins.
        
        if flim_x <= elim_x:
            lim_x = flim_x
            lim_p = flim_p
            lim_pa = flim_pa
            
        else:
            lim_x = elim_x
            lim_p = elim_p
            lim_pa = elim_pa
            
        # Note that here we always compare with the R90 of the galaxy before disk subtraction
        if lim_x > self.R90:
            lim_x = self.R90
            lim_p = self.e90
            lim_pa = self.PA90
            
        
        # Smooth the ellipticity curve with Savitzky-Golay filter and get the gradient profile.
        # You can tune the window length and poly-nomial order at your discretion.
        if len(csv)<9:
            smos = csv['eps'].to_numpy()
            grad = np.zeros_like(smos)
            # We do not smooth or calculate slope profile if there are less than ten data points.
        else:
            try:
                smos = sav(csv['eps'].to_numpy(),9,3,error=csv['ellip_err'])
                grad = sav(csv['eps'].to_numpy(),7,3,error=csv['ellip_err'],deriv=1) 
            except:
                # In some cases the Sav-Gol filter with error will not work, use the conventional Sav_Gol filter instead.
                smos = ssav(csv['eps'].to_numpy(),9,3)
                grad = ssav(csv['eps'].to_numpy(),7,3,deriv=1) 
                
        smas = csv['sma'].to_numpy()
        dx = smas[1:]-smas[:-1]
        dn1 = smas[0]/1.1
        dx = np.insert(dx,0,dn1)
        phy_grd = grad/(dx*self.pix_scale*self._scale) # Get gradient in physical units
        
        csv = csv.assign(smo = smos,grd = phy_grd)
        
        ins = csv[csv['sma']<=lim_x].reset_index(drop=True) # Now focus on the profile within the threshold
        last_id = len(ins)
        smo_ins = smos[:last_id]
        grd_ins = phy_grd[:last_id]
        ins = ins.assign(smo = smo_ins,grd = grd_ins)
        
        ins = ins.reset_index(drop=True)
        
        gradient_crossing_id = []
        for i in range(len(ins)-1):
            if (ins.at[i,'grd'] >= 0)&(ins.at[i+1,'grd'] <= 0):
                gradient_crossing_id.append(i+1)
        
        if not peak_from_gradient:
            
            peak_candit = rmax(np.array(ins['smo']),order=3)
        
            posi_peaks = ins[ins.index.isin(peak_candit[0])]
        
        else:
            peak_candit = gradient_crossing_id
            
            posi_peaks = ins[ins.index.isin(peak_candit)]
        # We do not trust peaks to be bars within 1.5 kpc.
        bar_peaks = posi_peaks[posi_peaks['sma']*self.pix_scale*self._scale>=1.5].sort_values(by='sma').reset_index(drop=True)
        
        if len(bar_peaks) >= 1:
            if len(bar_peaks) == 1:
                first_peak = bar_peaks.at[0,'eps']
                first_peak_x = bar_peaks.at[0,'sma']
                first_peak_pa = bar_peaks.at[0,'pa']
                
                # Profile outside the first peak but still within the threshold.
                osd = ins[ins['sma'] >= first_peak_x].reset_index(drop=True)
                
                peak_flag = 1
                
                
            elif (len(bar_peaks) > 1):
                # Get the over all highest peak, assign the bar as the second highest peak
                # if it fullfil the radius selection and does not have a drastically lower ellipticity.
                
                hid = bar_peaks['eps'].argmax()
                hpx = bar_peaks.at[hid,'sma']
                hpe = bar_peaks.at[hid,'eps']
                hpa = bar_peaks.at[hid,'pa']
                rest_peak = bar_peaks[bar_peaks['sma']<hpx].reset_index(drop=True)
                if len(rest_peak) > 0:
                
                    shid = rest_peak['eps'].argmax()
                    shpeak = rest_peak['eps'].max()
                    shpeak_x = rest_peak.at[shid,'sma']
                    shpeak_pa = rest_peak.at[shid,'pa']
                    if (shpeak_x*self.pix_scale*self._scale >= sma_thres)&(hpe-shpeak<=0.3):
                        first_peak = shpeak
                        first_peak_x = shpeak_x
                        first_peak_pa = shpeak_pa
                    else:
                        first_peak = hpe
                        first_peak_x = hpx
                        first_peak_pa = hpa
                else:
                    first_peak = hpe
                    first_peak_x = hpx
                    first_peak_pa = hpa
                
                
                osd = ins[ins['sma'] >= first_peak_x].reset_index(drop=True)
                
                peak_flag = 1
                
        else:
            first_peak,first_peak_x,first_peak_pa,first_peak_id = [0,0,0,0]
            peak_flag = 0
        
        # Finding the immediate minimum slope
        # If you want to find minimum slope between the peaks and the nearest valley, set "find_valley" to True.
        # If you just want to find the overall minimum slope beyond the peak, set "find_valley" to False (default).
        
        if (peak_flag == 1):
            out_valleys = rmin(np.array(osd['smo']),order=3)
            if len(out_valleys[0]) == 1:
                # If there is only one valley, then this is it.
                valleys = osd[osd.index.isin(out_valleys[0])].reset_index(drop=True)
                valx = valleys.at[0,'sma']
                min_valley = valleys.at[0,'smo']
               
                back_flag = 0 # This is only useful for peak finding in disk-subtracted images.
                              # If "back_flag" is 1, then you should use the minimum slope extracted
                              # from the peak finding process on original galaxy images, since we can
                              # find a legit valley in the disk subtracted profile.
                
            elif len(out_valleys[0]) > 1:
                # If there are more than one valleys, pick the lowest between the first two.
                valleys = osd[osd.index.isin(out_valleys[0])].sort_values(by='sma').reset_index(drop=True)
                fir_val = valleys.at[0,'smo']
                sec_val = valleys.at[1,'smo']
                if sec_val == valleys['smo'].min():
                    valx = valleys.at[1,'sma']
                    min_valley = valleys.at[1,'smo']
                else:
                    valx = valleys.at[0,'sma']
                    min_valley = valleys.at[0,'smo']
                
                back_flag = 0
                
            elif (len(out_valleys[0]) == 0)&(len(osd)>=10)&(not wo_disk):
                # if there is no valley at all, set the radius limit to the minimum value.
                val_seg = osd[osd.index<=9]
                valx = val_seg.at[val_seg['smo'].argmin(),'sma']
                min_valley = val_seg.at[val_seg['smo'].argmin(),'smo']
            
                back_flag = 0
                
            elif (len(out_valleys[0]) == 0)&(len(osd)<10)&(not wo_disk):
                val_seg = osd
                valx = val_seg.at[val_seg['smo'].argmin(),'sma']
                min_valley = val_seg.at[val_seg['smo'].argmin(),'smo']
                
                back_flag = 0
                
            elif (len(out_valleys[0]) == 0)&(wo_disk):

                back_flag = 1 # No valley can be found beyond the peak in disk-subtracted profile.
                valx = osd.at[len(osd)-1,'sma']
            
            
            if valley_lim: # Whether to limit to the gradient profile between peak and the next valley
                grd_seg = osd[osd['sma']<=valx].reset_index(drop=True)
            else:
                grd_seg = osd.reset_index(drop=True)
                
            min_grd = grd_seg['grd'].min() # The minimum slope.
            min_grd_x = grd_seg.at[grd_seg['grd'].argmin(),'sma']
            
        else:
            back_flag = 1
            min_grd = 0
            min_grd_x = 0
            
        
        #### PA flag ####
        inspect = ins[(ins['sma']<=first_peak_x)&(ins['eps']>=e_thres)].reset_index(drop=True)
        if len(inspect) >= 1:
            inspect = inspect.sort_values(by='sma',ascending=False).reset_index(drop=True)
            inspect_bod = inspect.loc[:len(inspect)-1,:]
            delpa = np.max(inspect_bod['pa'])-np.min(inspect_bod['pa'])
            if delpa <= 90:
                delpa = delpa
            else:
                delpa = 180 - delpa
        else:
            delpa = -99 # For those galaxies, there is simply no profile 
                        # fulfilling the requirement to do the PA inspection,
                        # they are generally non-bar galaxies. 
        
        if delpa > PA_thres:
            pa_badflag = 1
        else:
            pa_badflag = 0
            
        
        return (first_peak,first_peak_x,first_peak_pa), (peak_flag, pa_badflag, back_flag), (min_grd_x, min_grd) 


'''
def main():
    
    import os
    import matplotlib.pyplot as plt
    
    df = pd.read_csv('/home/qadeng/DESI_images/galaxy_info.csv')
    df =  df[df.index.isin(np.random.choice(df.index,size=20,replace=False))].reset_index(drop=True)
    
    for i in range(len(df)):
        efit_wd = pd.read_csv('/home/qadeng/DESI_images/efit_wd_csv/%d.csv'%df.at[i,'objID'])
        efit_wod = pd.read_csv('/home/qadeng/DESI_images/efit_wod_csv/%d.csv'%df.at[i,'objID'])
        ra = df.at[i,'ra']
        dec = df.at[i,'dec']
        z = df.at[i,'z']
        gid = df.at[i,'objID']
        Bars = Bar_finding(efit_wd,gid,ra,dec,z)
        peak_res, flags, slps = Bars.find_peak_and_valley() 
        peak_wod, flags_wod, slps_wod = Bars.find_peak_and_valley(csv = efit_wod,wo_disk=True)
        
        if (len(efit_wd) > 9)&(len(efit_wod)>9) :
            try:
                grad_wd = sav(efit_wd['eps'].to_numpy(),7,3,error=efit_wd['ellip_err'],deriv=1) 
                
                grad_wod = sav(efit_wod['eps'].to_numpy(),7,3,error=efit_wod['ellip_err'],deriv=1) 
            except:
                
                grad_wd = ssav(efit_wd['eps'].to_numpy(),7,3,deriv=1) 
                grad_wod = ssav(efit_wod['eps'].to_numpy(),7,3,deriv=1) 
        else:
            grad_wd = np.zeros((len(efit_wd),))
            grad_wod = np.zeros((len(efit_wod),))
            
        cm = 1/2.54  # centimeters in inches
        fig = plt.figure(figsize=(14*cm,16*cm))
        
        gs = fig.add_gridspec(2,1, height_ratios=(6, 2),hspace=0,bottom=0.1,top=0.95)
        
        ax_up = fig.add_subplot(gs[0])
        ax_up.set_box_aspect(1)
        ax_down = fig.add_subplot(gs[1])
        ax_down.set_box_aspect(1/3)

        ax_up.errorbar(efit_wd['sma'],efit_wd['eps'],efit_wd['ellip_err'],fmt='o',color='green')
        ax_up.errorbar(efit_wod['sma'],efit_wod['eps'],efit_wod['ellip_err'],fmt='s',color='red')
        ax_up.legend(['w disk','w/o disk'])
        ax_up.axvline(peak_res[1],ls='--',color='green')
        ax_up.axhline(peak_res[0],ls='--',color='green')
        ax_up.axvline(peak_wod[1],ls=':',color='red')
        ax_up.axhline(peak_wod[0],ls=':',color='red')
        ax_up.set_xlabel('$SMA\,[pix]$')
        ax_up.set_ylabel('e')
        ax_up.set_ylim([np.min([np.min(efit_wd['eps']),np.min(efit_wod['eps'])])-0.025,np.max([np.max(efit_wd['eps']),np.max(efit_wod['eps'])])+0.025])
        
        ax_down.plot(efit_wd['sma'],grad_wd,color='green')
        ax_down.plot(efit_wod['sma'],grad_wod,color='red')
        ax_down.axvline(slps[0],ls='--',color='green')
        # ax_down.axhline(slps[1],ls='--',color='green')
        ax_down.axvline(slps_wod[0],ls=':',color='red')
        # ax_down.axhline(slps_wod[1],ls=':',color='red')
        ax_down.set_xlabel('$SMA\,[pix]$')
        ax_down.set_ylabel('slope')
        
        plt.savefig('/home/qadeng/tests_peak_finding/%d.jpeg'%gid)
        plt.close()
        
if __name__ == "__main__":
 	main()
'''
