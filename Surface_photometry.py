#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 18 18:23:57 2025

@author: kenny
"""

import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from astropy.cosmology import LambdaCDM

from photutils.isophote import EllipseGeometry
from photutils.aperture import EllipticalAperture
from photutils.isophote import Ellipse

class Surface_photometry(object):
    
    '''
    This is a module to analyse the surface photometry on a single galaxy image.
    There are two major functions -- 1. efit: performing the ellipse fitting;
    2. profile_fit: fit the surface brightness profile obtained from ellipse
    fitting, build a 2D disk model and subtract it from the galaxy image.
    
    It can be easily integrated into a multi-process program to deal with
    a large amount of galaxies.
    
    The definition of ellipticity is e = 1 - b/a, where a is the semi-major axis and 
    b is the semi-minor axis.
    
    Parameters:
        
        image(numpy.array): The input image.
        
        psf(numpy.array): PSF of the corresponding input image, optional. If not provided, the 2D disk model will not be convolved with a PSF.
        
        galaxy_ID(int): Galaxy ID.
        
        RA(float): Right ascension of the galaxy (degree).
        
        DEC(float): Declination of the galaxy (degree).
        
        Z(float): Redshift of the galaxy.
        
        r_bulge(float): Scale radius of the bulge component (kpc).
        
        r_disk(float): Scale radius of the disk component (kpc).
        
        e_disk(float): Ellipticity of the disk component, optional.
        
        PA_disk(float): Position angle of the disk component (radian), optional.
        
        pix_scale(float): Pixel scale (arcsec/pixel), default: 0.262 arcsec/pixel.
        
        cosmo(astropy.cosmology.Cosmology object): The adopted cosmology, optional. Default: Flat LambdaCDM with h = 0.7, Omega_m0 = 0.3
        
    Usage:
        
        To run the whole process:
        
            phot = Surface_photometry(image = img, psf = psf, output_path = 'path',
                                     galaxy_ID = galaxy_ID, RA = RA, DEC = DEC, 
                                     Z = Z, r_bulge = r_bulge, r_disk = r_disk,
                                     e_disk = e_disk, PA_disk = PA_disk)
        
            # Note that the morphological parameters(i.e. r_bulge, r_disk, e_disk and PA_disk) are important
            # if you want to build the 2D disk model and do disk subtraction. 
            # I recommend using Simard+10 2D bulge+disk decomposition results for r_bulge and r_disk, but be cautious
            # about the PA_disk as the axis used in Simard+10 is not exactly the same as in DESI.
            # There is an option in the efit function("update_disk = True") that allows you to change the e_disk
            # and PA_disk to the value inferred from the ellipse fitting process(in this case e_disk and PA_disk would 
            # be measured from the ellipse at R90).
            # In principle, you can also infer r_disk from the ellipse fitting result, but I did not include this option.
            
            efit_result = phot.efit() # The e-fitting result on the original image.
            disk_model, disk_sub_image = phot.profile_fit() # Fitting the surface brightness profile and perform the disk subtraction. 
            efit_wo_disk_result = phot.efit(wo_disk=True, update_disk=False) # Run ellipse fitting on the disk-subtracted image.
        
        
        You can also invoke profile_fit/efit directly providing the "csv"/"image" argument. This is mainly used for fine tuning the 
        fitting process.
        
            phot.profile_fit(csv = Dataframe object containing the ellipse fitting results)
            phot.efit(image = Image you wish to perform ellipse fitting)
        
    '''
    
    def __init__(self, image = None, psf = None, output_path = None,
                 galaxy_ID = None, RA = None, DEC = None, Z = None, 
                 r_bulge = None, r_disk = None, e_disk = None, PA_disk = None,
                 pix_scale = 0.262,
                 cosmo = None, **kwargs):
        
        self.image = image
        self.psf = psf
        self.output_path = output_path
        
        self.galaxy_ID = galaxy_ID
        self.RA = RA
        self.DEC = DEC
        self.Z = Z
        self.r_bulge = r_bulge
        self.r_disk = r_disk
        
        self.pix_scale = pix_scale
        
        self.e_disk = e_disk
        self.PA_disk = PA_disk
        
        
        self.cosmo = cosmo
        
        self.phot_dat = None  # ellipse fitting result.
        self.R90 = None # R90 in pixels.
        self.R90_kpc = None # R90 in kpc.
        self.disk_sub_factor = None  # a factor that determines how much disk is subtracted.
        self.chsq = None # Chi squared per ddof of the 1D surface brightness profile fits.
        self.disk_model = None # 2D model of the disk component.
        self.image_wo_disk = None # the resulting disk-subtracted galaxy image.
        
        if self.cosmo is None:
            self.cosmo =  LambdaCDM(70, 0.3, 0.7)
        
        print('Cosmology: LambdaCDM h = %.2f, Om0 = %.2f, Ode0 = %.2f.'%(self.cosmo.h,self.cosmo.Om0,self.cosmo.Ode0))
        #print(self.psf)
        print('Processing Galaxy ID:%d'%self.galaxy_ID)
        
    def efit(self, image = None, iteration = True, update_disk = True, minsma = None, maxsma = None, fix_center = False, maxgerr = 5, sclip = 2, nclip = 1, 
             e_cen = 0.2, pa_cen = np.pi/2, sma_cen = 1/0.262, save = True, wo_disk = False, **kwargs):
        
        '''
        Perform ellipse fitting to the input image.
        This is essentially a wrapper around the photutils.Ellipse, please check
        https://photutils.readthedocs.io/en/stable/api/photutils.isophote.Ellipse.html#photutils.isophote.Ellipse
        for details.
        The default values should apply to most galaxies, but you should tailor them
        according your needs.
        
        Parameters:
            
            image(float): The input image for ellipse fitting, default: None, in this case it automaticly use self.image,
            or self.image_wo_disk if "wo_disk = True".
            
            iteration(bool): Whether or not to enable an iterative fitting process in case the simple fit 
            fails, default: True. Note that enabling this option could slow down the whole process.
            
            update_disk(bool): Whether or not to update the disk morphology using the ellipse at R90. This would 
            affect the 2D disk model and subsequently the disk-subtracted image, default: True.
            
            minsma(float): The minimum value for the semimajor axis length (pixel), optional.
            
            maxsma(float): The maximum value for the semimajor axis length (pixel), optional.
            
            fix_center(bool): Keep center of ellipse fixed during fit? The default is False.
            
            maxgerr(float): The maximum acceptable relative error in the local radial intensity gradient, default: 5.
            This is the main control for preventing ellipses to grow to regions of too low signal-to-noise ratio. 
            Please read the documentation of Ellipse carefully.
            
            sclip(float): The sigma-clip sigma value, default: 2
            
            nclip(float): The number of sigma-clip iterations, default: 1. If set to zero, do not perform sigma-clipping.
            
            e_cen(float): The initial guess of the ellipticity of the centroid, default: 0.2. ~\(OvO)/~ wu hu~
        
            pa_cen(float): The initial guess of the position angle of the centroid (radian), default: np.pi/2.
            
            sma_cen(float): The semi-major axis of the centroid (pixel), default: 1/self.pix_scale.
            
            save(bool): Whether or not to save the outputs, default: True.
            
            wo_disk(bool): Whether this is meant for disk-subtracted galaxy image, default: False
            
        Usage:
            
            efit_result = phot.efit()
            
            If the input is the disk-subtracted image:
            efit_result = phot.efit(wo_disk=True, update_disk=False)
            
        '''
        
        
        if wo_disk is True:
            if image is None:
                image = self.image_wo_disk
            else:
                pass
            
            image = np.where(image < 0, 0, image) # Dealing with negative pixels if there is any.
            
            self.fix_center = False
            self.maxgerr = 10
            self.sclip = 3
            #self.maxsma = self.R90
            # These above parameters are tuned for disk-subtracted images, they just loosen the allowed error.
            # You may want to adjust them.
        
        else:
            if image is None:
                image = self.image
            else:
                pass
            
        
        if minsma is None:
            minsma = 1 / self.pix_scale # if no minsma is provided, set to default: 1 arcsec.
        
        
        # Find the central pixel.
        if len(image)%2 == 1:
            center = [int((len(image)-1)/2),int((len(image)-1)/2)]
        else:
            center = [int((len(image))/2),int((len(image))/2)]
        
        # Find the centroid, you can change the init_geom parameter if you are not satisfied by the centroid finding result.
        try:
            init_geom = EllipseGeometry(center[1],center[0],eps = 0.5,pa = np.pi/2,sma = 1) # EllipseGeometry takes xy coordinates.  
            init_elps = Ellipse(image,init_geom)
            init_iso = init_elps.fit_isophote(sma_cen) # Fit the centroid geometry at this radius(pixel).
            tab = init_iso.to_table()
        except Exception:
            tab = pd.DataFrame({'sma':[]})
            print('Fail to fit the geometry of the centroid, switch to the default geometry.')
            pass
        
        if len(tab['sma'])>0:
            # Define the initial geometry as the centroid .
            new_geom = EllipseGeometry(tab['x0'][0],tab['y0'][0],eps = tab['ellipticity'][0],pa = tab['pa'][0].value,sma = tab['sma'][0])
        else:
            # Try a more general initial geometry(at your discretion).
            new_geom = EllipseGeometry(center[1],center[0],eps = e_cen,pa = pa_cen,sma = sma_cen)
            
        new_elps = Ellipse(image,new_geom)
        
        # Perform ellipse fitting to the image, using the above geometry.
        try:
            isophote_list = new_elps.fit_image(minsma = minsma,fix_center=fix_center,maxgerr=maxgerr,sclip=sclip,nclip=nclip) 
            # Tread carefully with the degree of sigma-clipping as it allows the fit to continue at low SNR region.
            tab_isophote = isophote_list.to_table(['sma','eps','ellip_err','intens','int_err','rms','pa','pa_err','tflux_e','grad','grad_error','grad_r_error','stop_code'])
            # Please check the exact meaning of each column in the photutils.Ellipse documentation.
            if len(tab_isophote) > 0:
                df_isophote = tab_isophote.to_pandas()
            else:
                df_isophote = pd.DataFrame([])
                
        except Exception:
            df_isophote = pd.DataFrame([])
            print('Fitting with the inferred geometry failed.')
            pass
        
        # Iterative fitting: Looping over a range of initial geometry to achieve successful fits. It may take a while.
        if (len(df_isophote) == 0) & iteration:
            print('Try iterative fitting.')
            
            eps_arr = np.linspace(0.01,0.99,10) 
            sma_fit = 1/0.262
            pa_arr = np.linspace(0,179,10)*np.pi/180
            isophote_tables = []
            for i in range(len(pa_arr)):
                for j in range(len(eps_arr)):
                    temp_geom = EllipseGeometry(center[1],center[0],sma = sma_fit,pa=pa_arr[i],eps=eps_arr[j])
                    temp_elps = Ellipse(image,temp_geom)
                    try:
                        temp_isolist = temp_elps.fit_image(minsma = minsma,fix_center=fix_center,maxgerr=maxgerr*1.5,sclip=sclip,nclip=nclip) # Loosen the maximum tolarent on the error of the intensity gradient
                        temp_tab = temp_isolist.to_table(['sma','eps','ellip_err','intens','int_err','rms','pa','pa_err','tflux_e','grad','grad_error','grad_r_error','stop_code'])
                        isophote_tables.append(temp_tab)
                    except Exception:
                        pass
        
            # There may be more than 1 results from the above, we take the most frequent profile to be the fiducial result.
            # Note: I assume the true/correct ellipticity profile would be recovered most frequently, and the erroneous
            # profiles would be the ones that stops prematurely or extends way beyond the galaxy.
            profile_lengths = []
            for k in range(len(isophote_tables)):
                temp_tab = isophote_tables[k]
                if len(temp_tab) != 0:
                    profile_lengths.append(len(isophote_tables[k]))
            
            if len(profile_lengths) == 0: 
                # This is the case where no fit is available exhausting all our initial guesses.
                # We simply throw this galaxy away.
                print('No available fit to %d'%self.galaxy_ID)
                
                df_isophote = pd.DataFrame([])
            
            else:        
                # Pick out the most frequent profile.
                freqt_length = st.mode(profile_lengths).mode[0]
                indice_fiducial = [x for x in range(len(profile_lengths)) if profile_lengths[x] == freqt_length]  
                table_fiducial = isophote_tables[indice_fiducial[0]]
                
                df_isophote = table_fiducial.to_pandas()
            
        else:
            pass
        
        self.phot_dat = df_isophote # This is the efit result. 
        
        if (save) & (self.output_path is not None) & (wo_disk is False):
            df_isophote.to_csv(self.output_path+'%d'%self.galaxy_ID+'_'+'efit_result.csv')
        elif (save) & (self.output_path is not None) & (wo_disk is True):
            df_isophote.to_csv(self.output_path+'%d'%self.galaxy_ID+'_'+'efit_result_disk_sub.csv')
        elif (save) & (self.output_path is None):
            print('Please specify the path to save results.')
            raise
        
        if update_disk:
            # Update the disk ellipticity and position angle using the median values
            # within 5 consecutive isophotes centered on R90.
            csv = self.phot_dat.sort_values(by='sma').reset_index(drop=True)
            flux = csv['tflux_e'].to_numpy()
            flux_ratio = flux/np.max(flux)
            R90_id = np.argmin(np.abs(flux_ratio - 0.9))
            
            self.e_disk = np.median(csv.loc[R90_id-2:R90_id+2,'eps'])
            self.PA_disk = np.deg2rad(np.median(csv.loc[R90_id-2:R90_id+2,'pa']))
            # You can also calculate the r_disk and update it as well.
            
        return df_isophote
    
    def _deV_profile(self,r,Ie):
        b4 = 7.669
        I = Ie*np.exp(-1*b4*((r/self.r_bulge)**(1/4)-1))
        return I
    
    def _exponential_profile(self,r,Id):
        I = Id*np.exp(-1*r/self.r_disk)
        return I
    
    def _composite_profile(self,r,Ie,Id):
        b4 = 7.669
        I = Ie*np.exp(-1*b4*( (r/self.r_bulge)**(1/4)-1 ))+Id*np.exp(-1*r/self.r_disk)
        return I
    
    # Chi squared per ddof
    def _chi_squared(self,L,r,obs,err):
        Ie,Id = L[:]
        I = self._composite_profile(r, Ie, Id)
        chisq = np.sum((I-obs)**2/err**2)/(len(obs)-1)
        if np.isfinite( chisq ):
            return chisq
        else:
            return np.inf
    
    # 2D exponential profile
    def _expon_2d(self, x = None, y = None, r_d = None, eps = None, theta = None, x_0 = None, y_0 = None, amp = None):
        a, b = r_d, (1 - eps) * r_d
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        x_maj = (x - x_0) * cos_theta + (y - y_0) * sin_theta
        x_min = -(x - x_0) * sin_theta + (y - y_0) * cos_theta
        z = np.sqrt((x_maj / a) ** 2 + (x_min / b) ** 2)
        
        return amp*np.exp(-z)

                
    def profile_fit(self, csv = None, subtract = True, show_results = True, save = True):
        '''
        Fit the surface brightness profile from ellipse fitting result using a n = 4 Sersic + exponential model.
        Build 2D disk model from the fits and subtract it from the galaxy image.
        
        Parameters:
            
            csv(pandas.DataFrame): The dataframe containing the ellipse fitting results. The default is None, and it will use self.phot_dat 
            Columns should include: sma(semi-major axis in pixels), tflux_e(total flux inside given sma), intens(intensity at sma), int_err(error in intensity) 
            
            subtract(bool): Whether or not to do the disk subtraction, default: True
            
            show_result(bool): Whether or not to show the surface brightness profile and/or disk subtracted image, default: False
            
            save(bool): Whether or not to save the results, default: True
        
        Usage:
            
            disk_model, disk_sub_image = phot.profile_fit(efit_result,subtract = True,
                                                             show_results = True,save = True)
            # If you want to perform ellipse fitting on the disk subtracted image, just call my_phot.efit()
            # again and set "wo_disk" to True.
            
            efit_wo_disk = phot.efit(wo_disk = True, update_disk = False)
        '''
        
        from scipy.optimize import minimize
        if csv is None:
            csv = self.phot_dat
            
        csv = csv.sort_values(by='sma').reset_index(drop=True)
        flux = csv['tflux_e'].to_numpy()
        flux_ratio = flux/np.max(flux)

        R90_id = np.argmin(np.abs(flux_ratio - 0.9)) # Find the R90
        R90 = csv.at[R90_id,'sma'] # in pixels
        snr = csv['intens']/csv['int_err']
        csv = csv.assign(snr = snr)
        
        
        _angular_dist = self.cosmo.angular_diameter_distance(self.Z).value * 1e3 # angular diameter distance in kpc
        _scale = 1 / 3600 * np.pi / 180 * _angular_dist # kpc/arcsec at the galaxy's redshift
        R90_kpc = R90 * self.pix_scale * _scale # R90 in kpc
        
        self.R90 = R90
        self.R90_kpc = R90_kpc
        
        # these three are for plotting only
        # smas_full = csv['sma'].to_numpy() * self.pix_scale * _scale # convert to sma in kpc
        # intens_full = csv['intens'].to_numpy() / (self.pix_scale * _scale)**2 # convert to intensity in flux per kpc^2
        # inten_err_full = csv['int_err'].to_numpy() / (self.pix_scale * _scale)**2
        
        
        csv = csv[csv.index <= R90_id].sort_values(by='sma').reset_index(drop=True) # for profile fitting, only fit within R90
        smas = csv['sma'].to_numpy() * self.pix_scale * _scale # convert to sma in kpc
        intens = csv['intens'].to_numpy() / (self.pix_scale * _scale)**2 # convert to intensity in flux per kpc^2
        inten_err = csv['int_err'].to_numpy() / (self.pix_scale * _scale)**2
        
        # 
        # smas_fit = csv_fit['sma'].to_numpy() * self.pix_scale * _scale # convert to sma in kpc
        # intens_fit = csv_fit['intens'].to_numpy() / (self.pix_scale * _scale)**2 # convert to intensity in flux per kpc^2
        # inten_err_fit = csv_fit['int_err'].to_numpy() / (self.pix_scale * _scale)**2
        
        
        inits = [1,1]
        bounds = [(0,None),(0,None)]
        E = minimize(self._chi_squared, x0 = np.array(inits),args = (smas[:],intens[:],inten_err[:]),
                     method = 'L-BFGS-B',bounds=bounds) # fit to the SB curves
        chsq = self._chi_squared(E.x,smas,intens,inten_err)
        
        self.chsq = chsq
        
        total_SB = self._composite_profile(smas, *E.x) # best fitting composite profile
        bulge_SB = self._deV_profile(smas, E.x[0]) # best fitting bulge profile
        disk_SB = self._exponential_profile(smas, E.x[1]) # best fitting disk profile
        
        disk_subtracted_SB = intens - disk_SB
        min_SB = np.min(disk_subtracted_SB)
        min_SB_id = np.argmin(disk_subtracted_SB)
        
        # check if the disk subtracted profile has negative value, if so, adjust the disk subtraction
        # factor so that the minimum after subtraction is non-negative.
        if min_SB < 0:
            self.disk_sub_factor = intens[min_SB_id] / disk_SB[min_SB_id]            
        else:
            self.disk_sub_factor = 1

        # check if the disk subtracted profile has positive gradient, if so reduce the 
        # disk subtraction factor so that the gradient is always negative.
        # Because ellipse fitting will fail for at positive gradient.
        
        disk_subtracted_SB = intens - self.disk_sub_factor * disk_SB
        SB_grad = np.gradient(disk_subtracted_SB)
        
        while len(SB_grad[SB_grad>0])>0:
            if self.disk_sub_factor >=0.05:
                self.disk_sub_factor -= 0.01
                disk_subtracted_SB = intens - self.disk_sub_factor * disk_SB
                SB_grad = np.gradient(disk_subtracted_SB)
            
            # Setting a minimum disk subtraction factor to be 0.05, you can remove this.
            elif self.disk_sub_factor < 0.05: 
                self.disk_sub_factor = 0.05
                disk_subtracted_SB = intens - self.disk_sub_factor * disk_SB
                break        
        
        
        
        if subtract:
            
            from scipy.signal import convolve2d as c2d
            
            if len(self.image)%2 == 1:
                centers = [int((len(self.image)-1)/2),int((len(self.image)-1)/2)]
            else:
                centers = [int((len(self.image))/2),int((len(self.image))/2)]  
                
            phy_cen_x = centers[1] * self.pix_scale * _scale
            phy_cen_y = centers[0] * self.pix_scale * _scale
            
            x,y = np.meshgrid(np.linspace(0,len(self.image[0,:])-1,num = len(self.image[0,:])), 
                              np.linspace(0,len(self.image[:,0])-1,num = len(self.image[:,0])))  
            
            x = x * self.pix_scale * _scale # convert to kpc
            y = y * self.pix_scale * _scale # convert to kpc
            
            Disk_2d = self._expon_2d(x = x, y = y, r_d = self.r_disk, eps = self.e_disk, theta = self.PA_disk, x_0 = phy_cen_x, y_0 = phy_cen_y, amp = E.x[1]) * (self.pix_scale * _scale)**2
            Disk_2d = Disk_2d * self.disk_sub_factor
            
            if (self.psf is not None) & (len(self.psf)>0):
                Disk_2d_conv = c2d(Disk_2d, self.psf, mode='same')
            else:
                Disk_2d_conv = Disk_2d
                
            Disk_subtracted_image = self.image - Disk_2d_conv            
            self.image_wo_disk = Disk_subtracted_image
            self.disk_model = Disk_2d_conv
            
            
        if (show_results):
            from matplotlib.lines import Line2D
            plt.figure()
            plt.errorbar(smas,intens,yerr = inten_err,fmt='s',color='black',mec='black',mfc='none')
            plt.scatter(smas,disk_subtracted_SB,marker='d',color='grey')
            # plt.scatter(smas,-1*disk_subtracted_SB,marker='d',edgecolor='grey',facecolor='none')
            plt.plot(smas,total_SB,ls='-',color='cyan')
            plt.plot(smas,bulge_SB,ls=':',color='grey')
            plt.plot(smas,disk_SB,ls='--',color='grey')
            
            hands = [Line2D([0],[0],marker='s',ls=' ',color='k',markerfacecolor='none',label='Obs.'),
            Line2D([0],[0],marker='d',ls=' ',color='grey',label='Obs. - Model disk'),
            Line2D([0],[0],ls='-',color='cyan',label='Model disk + bulge'),
            Line2D([0],[0],ls=':',color='grey',label='Model bulge'),
            Line2D([0],[0],ls='--',color='grey',label='Model disk'),]
            
            plt.legend(handles = hands)
            plt.text(0.7,0.8,'$\chi^2=$ %.3f'%self.chsq,transform=plt.gca().transAxes)
            plt.xlabel('$\mathrm{SMA}$ [kpc]')
            plt.ylabel('Surface Brightness [$10^{-9} \mathrm{maggie}/(kpc)^2$]')
            plt.yscale('log')
            plt.savefig(self.output_path+'%d'%self.galaxy_ID+'_'+'SB_profile.jpeg')
            plt.close()
        
            if (subtract):
            
                from matplotlib.patches import Ellipse as els
                
                # setting the ticks and ticklabels in kpc 
                xt0 = np.linspace(0,centers[1],4)
                xt1 = np.linspace(centers[1],len(self.image)-1,4)
                xt = np.append(xt0,xt1[1:])
                xt = np.round(xt,decimals=2)
                
                xtk0 = np.linspace(-centers[1],0,4)
                xtk1 = np.linspace(0,centers[1]-1,4)
                xtk = np.append(xtk0,xtk1[1:])
                xtk = xtk * _scale * self.pix_scale
                xtk = np.round(xtk,decimals=2)
                xtk = xtk.astype(str)
                
                # generating three ellipses representing the shape at R90
                elp_at_R90_0 = els((centers[1],centers[0]), 2 * self.R90, 2 * (1 - self.e_disk) * self.R90, np.rad2deg(self.PA_disk), ls='--', facecolor='none', edgecolor='red')
                elp_at_R90_1 = els((centers[1],centers[0]), 2 * self.R90, 2 * (1 - self.e_disk) * self.R90, np.rad2deg(self.PA_disk), ls='--', facecolor='none', edgecolor='red')
                elp_at_R90_2 = els((centers[1],centers[0]), 2 * self.R90, 2 * (1 - self.e_disk) * self.R90, np.rad2deg(self.PA_disk), ls='--', facecolor='none', edgecolor='red')
                
                thres = np.percentile(self.image[self.image>0],[10,99.9])
                
                fig,ax = plt.subplots(1,3,figsize=(18,6),sharex=True,sharey=True)
                ax[0].imshow(self.image ,origin='lower',cmap=plt.cm.binary,norm=colors.LogNorm(vmin = thres[0],vmax = thres[1]))
                ax[0].scatter(centers[1],centers[0],s=1,c='r')
                ax[0].set_title('Original image')
                ax[0].set_xticks(xt)
                ax[0].set_xticklabels(xtk)
                ax[0].set_yticks(xt)
                ax[0].set_yticklabels(xtk)
                ax[0].set_xlabel('x [kpc]')
                ax[0].set_ylabel('y [kpc]')
                
                ax[1].imshow(self.image_wo_disk,origin='lower',cmap=plt.cm.binary,norm=colors.LogNorm(vmin = thres[0],vmax = thres[1]))
                ax[1].scatter(centers[1],centers[0],s=1,c='r')
                ax[1].set_title('Disk-subtracted image')
                ax[1].set_xlabel('x [kpc]')
                
                ax[2].imshow(Disk_2d_conv,origin='lower',cmap=plt.cm.binary,norm=colors.LogNorm(vmin = thres[0],vmax = thres[1]))
                ax[2].scatter(centers[1],centers[0],s=1,c='r')
                ax[2].set_title('Disk model')
                ax[2].set_xlabel('x [kpc]')
                
                ax[0].add_artist(elp_at_R90_0)
                ax[1].add_artist(elp_at_R90_1)
                ax[2].add_artist(elp_at_R90_2)
                
                plt.savefig(self.output_path+'%d'%self.galaxy_ID+'_'+'disk_subtraction.jpeg')
                plt.close()
                
        if (save):
            np.save(self.output_path+'%d'%self.galaxy_ID+'_'+'disk_model.npy',self.disk_model)
            np.save(self.output_path+'%d'%self.galaxy_ID+'_'+'disk_subtracted_image.npy',self.image_wo_disk)
        
        return self.disk_model, self.image_wo_disk
    
    
'''
def main():
    
    import os
    from astropy.io import fits
    # df = pd.read_csv('/home/qadeng/galaxies.csv') 
    df = pd.read_csv('/home/qadeng/DESI_images_for_lensing_project/total_galaxies.csv')
    need = pd.read_csv('/home/qadeng/DESI_images_for_lensing_project/need_deblend.csv')
    
    ava = os.listdir('/home/qadeng/bar_lensing/desi/fits/')
    lst = []
    for i in range(len(ava)):
        if len(ava[i].split('_ps'))>1:
            lst.append(int(ava[i].split('_')[0]))
    
            
    df = df[df['objID'].isin(lst)].reset_index(drop=True) 
    df = df[df['objID'].isin(need['objID'])].reset_index(drop=True)
    df = df[df['objID'].isin(np.random.choice(df['objID'],5,replace=False))].reset_index(drop=True)
    
    
    for i in range(len(df)):
        # gal_image = np.load('/home/qadeng/bar_lensing/desi/fits/desi/%d_cut_r.npy'%df.at[i,'objID'])
        gal_image = np.load('/home/qadeng/DESI_images_for_lensing_project/deblending/%d.npy'%df.at[i,'objID'])
        gal_psf = fits.open('/home/qadeng/bar_lensing/desi/fits/%d_psf.fits'%df.at[i,'objID'])
        gal_psf = gal_psf[0].data.byteswap().newbyteorder()
        phot = Surface_photometry(image = gal_image, psf = gal_psf, output_path = '/home/qadeng/tests_for_the_integrated_bar_finding_code/modelling_and_efitting/', 
                                      RA = df.at[i,'ra'], DEC = df.at[i,'dec'], Z = df.at[i,'z'],
                                      galaxy_ID = df.at[i,'objID'], r_bulge = df.at[i,'Rb'], r_disk = df.at[i,'Rd'],
                                      e_disk = 1 - np.cos(np.deg2rad(df.at[i,'incd'])))
        
        e_res = phot.efit()
        disk_model, disk_sub = phot.profile_fit()
        e_res_wo_disk = phot.efit(wo_disk=True, update_disk=False)
        
    
if __name__ == "__main__":
 	main()
'''
