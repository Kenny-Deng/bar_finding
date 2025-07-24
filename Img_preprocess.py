#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 14:39:10 2025

@author: kenny
"""

import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.cosmology import LambdaCDM

import scarlet
import sep


class Img_preprocess(object):
    '''
    This is a class to pre-process a single galaxy image for further analysis.
    It heavily relies on the SEP and SCARLET python packages, please consult
    the corresponding documentations for their usage.
    
    The input image should be centered on the target galaxy.
    Default output is the r band deblended galaxy image(a numpy array) 
    saved to the output_path.
    
    It can be easily integrated into a multi-process program to deal with
    a large amount of galaxies.
    
    
    Parameters:
        
        img_fits(str): The absolute path of the image .fits file, optional.
        
        psf_fits(str): The absolute path of the PSF .fits file, optional.
        
        output_path(str): The directory to save the outputs.
        
        RA(float): Right ascension of the galaxy (degree).
        
        DEC(float): Declination of the galaxy (degree).
        
        Z(float): Redshift of the galaxy.
        
        galaxy_ID(int): Galaxy ID.
        
        pic_size(float): Size of the output deblended image (kpc).
        
        pix_scale(float): Pixel scale (arcsec/pixel), default: 0.262.
        
        band(str): Name of the channel, e.g. "g","r","i","z" etc., defualt: 'r'.
        
        cosmo(astropy.cosmology.Cosmology object): The adopted cosmology, default: Flat LambdaCDM with h = 0.7, Omega_m0 = 0.3
        
        image(numpy array): Numpy array containing the single band image, optional.
        
        weight_map(numpy array): Numpy array containing the single band weight map, optional.
        
        psf_map(numpy array): Numpy array containing the single band psf map, optional.
        
        wcs(fits): .fits file containing the WCS information about the galaxy image, optional.
        
        survey(str): Name of the survey, currently it is specifically designed for DESI.
        
        
        
    Usage:
        
        Normally, one should initialize the Img_preprocess by feeding in the image and PSF fits files,
        run the Unpack function and finally the Deblend function:
            
        my_img = Img_preprocess(img_fits='\home\path_to_image_fits',
                          psf_fits='\home\path_to_PSF_fits',
                          output_path = '\home\mypath\',RA = ra, DEC = dec, Z = z,
                          galaxy_ID = ID, pic_size = pic_size)
        
        my_img.Unpack()
        my_img.Deblend()
        
        One can input the images, weight_maps and PSF maps seperately as numpy arrays 
        alongside with the WCS information to directly do the deblending:
            
        my_img = Img_preprocess(output_path = '\home\mypath\',RA = ra, DEC = dec, Z = z,
                  galaxy_ID = ID, pic_size = pic_size, image='\home\path_to_image_array',
                  weight_map='\home\path_to_weight_array', psf_map='\home\path_to_psf_array', 
                  wcs='\home\path_to_WCS_info')
        
        my_img.Deblend()
        
    '''
    def __init__(self, 
                 img_fits = None,
                 psf_fits = None,
                 output_path = None,
                 RA = None,
                 DEC = None,
                 Z = None,
                 galaxy_ID = None,
                 pic_size = None,
                 pix_scale = None,
                 band = 'r',
                 cosmo = None,
                 image = None,
                 weight_map = None,
                 psf_map = None,
                 wcs = None,
                 survey = 'DESI'):
        
        self.cosmo = cosmo # The input cosmology
        self.survey = survey  # name of the survey, currently only support "DESI"
        self.output_path = output_path
        self.RA = RA 
        self.DEC = DEC
        self.Z = Z 
        self.galaxy_ID = galaxy_ID
        self.pic_size = pic_size
        self.pix_scale = pix_scale
        self.band = band
        
        if self.cosmo is None:
            self.cosmo =  LambdaCDM(70, 0.3, 0.7)
        
        print('Cosmology: LambdaCDM h = %.2f, Om0 = %.2f, Ode0 = %.2f.'%(self.cosmo.h,self.cosmo.Om0,self.cosmo.Ode0))
        
        if (self.survey == 'DESI') & (self.pix_scale is None):
            self.pic_scale = 0.262
        
        print('Image pixel scale: %.3f arcsec/pixel, processing %s band.'%(self.pic_scale,self.band))
        
        _angular_dist = self.cosmo.angular_diameter_distance(self.Z).value * 1e3 # angular diameter distance in kpc
        _pic_size_arcsec = (180/np.pi) * (self.pic_size / _angular_dist) * 3600 # size of the output image in arcsec
        
        self.pic_size_pix = int(np.round(_pic_size_arcsec / self.pic_scale,decimals=0)) # size of the output image in pixels
        
        
        self.image = image
        self.weight_map = weight_map
        self.psf_map = psf_map
        self.wcs = wcs
        
        self.img_fits = img_fits
        self.psf_fits = psf_fits
        
    
    def Unpack(self, save_unpack = True, save_path = None,
               band_mapping_image = {'g':1,'r':3,'z':5},
               band_mapping_psf = {'g':0,'r':1,'z':2}):
        '''
        A function used to unpack the DESI image into single band images;
        
        I am using the standard format of DESI's .fits image data(i.e. the 
        "subimage" in the Sky Viewer), where each file contains a HDUlist 
        recording images and inverse variance maps from g,r,z bands. 
        The PSF files(i.e. the "coadd psf") are also in this order.
        
        One should check the format of the input .fits and change 
        the band mappings accordingly.
        
        Parameters:
            
            save_unpack(bool): Whether or not to save the unpacked single band
            images, for the purpose of preprocessing, it should always be "True".
            save_path(str): The directory to save the outputs, optional.
            band_mapping_image(dict): The mapping between the order of the image HDUlist and the channels.
            band_mapping_psf(dict): The mapping between the order of the PSF HDUlist and the channels.
        
        Returns:
            None.
            
        '''
        
        if save_path is None:
            save_path = self.output_path
        
        if self.img_fits is not None:
            
            h = fits.open(self.img_fits)
            try:
                # Normally, the .fits file should have at least 7 HDUs where the first one is empty,
                # the 2nd., 4th., 6th. are images of g,r,z bands, while the 3rd., 5th., 7th. are the
                # weight maps of g,r,z bands.
                if len(h) > 1:
                    img = h[band_mapping_image[self.band]].data.byteswap().newbyteorder() # the image layer
                    wgt = h[band_mapping_image[self.band]+1].data.byteswap().newbyteorder() # the weight layer
                    hdr = h[band_mapping_image[self.band]].header # the header
                    
                    self.image = img
                    self.weight_map = wgt
                    self.wcs = WCS(hdr)
                    
                    h.close()
                # For some galaxies, there no weight maps, and the .fits has only one HDU containing
                # a 3d array recording the g,r,z band image.
                elif len(h) == 1:
                    img = h[0].data[band_mapping_psf[self.band]] # Note: For images that has only one HDU, the band mapping is the same as the PSF
                    hdr = h[0].header
                    fits.writeto('%s%d_r.fits'%(save_path,self.galaxy_ID),img, hdr,overwrite=True) # We only need one band
                    h.close()
                    
                    temp_h = fits.open('%s%d_r.fits'%(save_path,self.galaxy_ID))
                    img = temp_h[0].data.byteswap().newbyteorder()
                    hdr = temp_h[0].header
                    
                    wgt = 0.05*img # In cases where we don't have the weight maps, set the weight to be 5% of the image
                    
                    self.image = img
                    self.weight_map = wgt
                    self.wcs = WCS(hdr)
                    temp_h.close()
            
            except Exception:
                print('Corrupted image for galaxy %d, please check the integrity of the .fits file.'%self.galaxy_ID)
                h.close()
                raise
        
        else:
            print('No image provided.')
            raise
        
        if self.psf_fits is not None:
            
            h = fits.open(self.psf_fits) ### Careful, sometimes the psf file may not 
                                         ### have any content, try using the psf file 
                                         ### of a nearby galaxy instead.
            try:
                psf = h[band_mapping_psf[self.band]].data.byteswap().newbyteorder()
            
                self.psf_map = psf
                h.close()
                
            except Exception:
                print('Corrupted PSF for galaxy %d, please check the integrity of the .fits file.'%self.galaxy_ID)
                h.close()
                raise
        
        else:
            print('No PSF provided.')
            raise
            
        if save_unpack:
            
            print('Saving the unpacked data to %s'%(save_path+str(self.galaxy_ID)+'_xxx'))
            np.save('%s'%(save_path+str(self.galaxy_ID)+'_'+self.band+'_image.npy'),self.image)
            np.save('%s'%(save_path+str(self.galaxy_ID)+'_'+self.band+'_weight_map.npy'),self.weight_map)
            np.save('%s'%(save_path+str(self.galaxy_ID)+'_'+self.band+'_psf_map.npy'),self.psf_map)
            wcs_fits = self.wcs.to_fits()
            wcs_fits.writeto('%s'%(save_path+str(self.galaxy_ID)+'_'+self.band+'_wcs.fits'))

    
    def Deblend(self, thresh_de = 1.5, deblend_nthresh = 64, deblend_cont = 5e-3, minarea = 10, show_extract = False,
                sigma_psf_model = 0.8, max_components = 2, min_snr = 75, thresh_sc = 1, show_scarlet_model = False,
                save_deblend = True, save_path = None, **kwargs):
        
        '''
        This function does the deblending and foreground/background removal to the input single band image.
        The parameters are adopted straight from SEP and SCARLET, except for the boolean parameters.
        See "https://sep.readthedocs.io/en/v1.1.x/index.html" and "https://pmelchior.github.io/scarlet/index.html" for references.
        
        The default values are chosen empirically to strike a balance between the acurracy and computational expense,
        one should tune these values on his/hers own accord.
        
        Parameters:
            For SEP:
                thresh_de(float): Threshold pixel value for detection, the 'thresh' parameter in SEP.extract.
                deblend_nthresh(float): Number of thresholds used for object deblending.
                deblend_cont(float): Minimum contrast ratio used for object deblending.
                minarea(int): Minimum number of pixels required for an object. 
                show_extract(bool): Whether or not to show the sources identified in the SExtractor run.
            
            For SCARLET:
                sigma_psf_model(float): One-sigma for the Gaussian PSF model.
                max_components(int): The maximum number of components in a source.
                min_snr(float): Mininmum SNR per component of a multi-component source.
                thresh_sc(float): Multiple of the backround RMS used as a flux cutoff for morphology 
                initialization, the 'thresh' parameter in SCARLET.initialization.init_all_source.
                show_scarlet_model(bool): Whether or not to show the SCARLET models.
                save_deblend(bool): Whether or not to save the SCARLET deblended images, should always be 'True'.
                save_path(str): The directory to save the outputs, optional.
        
        Returns:
            None.
        '''
        
        if save_path is None:
            save_path = self.output_path
            
            
        bkg = sep.Background(self.image) # Estimate the sky background
        sub = self.image - bkg # Background subtracted image
        objects,maps = sep.extract(sub, thresh_de, 
                                   deblend_nthresh = deblend_nthresh, 
                                   err = bkg.globalrms,
                                   segmentation_map = True,
                                   deblend_cont = deblend_cont,
                                   minarea = minarea) # Run the source extraction
        if show_extract:
            ''' 
            Show the SExtract objects, adapted from https://sep.readthedocs.io/en/v1.1.x/tutorial.html
            '''
            import matplotlib.pyplot as plt
            from matplotlib.patches import Ellipse
            fig, ax = plt.subplots()
            m, s = np.mean(sub), np.std(sub)
            im = ax.imshow(sub, interpolation='nearest', cmap='gray',
                            vmin=m-s, vmax=m+s, origin='lower')
            
            # plot an ellipse for each object
            for i in range(len(objects)):
                e = Ellipse(xy=(objects['x'][i], objects['y'][i]),
                            width=6*objects['a'][i],
                            height=6*objects['b'][i],
                            angle=objects['theta'][i] * 180. / np.pi) # Beware of the scales of the ellipses(see SEP documentation)
                e.set_facecolor('none')
                e.set_edgecolor('red')
                ax.add_artist(e)
            
            plt.savefig('%s%d_SEXtract_sources.jpeg'%(save_path,self.galaxy_ID))
            plt.close()
        
        obx = objects['x'] # The list of x(pixel) coordinates of all detected sources within the image
        oby = objects['y'] # The list of y(pixel) coordinates of all detected sources within the image
        cen_coor = SkyCoord(self.RA,self.DEC,unit='deg') # On-sky coordinates of the target galaxy
        gal_x,gal_y = self.wcs.world_to_pixel(cen_coor) # Locating the target center on the image
                
        x_offset = (obx - gal_x)**2
        y_offset = (oby - gal_y)**2
        sum_offset = np.sqrt(x_offset + y_offset)
        
        cen_idx = np.argmin(sum_offset) # Finding the position of the target galaxy in the detected object list 
        centers = []
        cns = []
        for i in range(len(obx)):
            centers.append(self.wcs.pixel_to_world_values(obx[i],oby[i]))
            cns.append((obx[i],oby[i]))
            
        images_3d = np.tile(sub,(1,1,1)) # Making it a 3d array
        weights_3d = np.tile(self.weight_map,(1,1,1))
        
        model_psf = scarlet.GaussianPSF(sigma=(sigma_psf_model,))
        model_frame = scarlet.Frame(images_3d.shape, wcs = self.wcs, psf = model_psf, channels = self.band) # Generate a model Frame
        observation = scarlet.Observation(images_3d, wcs = self.wcs, psf = scarlet.psf.ImagePSF(self.psf_map), weights = weights_3d, channels=self.band).match(model_frame) # Matching observation to the model Frame
        
        
        # Initialize all possible sources(at the locations given by the 'centers' variable) 
        # in this Frame, i.e. make models for all sources.
        sources, skipped = scarlet.initialization.init_all_sources(model_frame,
                                                                      centers,
                                                                      observation,
                                                                      max_components = max_components,
                                                                      min_snr = min_snr,
                                                                      thresh = thresh_sc,
                                                                      fallback=True,
                                                                      silent=True,
                                                                      set_spectra=True
                                                                    )
        
        blend = scarlet.Blend(sources, observation)
        it, logL = blend.fit(80, e_rel=1e-4) # Fit all source models to the real data
        
        # Compute model
        model = blend.get_model()    
        chd_id = np.arange(len(obx)) 
        chd_id = chd_id[chd_id != cen_idx] # Get the index of all sources but the target galaxy(i.e. all children)
        
        # Render it in the observed frame
        model_ = observation.render(model)
        residual = images_3d-model_
        
        stars_all_chd = np.zeros_like(images_3d[0,:,:]) # Generate an array to store the fluxes from all children
        for i in range(len(chd_id)):
            chd_to_obs = sources[chd_id[i]].model_to_box(observation.bbox,) 
            stars_all_chd = np.add(stars_all_chd,chd_to_obs)
        
        # Render it in the observed frame and subtract them from the image
        stars_render = observation.render(stars_all_chd)
        obs_deblended = images_3d[0,:,:] - stars_render[0,:,:] 
        # Cropping the output to fit the desired size
        obs_deblended = obs_deblended[int(np.round(cns[cen_idx][1],decimals=0)) - self.pic_size: int(np.round(cns[cen_idx][1],decimals=0)) + self.pic_size + 1, int(np.round(cns[cen_idx][0],decimals=0)) - self.pic_size: int(np.round(cns[cen_idx][0],decimals=0)) + self.pic_size + 1]
        
        if save_deblend:
            np.save('%s%d_deblended.npy'%(save_path,self.galaxy_ID),obs_deblended)
        
        if show_scarlet_model:
            
            import matplotlib.pyplot as plt
            import matplotlib.colors as colors
            thres = np.percentile(self.image[self.image>0],[10,99.8])
            
            fig = plt.figure(figsize=(15,5))
            ax = [fig.add_subplot(1,3,n+1) for n in range(3)]
            
            ax[0].imshow(images_3d[0,:,:],cmap=plt.cm.binary,norm=colors.LogNorm(vmin=thres[0],vmax=thres[1]))
            ax[0].set_title("Data")
            ax[1].imshow(model_[0,:,:],cmap=plt.cm.binary,norm=colors.LogNorm(vmin=thres[0],vmax=thres[1]))
            ax[1].set_title("Model")
            ax[2].imshow(residual[0,:,:],cmap=plt.cm.coolwarm,norm=colors.Normalize(vmin=-0.03,vmax=0.03))
            ax[2].set_title("Residual")
            plt.savefig('%s%d_scarlet_data_model_residual.jpeg'%(save_path,self.galaxy_ID))
            plt.close()
        

            plt.figure()
            plt.imshow(obs_deblended,origin='lower',cmap=plt.cm.binary,norm=colors.LogNorm(vmin=thres[0],vmax=thres[1]))
            plt.savefig('%s%d_deblended_image.jpeg'%(save_path,self.galaxy_ID))
            plt.close()
            
'''     
def main():
    
    import pandas as pd
    
    df = pd.read_csv('/home/qadeng/galaxies.csv') 
    # df = df[df['objID'].isin(np.random.choice(df['objID'],5,replace=False))].reset_index(drop=True)
    
    for i in range(len(df)):
        my_img = Img_preprocess(output_path = '/home/qadeng/output/',
                                img_fits='/home/qadeng/fits_datas/%d.fits'%df.at[i,'objID'],
                                psf_fits='/home/qadeng/fits_datas/%d_psf.fits'%df.at[i,'objID'],
                                RA = df.at[i,'ra'], DEC = df.at[i,'dec'], Z = df.at[i,'z'],
                                galaxy_ID = df.at[i,'objID'], pic_size = 50)
        
        my_img.Unpack()
        my_img.Deblend(show_extract=True,show_scarlet_model=True)
    
if __name__ == "__main__":
 	main()
'''