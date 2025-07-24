# bar_finding
This is a pipeline that extract bar information from galaxy images described in Deng+2023(https://ui.adsabs.harvard.edu/abs/2023MNRAS.525.1520D/abstract).

Dependencies:

SEP (https://sep.readthedocs.io/en/stable/index.html)

SCARLET (https://pmelchior.github.io/scarlet/)

photutils (https://photutils.readthedocs.io/en/stable/index.html)

savitzky_golay_werrors (https://github.com/surhudm/savitzky_golay_with_errors)

numpy, pandas, astropy, scipy

Usage:

Run "Img_process.py" to unpack a DESI multi-channel fits image(Downloaded from Sky Viewer https://www.legacysurvey.org/viewer) into single channel, and do deblending. 

Perform ellipse fitting and/or disk subtraction on the deblended images by running "Surface_photometry.py".

Find peaks in the ellipticity profile by running "Bar_finding.py".

These functions can be easily incorperated into a multi-processing script.
