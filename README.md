'''
AstroPhoto -- Program to manipulate astronomical pictures
Copyright (c) 2015-2016, Emanuele Laface (Emanuele.Laface at gmail.com)

All rights reserved.

Redistribution and use, with or without modification, are permitted provided that the following conditions are met:
Redistributions must retain the above copyright notice, this list of conditions and the following disclaimer.
Neither the name of the AstroPhoto Author nor the names of any contributors may be used to endorse or promote
products derived from this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

Installation:

AstroPhoto is mainly a wrapper around astrometry.net so it requires solve-field installed and working.
Everything about astrometry can be found on their website, including the license.
You also need the index files. To establish which index you need you can use the FOV information provided by AstroPhoto
after the selction of your camera and compare with the instruction on this page: http://astrometry.net/doc/readme.html

AstroPhoto assume that astrometry is installed within the following directories:
directory: /usr/local/astrometry
solve-field: /usr/local/astrometry/bin/solve-field
index files: anywhere where the solve-filed can find them. The default is /usr/local/astrometry/data
the ngc2000.fits: /usr/local/astrometry/extras/ngc2000.fits
these informations are hard-coded in the python. I know, this is not a good practice but it's life.

Python libraries that you should have if you have python installed:

sys
os
subprocess
threading
numpy
scipy
matplotlib

Python libraries that you probably have to install:

pickle
rawpy
imageio
astropy
opencv (cv2)
PyQt4

I have as python installer Anaconda and for extra packages I use pip. 90% of the packages installed with no problem.
I had some problem with some of them but I am not sure which one, so if unsure feel free to contact me.
All my deep sky images are processed with AstroPhoto and can be found here: https://www.flickr.com/photos/125149679@N03/albums

Please consider that I am not a professinal coder, I did this program just to have fun, so don't complain if it is not good.

Use:

It has several buttons, try them and you will learn. Or ask me.
