'''
AstroPhoto -- Program to manipulate astronomical pictures 
Copyright (c) 2015-2016, Emanuele Laface (Emanuele.Laface@gmail.com)

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

import sys
import os
import subprocess
import pickle
import threading
import numpy
import rawpy
import rawpy.enhance
import imageio
import astropy.io.fits
import astropy.wcs
import astropy.modeling
import scipy.optimize
import cv2
import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot
from PyQt4 import QtGui, QtCore

class AstroImage:
  def __init__(self, filename):
    if os.path.isfile(filename):
      self.error = False
      self.filename = filename
      self.white = 65535
      self.is_loaded = False
      self.is_aligned = False
      self.is_flat = False
      self.is_solved = False
    else:
      self.error = True

  def loadDump(self):
    if not self.error:
      try:
        file_dump = open(self.filename, 'rb')
        self.__dict__ = pickle.load(file_dump)
        file_dump.close()
      except:
        self.error = True

  def loadRaw(self):
    if not self.error:
      try:
        bad_pixels = rawpy.enhance.find_bad_pixels([self.filename], find_hot=True, find_dead=True, confirm_ratio=0.9)
        raw = rawpy.imread(self.filename)
        rawpy.enhance.repair_bad_pixels(raw, bad_pixels, method='median')
        self.rgb16 = raw.postprocess(no_auto_bright=True, user_flip=False, output_bps=16)
        self.width = self.rgb16.shape[0]
        self.height = self.rgb16.shape[1]
        self.is_loaded = True
      except:
        self.error = True

  def openFile(self):
    if not self.error:
      name, extension = os.path.splitext(self.filename)
      if extension == '.raw':
        self.loadDump()
      else:
        self.loadRaw()

  def saveDump(self):
    if not self.error:
      try:
        name, extension = os.path.splitext(self.filename)
        file_dump = open(name+'.raw', 'wb')
        pickle.dump(self.__dict__, file_dump, pickle.HIGHEST_PROTOCOL)
        file_dump.close()
      except:
        self.error = True

  def saveTiff(self):
    if not self.error and self.is_loaded:
      try:
        name, extension = os.path.splitext(self.filename)
        imageio.imsave(name+'.tiff', self.rgb16)
      except:
        self.error = True

  def savePpm(self):
    if not self.error and self.is_loaded:
      try:
        name, extension = os.path.splitext(self.filename)
        imageio.imsave(name+'.ppm', self.rgb16)
      except:
        self.error = True

  def flat(self):
    if not self.error and not self.is_flat:
      steps = 200
      y = numpy.arange(10, self.width-10, steps)
      x = numpy.arange(10, self.height-10, steps)
      x, y = numpy.meshgrid(x,y)
      r = self.rgb16[10:self.width-10:steps,10:self.height-10:steps,0]
      g = self.rgb16[10:self.width-10:steps,10:self.height-10:steps,1]
      b = self.rgb16[10:self.width-10:steps,10:self.height-10:steps,2]
      poly_init = astropy.modeling.models.Polynomial2D(degree=2)
      fit_poly = astropy.modeling.fitting.LevMarLSQFitter()
      poly_r = fit_poly(poly_init, x, y, r)
      poly_g = fit_poly(poly_init, x, y, g)
      poly_b = fit_poly(poly_init, x, y, b)
      y = numpy.arange(0, self.width)
      x = numpy.arange(0, self.height)
      x, y = numpy.meshgrid(x,y)
      r = poly_r(x,y)
      g = poly_g(x,y)
      b = poly_b(x,y)

      # Align approximatively the maximum of the histogram with self.white/10
      r = (self.rgb16[:,:,0] - r)+self.white/10.0
      g = (self.rgb16[:,:,1] - g)+self.white/10.0
      b = (self.rgb16[:,:,2] - b)+self.white/10.0

      r = r.clip(0,self.white)
      g = g.clip(0,self.white)
      b = b.clip(0,self.white)
      r = r.astype(numpy.uint16)
      g = g.astype(numpy.uint16)
      b = b.astype(numpy.uint16)

      # Fine alignment of the histogram
      hist_r = cv2.calcHist([r],[0],None,[self.white],[0,self.white])[1:]
      hist_g = cv2.calcHist([g],[0],None,[self.white],[0,self.white])[1:]
      hist_b = cv2.calcHist([b],[0],None,[self.white],[0,self.white])[1:]
      r = r + (self.white/10.0 - hist_r[:,0].argsort()[::-1][0])
      g = g + (self.white/10.0 - hist_g[:,0].argsort()[::-1][0])
      b = b + (self.white/10.0 - hist_b[:,0].argsort()[::-1][0])

      r = r.clip(0,self.white)
      g = g.clip(0,self.white)
      b = b.clip(0,self.white)
      r = r.astype(numpy.uint16)
      g = g.astype(numpy.uint16)
      b = b.astype(numpy.uint16)

      self.rgb16[:,:,0] = r
      self.rgb16[:,:,1] = g
      self.rgb16[:,:,2] = b

      self.is_flat = True

  def solve(self, scale):
    if not self.error and not self.is_solved:
      name, extension = os.path.splitext(self.filename)
      self.savePpm()
      scale_low = str(scale*80.0/100.0)
      scale_high = str(scale*120.0/100.0)
      subprocess.call(["/usr/local/astrometry/bin/solve-field", "--downsample", "2", "--tweak-order", "2", "--scale-units", "arcsecperpix", "--scale-low", scale_low, "--scale-high", scale_high, "--no-plots", "--overwrite", name+".ppm"])
      if os.path.isfile(name+'.solved'):
        correlation = astropy.io.fits.open(name+'.corr')
        self.correlation = correlation[1].data
        wcs = astropy.wcs.WCS(astropy.io.fits.open(name+'.new')[0].header)
        # Search for deep sky objects
        galaxy = astropy.io.fits.open('/usr/local/astrometry/extra/ngc2000.fits')
        self.galaxy = numpy.empty((1000, 4), dtype=numpy.int)
        galaxy_num = 0
        galaxy_scale = astropy.wcs.utils.proj_plane_pixel_scales(wcs).mean()
        for i in galaxy[1].data:
          galaxy_pix = wcs.wcs_world2pix(numpy.array([[i[1], i[2]]]),1)
          if galaxy_pix[0][0] > 0 and galaxy_pix[0][0] < self.rgb16.shape[0] and galaxy_pix[0][1] > 0 and galaxy_pix[0][1] < self.rgb16.shape[1]:
            self.galaxy[galaxy_num, 0] = i[0]
            self.galaxy[galaxy_num, 1] = galaxy_pix[0][0]
            self.galaxy[galaxy_num, 2] = galaxy_pix[0][1]
            self.galaxy[galaxy_num, 3] = i[3]/galaxy_scale
            galaxy_num = galaxy_num + 1
        self.galaxy = self.galaxy[0:galaxy_num]

        self.stars = numpy.empty((len(correlation[1].data),3))
        for i in range(0,len(correlation[1].data)):
            star_x = correlation[1].data[i][5]
            star_y = correlation[1].data[i][4]
            self.stars[i][0] = star_x
            self.stars[i][1] = star_y
            self.stars[i][2] = correlation[1].data[i][11]

        self.stars = self.stars[self.stars[:,2].argsort()[::-1]]
        if self.stars.shape[0] > 20:
          self.stars = self.stars[0:20]

        self.is_solved = True

      try:
        os.remove(name+'-indx.xyls')
        os.remove(name+'.axy')
        os.remove(name+'.corr')
        os.remove(name+'.match')
        os.remove(name+'.new')
        os.remove(name+'.ppm')
        os.remove(name+'.rdls')
        os.remove(name+'.solved')
        os.remove(name+'.wcs')
      except:
        print "    Some file was not here."

  def stars_hash(self):
      hash_size = 1
      for i in range(self.stars.shape[0], self.stars.shape[0]-5, -1):
          hash_size = hash_size * i
      hash_size = hash_size/120
      index = 0
      self.starsHash = numpy.empty(shape=(hash_size,10))
      self.starsSequence = numpy.empty(shape=(hash_size,5), dtype='int')
      for a in range(0,self.stars.shape[0]-4):
          for b in range(a+1,self.stars.shape[0]-3):
              for c in range(b+1,self.stars.shape[0]-2):
                  for d in range (c+1,self.stars.shape[0]-1):
                      for e in range (d+1,self.stars.shape[0]):
                          sequence = [a,b,c,d,e]
                          Distance=numpy.zeros(10)
                          k=0
                          for i in range(0,4):
                              for j in range(i+1, 5):
                                  Distance[k]=(self.stars[sequence[i]][0]-self.stars[sequence[j]][0])**2+(self.stars[sequence[i]][1]-self.stars[sequence[j]][1])**2
                                  k=k+1
                          Ratios=numpy.zeros(10)
                          for i in range(0, 10):
                              Ratios[i]=Distance[i]/Distance.max()
                          Ratios.sort()
                          self.starsHash[index]=Ratios
                          self.starsSequence[index]=sequence
                          index = index + 1


  def rotate(self, angle):
    height_pad = numpy.sqrt(self.rgb16.shape[0]**2+self.rgb16.shape[1]**2)/2.0 - self.rgb16.shape[1]/2
    width_pad = numpy.sqrt(self.rgb16.shape[0]**2+self.rgb16.shape[1]**2)/2.0 - self.rgb16.shape[0]/2
    self.rgb16 = numpy.lib.pad(self.rgb16,((abs(int(width_pad)),abs(int(width_pad))),(abs(int(height_pad)),abs(int(height_pad))),(0,0)), 'constant', constant_values=0)
    matrix = cv2.getRotationMatrix2D((self.rgb16.shape[1]/2,self.rgb16.shape[0]/2),angle/numpy.pi*180,1)
    self.rgb16 = cv2.warpAffine(self.rgb16,matrix,(self.rgb16.shape[1],self.rgb16.shape[0]))

  def translate(self, x, y):
    self.rgb16 = numpy.lib.pad(self.rgb16,((abs(int(x)),abs(int(x))),(abs(int(y)),abs(int(y))),(0,0)), 'constant', constant_values=0)
    matrix = numpy.float32([[1,0,y],[0,1,x]])
    self.rgb16 = cv2.warpAffine(self.rgb16,matrix,(self.rgb16.shape[1],self.rgb16.shape[0]))

  def crop(self):
    self.rgb16 = self.rgb16[self.rgb16.shape[0]/2-self.width/2:self.rgb16.shape[0]/2+self.width/2, self.rgb16.shape[1]/2-self.height/2:self.rgb16.shape[1]/2+self.height/2]

class AstroUI(QtGui.QWidget):
  
  def __init__(self):
    super(AstroUI, self).__init__()
    self.createWidgets()
    self.createWindow()
    self.createEvents()
    self.image_update = False
    self.update_histo = False
    self.show_solve = False
    self.show_stars = False
    self.show_ref = False
    self.reference_image = None

  def createWidgets(self):
    self.open_button = QtGui.QPushButton()
    self.open_button.setFlat(True)
    self.open_button.setAutoFillBackground(True)
    self.open_button.setStyleSheet("QPushButton {background-image: url(icons/open.png); background-repeat: no-repeat; width: 60px; height: 60px;}" "QPushButton:pressed { border:invisible; }")

    self.left_arrow_button = QtGui.QPushButton()
    self.left_arrow_button.setFlat(True)
    self.left_arrow_button.setAutoFillBackground(True)
    self.left_arrow_button.setStyleSheet("QPushButton {background-image: url(icons/left_arrow.png); background-repeat: no-repeat; width: 29px; height: 35px;}" "QPushButton:pressed { border:invisible; }")

    self.right_arrow_button = QtGui.QPushButton()
    self.right_arrow_button.setFlat(True)
    self.right_arrow_button.setAutoFillBackground(True)
    self.right_arrow_button.setStyleSheet("QPushButton {background-image: url(icons/right_arrow.png); background-repeat: no-repeat; width: 30px; height: 35px;}" "QPushButton:pressed { border:invisible; }")

    self.ref_stars_button = QtGui.QPushButton()
    self.ref_stars_button.setFlat(True)
    self.ref_stars_button.setAutoFillBackground(True)
    self.ref_stars_button.setStyleSheet("QPushButton {background-image: url(icons/ref_stars.png); background-repeat: no-repeat; width: 66px; height: 35px;}" "QPushButton:pressed { border:invisible; }")

    self.img_stars_button = QtGui.QPushButton()
    self.img_stars_button.setFlat(True)
    self.img_stars_button.setAutoFillBackground(True)
    self.img_stars_button.setStyleSheet("QPushButton {background-image: url(icons/img_stars.png); background-repeat: no-repeat; width: 66px; height: 35px;}" "QPushButton:pressed { border:invisible; }")

    self.flat_button = QtGui.QPushButton()
    self.flat_button.setFlat(True)
    self.flat_button.setAutoFillBackground(True)
    self.flat_button.setStyleSheet("QPushButton {background-image: url(icons/flat.png); background-repeat: no-repeat; width: 60px; height: 61px;}" "QPushButton:pressed { border:invisible; }")

    self.align_button = QtGui.QPushButton()
    self.align_button.setFlat(True)
    self.align_button.setAutoFillBackground(True)
    self.align_button.setStyleSheet("QPushButton {background-image: url(icons/align.png); background-repeat: no-repeat; width: 61px; height: 61px;}" "QPushButton:pressed { border:invisible; }")

    self.solve_button = QtGui.QPushButton()
    self.solve_button.setFlat(True)
    self.solve_button.setAutoFillBackground(True)
    self.solve_button.setStyleSheet("QPushButton {background-image: url(icons/solve.png); background-repeat: no-repeat; width: 60px; height: 61px;}" "QPushButton:pressed { border:invisible; }")

    self.stack_button = QtGui.QPushButton()
    self.stack_button.setFlat(True)
    self.stack_button.setAutoFillBackground(True)
    self.stack_button.setStyleSheet("QPushButton {background-image: url(icons/stack.png); background-repeat: no-repeat; width: 61px; height: 61px;}" "QPushButton:pressed { border:invisible; }")

    self.save_tiff_button = QtGui.QPushButton()
    self.save_tiff_button.setFlat(True)
    self.save_tiff_button.setAutoFillBackground(True)
    self.save_tiff_button.setStyleSheet("QPushButton {background-image: url(icons/save_tiff.png); background-repeat: no-repeat; width: 61px; height: 62px;}" "QPushButton:pressed { border:invisible; }")

    self.save_raw_button = QtGui.QPushButton()
    self.save_raw_button.setFlat(True)
    self.save_raw_button.setAutoFillBackground(True)
    self.save_raw_button.setStyleSheet("QPushButton {background-image: url(icons/save_raw.png); background-repeat: no-repeat; width: 61px; height: 63px;}" "QPushButton:pressed { border:invisible; }")

    self.check_reference = QtGui.QCheckBox('Not Set', self)

    self.batch_button = QtGui.QPushButton("I'm Feeling Lucky")

    self.camera_list = { 'Canon 10D': [ 22.7, 15.1, 7.4], 'Canon 20D': [ 22.5, 15, 6.42], 'Canon 30D': [ 22.5, 15, 6.42],
        'Canon 40D': [ 22.2, 14.8, 5.71], 'Canon 50D': [ 22.3, 14.9, 4.7], 'Canon 60D': [ 22.3, 14.9, 4.3],
        'Canon 300D': [ 22.7, 15.1, 7.4], 'Canon 350D': [ 22.2, 14.8, 6.42], 'Canon 400D': [ 22.2, 14.8, 5.71],
        'Canon 450D': [ 22.2, 14.8, 5.2], 'Canon 500D': [ 22.3, 14.9, 4.7], 'Canon 550D': [ 22.3, 14.9, 4.3],
        'Canon 600D': [ 22.3, 14.9, 4.3], 'Canon 1000D': [ 22.2, 14.8, 5.7], 'Canon 1100D': [ 22.3, 14.7, 5.2],
        'Canon 5D': [ 35.8, 23.9, 8.2], 'Canon 5D Mark II': [ 36, 24, 6.41], 'Canon 7D': [ 22.3, 14.9, 4.3],
        'Canon 1Ds Mark II': [ 36, 24, 7.2], 'Canon 1D Mark III': [ 28.1, 18.7, 7.2], 'Canon 1Ds Mark III': [ 36, 24, 6.42],
        'Nikon D3': [ 36, 23.9, 8.45], 'Nikon D3X': [ 35.9, 24, 5.9], 'Nikon D40': [ 23.7, 15.6, 7.8],
        'Nikon D50': [ 23.7, 15.6, 7.8], 'Nikon D60': [ 23.6, 15.8, 6.08], 'Nikon D70': [ 23.7, 15.6, 7.8],
        'Nikon D80': [ 23.6, 15.8, 6.05], 'Nikon D90': [ 23.6, 15.8, 5.5], 'Nikon D200': [ 23.6, 15.8, 6.05],
        'Nikon D300': [ 23.6, 15.8, 5.4], 'Nikon D700': [ 36, 23.9, 8.45], 'Nikon D3000': [ 23.6, 15.8, 5.08],
        'Nikon D3100': [ 23.1, 15.4, 5.02], 'Nikon D5000': [ 23.1, 15.4, 5.5], 'Nikon D7000': [ 23.6, 15.6, 4.78],
        'Olympus E-5': [ 17.3, 13, 4.7] }

    self.camera_select = QtGui.QComboBox(self)
    for i in self.camera_list.keys():
      self.camera_select.addItem(i)

    self.text_line = QtGui.QLabel(self)
    self.camera_select.setCurrentIndex(self.camera_list.keys().index('Canon 1000D'))
    self.pixel_size = QtGui.QLineEdit(self)
    self.choose_camera()
    self.pixel_size.setFixedWidth(50)
    self.focal_length = QtGui.QLineEdit(self)
    self.focal_length.setText('1200')
    self.focal_length.setFixedWidth(50)
    self.solve_scale = QtGui.QLineEdit(self)
    self.solve_scale.setFixedWidth(50)
    self.scale_calculator()
    self.text_line.setText('AstroPhoto ready')
    self.raw_saved = []

    self.image_label = QtGui.QLabel(self)
    self.myImage = QtGui.QImage((numpy.ones((768,512,3))*155).astype(numpy.uint8), 768, 512, QtGui.QImage.Format_RGB888)

    self.histogram_label = QtGui.QLabel(self)
    self.myHistogram = QtGui.QImage((numpy.ones((384,254,3))*155).astype(numpy.uint8), 384, 254, QtGui.QImage.Format_RGB888)

    self.star_label = QtGui.QLabel(self)
    self.myStar = QtGui.QImage((numpy.ones((384,254,3))*155).astype(numpy.uint8), 384, 254, QtGui.QImage.Format_RGB888)

  def createWindow(self):
    self.left_arrow_button.setEnabled(False)
    self.right_arrow_button.setEnabled(False)
    self.ref_stars_button.setEnabled(False)
    self.img_stars_button.setEnabled(False)
    self.flat_button.setEnabled(False)
    self.solve_button.setEnabled(False)
    self.align_button.setEnabled(False)
    self.stack_button.setEnabled(False)
    self.save_tiff_button.setEnabled(False)
    self.save_raw_button.setEnabled(False)
    self.check_reference.setEnabled(False)
    self.batch_button.setEnabled(False)

    grid = QtGui.QGridLayout()
    grid.setSpacing(4)
    grid.addWidget(self.open_button,0, 0, 2, 1)
    grid.addWidget(self.left_arrow_button,0, 1, 2, 1)
    grid.addWidget(self.right_arrow_button,0, 2, 2, 1)
    grid.addWidget(self.ref_stars_button,0, 3, 2, 1)
    grid.addWidget(self.img_stars_button,0, 4, 2, 1)
    grid.addWidget(self.flat_button,0, 5, 2, 1)
    grid.addWidget(self.solve_button,0, 6, 2, 1)
    grid.addWidget(self.align_button,0, 7, 2, 1)
    grid.addWidget(self.stack_button,0, 8, 2, 1)
    grid.addWidget(self.save_tiff_button,0, 9, 2, 1)
    grid.addWidget(self.save_raw_button,0, 10, 2, 1)

    hbox1 = QtGui.QHBoxLayout()
    hbox1.addWidget(self.camera_select)
    hbox1.addStretch(28)
    hbox1.addWidget(QtGui.QLabel('Pixel Size'))
    hbox1.addWidget(self.pixel_size)
    hbox1.addWidget(QtGui.QLabel(u'\u03bc\u006d'))
    hbox1.addStretch(50)
    grid.addLayout(hbox1, 0, 11, 1, 1)

    hbox2 = QtGui.QHBoxLayout()
    hbox2.addWidget(QtGui.QLabel('Focal Length'))
    hbox2.addWidget(self.focal_length)
    hbox2.addWidget(QtGui.QLabel('mm'))
    hbox2.addStretch()
    hbox2.addWidget(QtGui.QLabel('Image Scale'))
    hbox2.addWidget(self.solve_scale)
    hbox2.addWidget(QtGui.QLabel('Arcsec/Pixel'))
    grid.addLayout(hbox2, 1, 11, 1, 1)

    grid.addWidget(self.check_reference, 1, 3, 1, 1, QtCore.Qt.AlignLeft)

    grid.addWidget(self.image_label,2, 0, 2, 11)
    grid.addWidget(self.histogram_label,2, 11, 1, 1)
    grid.addWidget(self.star_label,3, 11, 1, 1)

    hbox3 = QtGui.QHBoxLayout()
    hbox3.addWidget(self.text_line)
    hbox3.addStretch()
    hbox3.addWidget(self.batch_button)

    grid.addLayout(hbox3,4, 0, 1, 12)
    self.setLayout(grid)

    self.setGeometry(50, 50, -1 ,-1)
    self.setWindowTitle('AstroPhoto')
    self.show()

  def createEvents(self):
    self.open_button.clicked[bool].connect(self.openFiles)
    self.left_arrow_button.clicked[bool].connect(self.previousFile)
    self.right_arrow_button.clicked[bool].connect(self.nextFile)
    self.ref_stars_button.clicked[bool].connect(self.showRef)
    self.img_stars_button.clicked[bool].connect(self.showStars)
    self.flat_button.clicked[bool].connect(self.flat)
    self.save_raw_button.clicked[bool].connect(self.saveDump)
    self.save_tiff_button.clicked[bool].connect(self.saveTiff)
    self.solve_button.clicked[bool].connect(self.solve)
    self.align_button.clicked[bool].connect(self.align)
    self.stack_button.clicked[bool].connect(self.stack)
    self.batch_button.clicked[bool].connect(self.batch)
    self.check_reference.hitButton = self.toggleReference
    self.image_label.mouseMoveEvent = self.imageMagnify
    self.camera_select.currentIndexChanged[int].connect(self.choose_camera)
    self.pixel_size.textChanged[str].connect(self.scale_calculator)
    self.focal_length.textChanged[str].connect(self.scale_calculator)

  def paintEvent(self, e):
    if self.image_update:
      display_image = numpy.copy(self.current_image.rgb16)
      if self.show_solve:
        white = self.current_image.white
        lines_thickness = 5
        for i in self.current_image.correlation:
            cv2.line(display_image, (int(i[4]), int(i[5]-45)), (int(i[4]), int(i[5]-15)), (white, white, white), lines_thickness)
            cv2.line(display_image, (int(i[4]), int(i[5]+45)), (int(i[4]), int(i[5]+15)), (white, white, white), lines_thickness)
            cv2.line(display_image, (int(i[4]-45), int(i[5])), (int(i[4]-15), int(i[5])), (white, white, white), lines_thickness)
            cv2.line(display_image, (int(i[4]+45), int(i[5])), (int(i[4]+15), int(i[5])), (white, white, white), lines_thickness)
        for i in self.current_image.galaxy:
            cv2.circle(display_image, (i[1], i[2]), i[3], (white, white, white), lines_thickness)
            cv2.putText(display_image, 'NGC '+str(i[0]), (i[1], i[2]), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (white, white, white), lines_thickness)

      if self.show_stars:
        white = self.current_image.white
        lines_thickness = 5
        star_number = 0
        for i in self.current_image.stars:
          cv2.rectangle(display_image, (int(i[1]-30), int(i[0]-30)), (int(i[1]+30), int(i[0]+30)), (white, white, white), lines_thickness)
          cv2.putText(display_image, str(star_number), (int(i[1]+40),int(i[0]-40)), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (white, white, white), lines_thickness)
          star_number = star_number + 1

      if self.show_ref:
        white = self.current_image.white
        lines_thickness = 5
        star_number = 0
        for i in self.ref_stars:
            cv2.circle(display_image, (int(i[1]), int(i[0])), 30, (white, 0, 0), lines_thickness)
            cv2.putText(display_image, str(star_number), (int(i[1]+40),int(i[0]-40)), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (white, 0, 0), lines_thickness)
            star_number = star_number + 1

      if self.update_histo:
        white = self.current_image.white
        color = ('b', 'g', 'r')
        fig = matplotlib.pyplot.figure()
        ax = fig.add_subplot(111)
        for i, col in enumerate(color):
          histr = cv2.calcHist([self.current_image.rgb16],[i],None,[white],[0,white])
          ax.plot(histr, color = col)
          ax.set_xlim([0,white])
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        display_hist = numpy.fromstring ( fig.canvas.tostring_rgb(), dtype=numpy.uint8 )
        display_hist.shape = ( h, w, 3 )
        display_hist = numpy.roll ( display_hist, 3, axis = 2 )
        display_hist = cv2.resize(display_hist,(384, 254))
        display_hist = cv2.cvtColor(display_hist, cv2.COLOR_BGR2RGB)
        self.myHistogram = QtGui.QImage(display_hist.astype(numpy.uint8), 384, 254, QtGui.QImage.Format_RGB888)
        self.update_histo = False

      display_image = cv2.resize(display_image,(768, 512))
      display_image = display_image/256
      self.myImage = QtGui.QImage(display_image.astype(numpy.uint8), 768, 512, QtGui.QImage.Format_RGB888)

      self.update()
      self.image_update = False

    self.image_label.setPixmap(QtGui.QPixmap.fromImage(self.myImage))
    self.histogram_label.setPixmap(QtGui.QPixmap.fromImage(self.myHistogram))
    self.star_label.setPixmap(QtGui.QPixmap.fromImage(self.myStar))

  def imageMagnify(self, mouse):
    if hasattr(self, 'current_image'):
      if not self.current_image.error:
        x = mouse.pos().x()*self.current_image.rgb16.shape[1]/768
        y = mouse.pos().y()*self.current_image.rgb16.shape[0]/512
        if x > 16 and x < self.current_image.rgb16.shape[1]-16 and y > 10 and y < self.current_image.rgb16.shape[0]-10:
          display_star = self.current_image.rgb16[y-10:y+10, x-16:x+16, :]/256
          display_star = cv2.resize(display_star,(384, 254))
          self.myStar = QtGui.QImage(display_star.astype(numpy.uint8), 384, 254, QtGui.QImage.Format_RGB888)
          self.update()

  def openFiles(self):
    self.file_list = QtGui.QFileDialog.getOpenFileNames(self, 'Open file')
    self.text_line.setText('Please select your files')
    self.raw_saved = []
    self.batch_button.setEnabled(False)
    if self.file_list.count() > 0:
      self.file_current = 0
      self.img_stars_button.setEnabled(False)
      self.ref_stars_button.setEnabled(False)
      self.show_solve = False
      self.show_ref = False
      self.show_stars = False
      self.reference_image = None
      self.text_line.setText('Loading '+str(self.file_list[self.file_current]))
      self.current_image = AstroImage(str(self.file_list[self.file_current]))
      self.current_image.openFile()
      if not self.current_image.error:
        self.left_arrow_button.setEnabled(True)
        self.right_arrow_button.setEnabled(True)
        self.flat_button.setEnabled(True)
        self.solve_button.setEnabled(True)
        self.save_tiff_button.setEnabled(True)
        self.save_raw_button.setEnabled(True)
        self.check_reference.setEnabled(False)
        self.align_button.setEnabled(False)
        if self.file_current == self.reference_image:
          self.check_reference.setCheckState(QtCore.Qt.Checked)
        else:
          self.check_reference.setCheckState(QtCore.Qt.Unchecked)
        if self.current_image.is_solved:
          self.img_stars_button.setEnabled(True)
          self.check_reference.setEnabled(True)
          if self.reference_image is not None:
            self.align_button.setEnabled(True)
        self.image_update = True
        self.update_histo = True
        self.text_line.setText(self.file_list[self.file_current])
        if self.file_list.count() > 2:
          self.batch_button.setEnabled(True)
      else:
        self.text_line.setText('Error opening file '+self.file_list[self.file_current])
    else:
      self.left_arrow_button.setEnabled(False)
      self.right_arrow_button.setEnabled(False)
      self.flat_button.setEnabled(False)
      self.solve_button.setEnabled(False)
      self.save_tiff_button.setEnabled(False)
      self.save_raw_button.setEnabled(False)
      self.check_reference.setEnabled(False)
      self.img_stars_button.setEnabled(False)
      self.ref_stars_button.setEnabled(False)
      self.reference_image = None
      self.text_line.setText('No file selected')

  def previousFile(self):
    if self.file_current > 0:
      self.img_stars_button.setEnabled(False)
      self.show_solve = False
      self.show_ref = False
      self.show_stars = False
      self.file_current = self.file_current - 1
      self.text_line.setText('Loading '+str(self.file_list[self.file_current]))
      self.current_image = AstroImage(str(self.file_list[self.file_current]))
      self.current_image.openFile()
    if not self.current_image.error:
      if self.file_current == self.reference_image:
        self.check_reference.setCheckState(QtCore.Qt.Checked)
      else:
        self.check_reference.setCheckState(QtCore.Qt.Unchecked)
      self.check_reference.setEnabled(False)
      self.align_button.setEnabled(False)
      if self.current_image.is_solved:
        self.img_stars_button.setEnabled(True)
        self.check_reference.setEnabled(True)
        if self.reference_image is not None:
          self.align_button.setEnabled(True)
      self.image_update = True
      self.update_histo = True
      self.text_line.setText(self.file_list[self.file_current])
    else:
      self.text_line.setText('Error opening file '+self.file_list[self.file_current])

  def nextFile(self):
    if self.file_current < self.file_list.count()-1:
      self.img_stars_button.setEnabled(False)
      self.show_solve = False
      self.show_ref = False
      self.show_stars = False
      self.file_current = self.file_current + 1
      self.text_line.setText('Loading '+str(self.file_list[self.file_current]))
      self.current_image = AstroImage(str(self.file_list[self.file_current]))
      self.current_image.openFile()
    if not self.current_image.error:
      if self.file_current == self.reference_image:
        self.check_reference.setCheckState(QtCore.Qt.Checked)
      else:
        self.check_reference.setCheckState(QtCore.Qt.Unchecked)
      self.check_reference.setEnabled(False)
      self.align_button.setEnabled(False)
      if self.current_image.is_solved:
        self.img_stars_button.setEnabled(True)
        self.check_reference.setEnabled(True)
        if self.reference_image is not None:
          self.align_button.setEnabled(True)
      self.image_update = True
      self.update_histo = True
      self.text_line.setText(self.file_list[self.file_current])
    else:
      self.text_line.setText('Error opening file '+self.file_list[self.file_current])

  def saveDump(self):
    self.text_line.setText('Dump the full object')
    self.current_image.saveDump()
    if not self.current_image.error:
      self.text_line.setText('Dump saved')
      name, extension = os.path.splitext(str(self.file_list[self.file_current]))
      if name+'.raw' not in self.raw_saved:
        self.raw_saved.append(name+'.raw')
      if len(self.raw_saved) > 2:
        self.stack_button.setEnabled(True)
    else:
      self.text_line.setText('Dump failed')

  def saveTiff(self):
    self.text_line.setText('Saving Tiff')
    self.current_image.saveTiff()
    if not self.current_image.error:
      self.text_line.setText('Tiff saved')
    else:
      self.text_line.setText('Tiff failed')

  def flat(self):
    self.text_line.setText('Flatting image')
    if not self.current_image.is_flat:
      self.current_image.flat()
      self.text_line.setText('Flat done')
      self.image_update = True
      self.update_histo = True
    else:
      self.text_line.setText('Image already flat')

  def choose_camera(self):
    self.pixel_size.setText(str(self.camera_list[str(self.camera_select.currentText())][2]))

  def scale_calculator(self):
    try:
      pixel_size = float(self.pixel_size.text())
      focal_length = float(self.focal_length.text())
      solve_scale = pixel_size / focal_length * 206.265
      self.solve_scale.setText(str("%.3f" % solve_scale)) 
      x = float(self.camera_list[str(self.camera_select.currentText())][0])
      y = float(self.camera_list[str(self.camera_select.currentText())][1])
      x_size = x/pixel_size*1000.0*solve_scale/60.0
      y_size = y/pixel_size*1000.0*solve_scale/60.0
      self.text_line.setText('Your FOV is '+str(x_size)+' x '+str(y_size)+' arcmin')
    except:
      self.solve_scale.setText('None')

  def solve(self):
    self.text_line.setText('Solving image')
    if not self.current_image.is_solved:
      try:
        self.current_image.solve(float(self.solve_scale.text()))
      except:
        self.text_line.setText('Please set a correct scale')
        return
      if not self.current_image.is_solved:
        self.text_line.setText('Image not solved')
      else:
        self.text_line.setText('Hashing stars')
        self.current_image.stars_hash()
        self.check_reference.setEnabled(True)
        self.img_stars_button.setEnabled(True)
        if self.reference_image is not None:
          self.align_button.setEnabled(True)
        self.text_line.setText('Image solved')
        self.show_solve = not self.show_solve
        self.image_update = True
    else:
      self.text_line.setText('Image already solved')
      self.show_solve = not self.show_solve
      self.image_update = True

  def align(self):
    self.text_line.setText('Starting alignment')
    self.show_solve = False
    self.show_ref = True
    self.show_stars = True
    if not self.current_image.is_aligned:
      self.text_line.setText('Search best matching stars')
      ref_tree = scipy.spatial.KDTree(self.ref_hash)
      best_img_sequence = 0
      for i in self.current_image.starsHash:
        match = ref_tree.query(i)
        if match[0] < 1e-3:
          break
        best_img_sequence = best_img_sequence + 1
      best_ref_sequence = match[1]
      self.text_line.setText('Reference sequence ' + str(best_ref_sequence) + ' matches image sequence ' + str(best_img_sequence) + ' with an error of ' + str(match[0]))
      ref_stars = self.ref_sequence[best_ref_sequence].astype(int)
      img_stars = self.current_image.starsSequence[best_img_sequence].astype(int)

      ref_stars_center = numpy.array([self.ref_stars[ref_stars][:,0].mean(), self.ref_stars[ref_stars][:,1].mean()])
      img_stars_center = numpy.array([self.current_image.stars[img_stars][:,0].mean(), self.current_image.stars[img_stars][:,1].mean()])
      shift = numpy.array([self.current_image.width/2.0, self.current_image.height/2.0]) - img_stars_center
      ref_angle = 0.0
      max_dist = 0.0
      for i in ref_stars:
        distance = (ref_stars_center[0] - self.ref_stars[i][0])**2 + (ref_stars_center[1] - self.ref_stars[i][1])**2
        if distance > max_dist:
          max_dist = distance
          angle_star = i
      ref_angle = numpy.arctan2(self.ref_stars[angle_star][1]-ref_stars_center[1], self.ref_stars[angle_star][0]-ref_stars_center[0])

      img_angle = 0.0
      max_dist = 0.0
      for i in img_stars:
        distance = (img_stars_center[0] - self.current_image.stars[i][0])**2+(img_stars_center[1] - self.current_image.stars[i][1])**2
        if distance > max_dist:
          max_dist = distance
          angle_star = i
      img_angle = numpy.arctan2(self.current_image.stars[angle_star][1]-img_stars_center[1], self.current_image.stars[angle_star][0]-img_stars_center[0])

      self.text_line.setText('Align')
      self.current_image.translate(shift[0], shift[1])
      self.current_image.rotate(ref_angle-img_angle)  
      shift = ref_stars_center - numpy.array([self.current_image.width/2.0, self.current_image.height/2.0])
      self.current_image.translate(shift[0], shift[1])
      self.current_image.crop()
      self.text_line.setText('Re-match stars after alignment')
      self.current_image.stars = numpy.copy(self.ref_stars)
      j=0
      for i in self.current_image.stars:
         self.current_image.stars[j][2] = self.current_image.rgb16[i[0]-30:i[0]+30,i[1]-30:i[1]+30,:].sum()
         j=j+1

      self.current_image.is_aligned = True
      self.current_image.is_solved = False
      self.image_update = True
      self.text_line.setText('Alignment done')
    else:
      self.text_line.setText('Image already aligned')

  def stack(self):
    self.show_solve = False
    average = numpy.empty(shape=self.current_image.rgb16.shape, dtype=float)
    average.fill(0.0)
    stdev = numpy.empty(shape=self.current_image.rgb16.shape, dtype=float)
    stdev.fill(0.0)
    stack = numpy.empty(shape=self.current_image.rgb16.shape, dtype=float)
    stack.fill(0.0)
    count = numpy.empty(shape=self.current_image.rgb16.shape, dtype=float)
    count.fill(1.0)
    
    frame_number = 1.0
    for filename in self.raw_saved:
      self.current_image = AstroImage(filename)
      self.current_image.openFile()
      print 'Loading '+filename+' for average and stdev calculation'

      delta = self.current_image.rgb16.astype(numpy.float) - average
      average = average + delta/frame_number
      stdev = stdev + delta*(self.current_image.rgb16.astype(numpy.float) - average)
      frame_number = frame_number + 1.0

    stdev = numpy.sqrt(stdev/frame_number)
      
    tolerance = 1.5
    for filename in self.raw_saved:
      self.current_image = AstroImage(filename)
      self.current_image.openFile()
      print 'Loading '+filename+' for stack'
      mask = (numpy.fabs(self.current_image.rgb16 - average) <= tolerance * stdev).astype(numpy.float)
      stack = stack + mask*self.current_image.rgb16.astype(numpy.float)
      count = count + mask
      #os.remove(filename)

    stack = stack / count
    self.current_image.rgb16 = stack.astype(numpy.uint16)
    self.current_image.filename = 'final.tiff'
    self.image_update = True
    self.solve()
    self.text_line.setText('Stack complete')

  def showStars(self):
    self.show_stars = not self.show_stars
    self.image_update = True

  def showRef(self):
    self.show_ref = not self.show_ref
    self.image_update = True

  def toggleReference(self, message):
    if self.check_reference.checkState() == QtCore.Qt.Unchecked:
      self.ref_hash = self.current_image.starsHash
      self.ref_sequence = self.current_image.starsSequence
      self.ref_stars = self.current_image.stars
      self.ref_stars_button.setEnabled(True)
      self.align_button.setEnabled(True)
      self.reference_image = self.file_current
      self.text_line.setText('Image ' + str(self.file_list[self.file_current]) + ' selected as reference for alignment')
      self.check_reference.setText('Set')
      self.check_reference.setCheckState(QtCore.Qt.Checked)
    else:
      self.ref_stars_button.setEnabled(False)
      self.align_button.setEnabled(False)
      self.reference_image = None
      self.text_line.setText('Reference for alignment removed')
      self.check_reference.setText('Not Set')
      self.check_reference.setCheckState(QtCore.Qt.Unchecked)
    return 0

  def batch_thread(self):
    for i in range(0, self.file_list.count()):
      self.solve()
      if i == 0:
        self.toggleReference(None)
      self.flat()
      self.align()
      self.saveDump()
      self.nextFile()

    self.stack()
    self.saveTiff()

  def batch(self):
    tr = threading.Thread(target=self.batch_thread)
    tr.start()

def main():
    app = QtGui.QApplication(sys.argv)
    ex = AstroUI()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
