import pandas as pd
import numpy as np
import astropy.units as u
from astropy.table import QTable
from astropy.time import Time
from astropy.coordinates import SkyCoord
import sunpy
import sunpy.map
from sunpy.coordinates import (propagate_with_solar_surface,
                               Helioprojective)
from eispac.download.convert import tai2utc, utc2tai
from eispac.download import download_db
from eispac.download.eis_obs_struct import EIS_DB
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sqlite3
import os
import hvpy
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
import requests
from regions import RectangleSkyRegion



class HMFList:
    def __init__(self, list_file='./hard-microflares_with_location_version-2024-02-25.csv',
                 eis_list_file='/home/yjzhu/ssw/hinode/eis/database/catalog/eis_cat.sqlite'):
        self.hmf_qtable = _df_to_QTable(pd.read_csv(list_file))
        
        if not os.path.exists(eis_list_file):
            if input('EIS list file not found. Download it in the current directory now? (y/n)') == 'y':
                self.eis_list_file = download_db.download_eis_db()
            else:
                raise FileNotFoundError('EIS list file not found.')
        else:
            self.eis_list_file = eis_list_file
        
        self.eis_db = EIS_DB(self.eis_list_file)
        self.hmf_events = {}
        self.eis_lists_linked = {}
        self.iris_lists_linked = {}
    
    def get_eis_list(self, fovx_buffer=10*u.arcsec, fovy_buffer=10*u.arcsec,
                     date_beg_buffer=1*u.h, date_end_buffer=1*u.h):
        for ii, (date_beg, date_end, date_peak, event_coord, goes_class)  in \
            enumerate(self.hmf_qtable.iterrows('SO_start_time', 'SO_end_time', 'SO_NTH_peak_time',
                         'SO_earth_skycoord', 'GOES_class_peak')):
            eis_str_shortened = _get_masked_eis_list(date_beg, date_end, date_beg_buffer, 
                                                   date_end_buffer, self.eis_db)
            
            eis_str_selected = []
            
            if len(eis_str_shortened) != 0:
                # print(eis_list_masked.loc[:,['date_obs','filename']])
                
                for eis_str in eis_str_shortened:
                    if _in_fov_eis(event_coord, eis_str, fovx_buffer, fovy_buffer):
                        eis_str_selected.append(eis_str)
                if len(eis_str_selected) != 0:
                    self.hmf_qtable[ii]['has_eis'] = True
                    self.eis_lists_linked[ii] = eis_str_selected
                    if ii in self.iris_lists_linked.keys():
                        self.hmf_events[ii].eis_list = eis_str_selected
                    else:
                        self.hmf_events[ii] = HMFEvent(date_beg, date_end, date_peak, event_coord, goes_class, list_id=ii,
                                                    eis_list=eis_str_selected)

    def get_iris_list(self, fovx_buffer=15*u.arcsec, fovy_buffer=15*u.arcsec,
                      date_beg_buffer=1*u.h, date_end_buffer=1*u.h):
        for ii, (date_beg, date_end, date_peak, event_coord, goes_class)  in \
            enumerate(self.hmf_qtable.iterrows('SO_start_time', 'SO_end_time', 'SO_NTH_peak_time',
                         'SO_earth_skycoord', 'GOES_class_peak')):

            iris_dict_shortened = _get_masked_iris_list(date_beg, date_end, date_beg_buffer, date_end_buffer)
            iris_dict_selected = []

            if len(iris_dict_shortened) != 0:
                for event in iris_dict_shortened['Events']:
                    if _in_fov_iris(event_coord, event, fovx_buffer, fovy_buffer):
                        iris_dict_selected.append(event)
                if len(iris_dict_selected) != 0:
                    self.hmf_qtable[ii]['has_iris'] = True
                    self.iris_lists_linked[ii] = iris_dict_selected
                    if ii in self.eis_lists_linked.keys():
                        self.hmf_events[ii].iris_list = iris_dict_selected
                    else:
                        self.hmf_events[ii] = HMFEvent(date_beg, date_end, date_peak, event_coord, goes_class, list_id=ii,
                                                    iris_list=iris_dict_selected)



                    
    def get_fov_plots(self, fovx=400*u.arcsec, fovy=400*u.arcsec, save_path=None):
        for event_key in self.hmf_events:
            self.hmf_events[event_key].save_all_fov_eis_iris(fovx=fovx, fovy=fovy, save_path=save_path)
                    
            

class HMFEvent:
    def __init__(self, date_beg, date_end, date_peak, coord, goes_class, list_id=None,
                 eis_list=None, iris_list=None):
        self.date_beg = date_beg
        self.date_end = date_end
        self.date_peak = date_peak
        self.coord = coord
        self.goes_class = goes_class
        self.list_id = list_id
        self.eis_list = eis_list
        self.iris_list = iris_list

    def save_all_fov_eis_iris(self, fovx=400*u.arcsec, fovy=400*u.arcsec,
                              save_path=None):
        if save_path is None:
            save_path = os.path.join('./fov_plot_sav/',f'{self.date_peak.isot[:-4].replace(':','_')}_eis_iris_fov.pdf')
        if (self.eis_list is not None and len(self.eis_list) != 0) and (self.iris_list is not None and len(self.iris_list) != 0):
            with PdfPages(save_path) as pdf:
                self.plot_fov_eis(fovx=fovx, fovy=fovy, save_obj=pdf)
                self.plot_fov_iris(fovx=fovx, fovy=fovy, save_obj=pdf)
        elif (self.eis_list is not None and len(self.eis_list) != 0):
            with PdfPages(save_path) as pdf:
                self.plot_fov_eis(fovx=fovx, fovy=fovy, save_obj=pdf)
        elif (self.iris_list is not None and len(self.iris_list) != 0):
            with PdfPages(save_path) as pdf:
                self.plot_fov_iris(fovx=fovx, fovy=fovy, save_obj=pdf)

    def plot_fov_eis(self, fovx=400*u.arcsec, fovy=400*u.arcsec,
                 save_obj=None, tmp_file_path="./hvpy_tmp/"):
        
        filepath = Path(tmp_file_path)/ f'aia_193_{self.date_peak.isot.replace(':','_')}.jp2'
        if not filepath.exists():
            filepath = hvpy.save_file(hvpy.getJP2Image(date=self.date_peak.to_datetime(),
                                                    sourceId=hvpy.DataSource.AIA_193),
                                                    filename=filepath)
        else:
            pass
        
        aia_map = sunpy.map.Map(filepath)
        aia_map = aia_map.submap(SkyCoord(self.coord.Tx - fovx/2, self.coord.Ty - fovy/2,
                                           frame=aia_map.coordinate_frame),
                                 top_right=SkyCoord(self.coord.Tx + fovx/2, self.coord.Ty + fovy/2,
                                          frame=aia_map.coordinate_frame))
        
        for eis_str in self.eis_list:
            fig = plt.figure(figsize=(10,8),layout='constrained')
            fig_top, fig_bottom = fig.subfigures(2,1, height_ratios=[6.5,1.5], hspace=0)
            fig1, fig2 = fig_top.subfigures(1,2, width_ratios=[6,4], wspace=0,)
            ax_img = fig1.add_subplot(projection=aia_map)
            aia_map.plot(axes=ax_img, title=f'SDO/AIA 19.3 nm {aia_map.date.isot} \n' + f'HMF Peak Time: {self.date_peak.isot}')

            bottom_left_coord = SkyCoord((eis_str.xcen - eis_str.fovx/2)*u.arcsec, 
                                         (eis_str.ycen - eis_str.fovy/2)*u.arcsec,
                                         frame=Helioprojective(obstime=Time(eis_str.date_obs), observer='earth'))
            
            top_right_coord = SkyCoord((eis_str.xcen + eis_str.fovx/2)*u.arcsec,
                                        (eis_str.ycen + eis_str.fovy/2)*u.arcsec,
                                        frame=Helioprojective(obstime=Time(eis_str.date_obs), observer='earth'))
            
            with propagate_with_solar_surface():
                ax_img_lim = ax_img.axis()
                aia_map.draw_quadrangle(bottom_left_coord, top_right=top_right_coord, edgecolor='red')
                ax_img.plot_coord(self.coord, 'x', color='blue', label='HMF Peak Location')
                ax_img.axis(ax_img_lim)
            
            ax_text = fig2.add_subplot()

            info_text = f'Hard Microflare (HMF) Info:\n' + \
                        f'HMF start time: {self.date_beg.isot[:-4]}\n' + \
                        f'HMF end time: {self.date_end.isot[:-4]}\n' + \
                        f'HMF peak time: {self.date_peak.isot[:-4]}\n' + \
                        f'GOES class: {self.goes_class}\n' + \
                        f'\n' + \
                        f'EIS Observation Info:\n' + \
                        f'filename: {eis_str.filename}\n' + \
                        f'date_obs: {eis_str.date_obs[:-4]}\n' + \
                        f'date_end: {eis_str.date_end[:-4]}\n' + \
                        f'xcen: {eis_str.xcen:<6.2f}"' +' '*7 + \
                        f'ycen: {eis_str.ycen:<.2f}"\n' + \
                        f'fovx: {eis_str.fovx:>6.2f}"' + ' '*7 + \
                        f'fovy: {eis_str.fovy:>6.2f}"\n' + \
                        f'stepsize: {eis_str.scan_fm_stepsize:>5.2f}"' + ' '*4 + \
                        f'nsteps: {eis_str.scan_fm_nsteps:>5}\n' + \
                        f'slit_ind: {eis_str.slit_index:<10}' + \
                        f'exptime: {np.nanmean(eis_str.exptime):>.2f} s\n' + \
                        f'stud_acr: {eis_str.stud_acr:<24}' + \
                        f'\n' + \
                        f'EIS Windows Info:\n' + \
                        f'{"Index":<5} {"Title":<20} {"Wavemin":<9} {"Wavemax":<9}\n'
            
            for ii in range(len(eis_str.ll_title)):
                info_text += f"{ii:<5} {eis_str.ll_title[ii]:<20} " + \
                             f"{eis_str.wavemin[ii]:<9.2f} {eis_str.wavemax[ii]:<9.2f}\n"

                            
            
            ax_text.text(0.05, 0.95, info_text, fontsize=10, ha='left', va='top',
                         transform=ax_text.transAxes, linespacing=1.3, family='monospace')
            
            ax_text.axis('off')

            ax_timeline = fig_bottom.add_subplot()
            ax_time_xlim_left = np.min([Time(eis_str.date_obs), self.date_beg])
            ax_time_xlim_right = np.max([Time(eis_str.date_end), self.date_end])
            ax_time_dur = ax_time_xlim_right - ax_time_xlim_left
            ax_time_xlim_left -= ax_time_dur*0.1 # leave some space on the left
            ax_time_xlim_right += ax_time_dur*0.1 # leave some space on the right

            ax_timeline.axvspan(Time(eis_str.date_obs).to_datetime(), Time(eis_str.date_end).to_datetime(), facecolor='#F6C555', alpha=0.5,
                                edgecolor='none')
            
            ax_timeline.axvline(self.date_beg.to_datetime(), color='#0089A7', linestyle='--', label='HMF Start Time', alpha=0.8)
            ax_timeline.axvline(self.date_end.to_datetime(), color='#8A6BBE', linestyle='--', label='HMF End Time', alpha=0.8)
            ax_timeline.axvline(self.date_peak.to_datetime(), color='#E83015', linestyle='-', label='HMF Peak Time', alpha=0.8)

            ax_timeline.set_xlim(ax_time_xlim_left.to_datetime(), ax_time_xlim_right.to_datetime())
            ax_timeline.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

            ax_timeline.set_xlabel('Time (hh:mm:ss) on ' + ax_time_xlim_left.isot[:10])
            ax_timeline.tick_params(axis='y', left=False, labelleft=False,)
            

            if save_obj is not None:
                save_obj.savefig(fig)
                plt.close(fig)
            else:
                plt.show()

    def plot_fov_iris(self, fovx=400*u.arcsec, fovy=400*u.arcsec,
                 save_obj=None, tmp_file_path="./hvpy_tmp/"):
        
        filepath = Path(tmp_file_path)/ f'aia_193_{self.date_peak.isot.replace(':','_')}.jp2'
        if not filepath.exists():
            filepath = hvpy.save_file(hvpy.getJP2Image(date=self.date_peak.to_datetime(),
                                                    sourceId=hvpy.DataSource.AIA_193),
                                                    filename=filepath)
        else:
            pass
        
        aia_map = sunpy.map.Map(filepath)
        aia_map = aia_map.submap(SkyCoord(self.coord.Tx - fovx/2, self.coord.Ty - fovy/2,
                                           frame=aia_map.coordinate_frame),
                                 top_right=SkyCoord(self.coord.Tx + fovx/2, self.coord.Ty + fovy/2,
                                          frame=aia_map.coordinate_frame))
        
        raster_color = ['#F6C555', '#90B44B']
        
        for event in self.iris_list:
            fig = plt.figure(figsize=(10,8),layout='constrained')
            fig_top, fig_bottom = fig.subfigures(2,1, height_ratios=[6.5,1.5], hspace=0)
            fig1, fig2 = fig_top.subfigures(1,2, width_ratios=[6,4], wspace=0,)
            ax_img = fig1.add_subplot(projection=aia_map)
            aia_map.plot(axes=ax_img, title=f'SDO/AIA 19.3 nm {aia_map.date.isot} \n' + f'HMF Peak Time: {self.date_peak.isot}')

            iris_fov_cen = SkyCoord(event['xCen']*u.arcsec, event['yCen']*u.arcsec, 
                                    frame=Helioprojective(obstime=Time(event['startTime']), observer='earth'))
            
            iris_fov_region = RectangleSkyRegion(center=iris_fov_cen, width=np.abs(event['raster_fovx'] + 1e-3)*u.arcsec,
                                                    height=event['raster_fovy']*u.arcsec, angle=-event['roll_angle']*u.deg)
            
            with propagate_with_solar_surface():
                ax_img_lim = ax_img.axis()
                iris_fov_region.to_pixel(aia_map.wcs).plot(ax=ax_img, edgecolor='red')
                ax_img.plot_coord(self.coord, 'x', color='blue', label='HMF Peak Location')
                ax_img.axis(ax_img_lim)
            
            ax_text = fig2.add_subplot()

            for group in event['groups']:
                if group['group_name'] == 'Raster':
                    stepsize = group['raster_stepsize']
                    if 'max_exptime' in group.keys():
                        exptime = group['max_exptime']
                    elif 'min_exptime' in group.keys():
                        exptime = group['min_exptime']
                    elif 'mean_exptime' in group.keys():
                        exptime = group['mean_exptime']
                    else:
                        exptime = np.nan
                    nsteps = group['raster_numsteps']
                    nrasters = group['numrasters']
                    rad_cad = group['cadence_avg_asrun']


            info_text = f'Hard Microflare (HMF) Info:\n' + \
                        f'HMF start time: {self.date_beg.isot[:-4]}\n' + \
                        f'HMF end time: {self.date_end.isot[:-4]}\n' + \
                        f'HMF peak time: {self.date_peak.isot[:-4]}\n' + \
                        f'GOES class: {self.goes_class}\n' + \
                        f'\n' + \
                        f'IRIS Observation Info:\n' + \
                        f'goal: {event['goal']}\n' + \
                        f'startTime: {event["startTime"]}\n' + \
                        f'stopTime: {event["stopTime"]}\n' + \
                        f'xcen: {event["xCen"]:<6.2f}"' +' '*7 + \
                        f'ycen: {event["yCen"]:<6.2f}"\n' + \
                        f'fovx: {event["raster_fovx"]:>6.2f}"' + ' '*7 + \
                        f'fovy: {event["raster_fovy"]:>6.2f}"\n' + \
                        f'stepsize: {stepsize:>5.2f}"' + ' '*4 + \
                        f'nsteps: {nsteps:<5}\n' + \
                        f'exptime: {exptime:>5.2f} s' + ' '*4 + \
                        f'nraster: {nrasters:<5}\n' + \
                        f'roll_angle: {event["roll_angle"]:>2.0f} deg  ' + \
                        f'ras_cad: {rad_cad:>5.1f} s\n' + \
                        "\n" + \
                        f'IRIS Spectral Line Info (for Reference):\n'
            
            for ii, spectral_line in enumerate(event['spectral_window_names'][1:-1].split(',')):
                if ii%2 == 0:
                    info_text += f'{spectral_line:<15}'
                else:
                    info_text += f'{spectral_line}\n'
            
            ax_text.text(0.05, 0.95, info_text, fontsize=10, ha='left', va='top',
                         transform=ax_text.transAxes, linespacing=1.3, family='monospace',
                         wrap=True)
            
            ax_text.axis('off')

            ax_timeline = fig_bottom.add_subplot()
            ax_time_xlim_left = np.min([Time(event['startTime']), self.date_beg])
            ax_time_xlim_right = np.max([Time(event['stopTime']), self.date_end])
            ax_time_dur = ax_time_xlim_right - ax_time_xlim_left
            ax_time_xlim_left -= ax_time_dur*0.1 # leave some space on the left
            ax_time_xlim_right += ax_time_dur*0.1 # leave some space on the right

            
        
            for ii in range(nrasters):
                if stepsize < 1e-2: # sit-and-stare
                    raster_start = Time(event['startTime']) + ii*rad_cad*nsteps*u.s
                    raster_end = Time(event['startTime']) + (ii + 1)*rad_cad*nsteps*u.s
                else:
                    raster_start = Time(event['startTime']) + ii*rad_cad*u.s
                    raster_end = Time(event['startTime']) + (ii + 1)*rad_cad*u.s

                ax_timeline.axvspan(raster_start.to_datetime(), raster_end.to_datetime(), facecolor=raster_color[ii%2], alpha=0.5,
                                    edgecolor='none')
            
            ax_timeline.axvline(self.date_beg.to_datetime(), color='#0089A7', linestyle='--', label='HMF Start Time', alpha=0.8)
            ax_timeline.axvline(self.date_end.to_datetime(), color='#8A6BBE', linestyle='--', label='HMF End Time', alpha=0.8)
            ax_timeline.axvline(self.date_peak.to_datetime(), color='#E83015', linestyle='-', label='HMF Peak Time', alpha=0.8)
            
            ax_timeline.set_xlim(ax_time_xlim_left.to_datetime(), ax_time_xlim_right.to_datetime())
            ax_timeline.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

            ax_timeline.set_xlabel('Time (hh:mm:ss) on ' + ax_time_xlim_left.isot[:10])
            ax_timeline.tick_params(axis='y', left=False, labelleft=False,)
                



            if save_obj is not None:
                save_obj.savefig(fig)
                plt.close(fig)
            else:
                plt.show()


def _df_to_QTable(df):
    SO_start_time = Time(df['SO_start_time'].values.tolist())
    SO_end_time = Time(df['SO_end_time'].values.tolist())
    SO_NTH_peak_time = Time(df['SO_NTH_peak_time'].values.tolist())

    SO_earth_skycoord = SkyCoord(df['x_loc_Earth'].values*u.arcsec,
                                 df['y_loc_Earth'].values*u.arcsec,
                                 obstime=SO_NTH_peak_time,
                                 observer='earth',
                                 frame='helioprojective')
    
    has_eis = np.zeros(len(df), dtype=bool)
    has_iris = np.zeros(len(df), dtype=bool)

    rest_keys = [key for key in df.keys() if key not in ['SO_start_time', 'SO_end_time',
                                                            'SO_NTH_peak_time', 'x_loc_Earth', 'y_loc_Earth']]
    rest_values = [df[key].values.tolist() for key in rest_keys]
    
    return QTable([SO_start_time, SO_end_time, 
                   SO_NTH_peak_time, SO_earth_skycoord, has_eis, has_iris] + rest_values,
                  names=('SO_start_time', 'SO_end_time', 'SO_NTH_peak_time',
                         'SO_earth_skycoord', 'has_eis', 'has_iris', *rest_keys))

def _get_masked_eis_list(date_beg, date_end, date_beg_buffer, date_end_buffer, eis_db):
    date_range = Time([date_beg - date_beg_buffer, date_end + date_end_buffer]).isot

    eis_db.get_by_date(date_range[0], date_range[1])

    return eis_db.eis_str

def _get_masked_iris_list(date_beg, date_end, date_beg_buffer, date_end_buffer):
    date_range = Time([date_beg - date_beg_buffer, date_end + date_end_buffer]).isot
    request_url = f'https://www.lmsal.com/hek/hcr?cmd=search-events-corr&outputformat=json&' + \
    f'startTime={date_range[0]}&stopTime={date_range[1]}&instrument=IRIS&hasData=true'

    iris_dict = requests.get(request_url).json()
    
    # for events in iris_dict['Events']:
    #     print(events['startTime'])
    #     print(events['stopTime'])
    
    return iris_dict

def _in_fov_eis(coord, eis_struct, fovx_buffer, fovy_buffer):
    
    eis_coordinate_frame = Helioprojective(obstime=Time(eis_struct.date_obs), observer='earth')
    with propagate_with_solar_surface():
        coord = coord.transform_to(eis_coordinate_frame)

    bottom_left_x = (eis_struct.xcen - eis_struct.fovx/2)*u.arcsec
    bottom_left_y = (eis_struct.ycen - eis_struct.fovy/2)*u.arcsec
    top_right_x = (eis_struct.xcen + eis_struct.fovx/2)*u.arcsec
    top_right_y = (eis_struct.ycen + eis_struct.fovy/2)*u.arcsec

    return (coord.Tx > bottom_left_x - fovx_buffer) & \
           (coord.Tx < top_right_x + fovx_buffer) & \
           (coord.Ty > bottom_left_y - fovy_buffer) & \
           (coord.Ty < top_right_y + fovy_buffer)

def _in_fov_iris(coord, iris_event, fovx_buffer, fovy_buffer):
    iris_coordinate_frame = Helioprojective(obstime=Time(iris_event['startTime']), observer='earth')
    with propagate_with_solar_surface():
        coord = coord.transform_to(iris_coordinate_frame)
    
    # some times IRIS has an non-zero roll angle, to compensate for that, 
    # we first rotate the point back, so that we have a "normal" rectangle
    # then we can calculate the bottom left and top right corner of the rectangle
    # and check if the point is in the rectangle
    # also note that the roll_angle is in degrees, clockwise is positive

    coord_vec = np.array([coord.Tx.value - iris_event['xCen'], coord.Ty.value - iris_event['yCen']])
    roll_angle = np.deg2rad(iris_event['roll_angle'])

    rot_mat = np.array([[np.cos(roll_angle), -np.sin(roll_angle)],
                        [np.sin(roll_angle), np.cos(roll_angle)]])
    
    coord_vec_rot = np.dot(rot_mat, coord_vec)
    
    bottom_left_x = - iris_event['raster_fovx']/2
    bottom_left_y = - iris_event['raster_fovy']/2
    top_right_x = iris_event['raster_fovx']/2
    top_right_y = iris_event['raster_fovy']/2


    return (coord_vec_rot[0] > bottom_left_x - fovx_buffer.to_value(u.arcsec)) & \
            (coord_vec_rot[0] < top_right_x + fovx_buffer.to_value(u.arcsec)) & \
            (coord_vec_rot[1] > bottom_left_y - fovy_buffer.to_value(u.arcsec)) & \
            (coord_vec_rot[1] < top_right_y + fovy_buffer.to_value(u.arcsec))



if __name__ == '__main__':
    # date = ['2018-09-17T19:46:25.000', '2018-09-17 19:46:25.000',
    #         '2018-sep-17T19:46:25.000','2018-sep-17 19:46:25.000',
    #         '20180917 19:46:25.000', '17-Sep-2018 19:46:25.000',
    #         '17-Sep-2018 19:46:25', '2018/09/17 19:46:25.000']
    # for day in date:
    #     tai = utc2tai(day)
    #     utc = tai2utc(tai)
    #     print(f'{day:>30} {tai:20.0f} {utc:>30}')
    hmflist = HMFList(list_file='./hard-microflares_with_location_version-2024-02-25.csv')
    hmflist.get_eis_list()
    hmflist.get_iris_list()
    # hmflist.get_eis_list()
    hmflist.get_fov_plots()



