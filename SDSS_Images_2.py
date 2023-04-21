
from memory_profiler import profile
from astroquery.sdss import SDSS
from astroquery.exceptions import TimeoutError as AQTimeoutError
from urllib.error import URLError
from astropy.table import Table
import pandas as pd
import numpy as np
# astronomy things
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle 
import astropy.units as u
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from astropy.visualization import MinMaxInterval, SqrtStretch, ImageNormalize, ZScaleInterval, LogStretch, LinearStretch, SinhStretch, AsinhStretch, make_lupton_rgb
from photutils.segmentation import detect_sources, detect_threshold, SourceFinder, SourceCatalog

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.optimize import curve_fit
from PIL import Image
from sklearn.preprocessing import MinMaxScaler

import time

from pathlib import Path


def scale_img(img_data):

  # scale the image so its viewable
  interval = ZScaleInterval(max_iterations=10)
  vmin, vmax = interval.get_limits(img_data)
  norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch(), clip=True)
  img_data = norm(img_data)

  return img_data


def gaussian_2d(xy, A, mu_x, sig_x, mu_y, sig_y, theta, offset):
    x, y = xy
    a = (   np.cos(theta) ** 2 )   / (2 * sig_x ** 2) + (np.sin(theta) ** 2) / (2 * sig_y ** 2)
    b = (  -np.sin(2*theta)    )   / (4 * sig_x ** 2) + (np.sin(2*theta)   ) / (4 * sig_y ** 2)
    c = (   np.sin(theta) ** 2 )   / (2 * sig_x ** 2) + (np.cos(theta) ** 2) / (2 * sig_y ** 2)
    g =  offset + A * np.exp( -1 * ( a * (x - mu_x) ** 2  + 2 * b * (x - mu_x)*(y - mu_y) + c * (y - mu_y) ** 2 ))
    return g.ravel()
    
    
    
df = pd.read_csv("data/star_classification.csv")
bands = ['u', 'g', 'r', 'i', 'z']




def fit_2d_gaussian(chip_data, object, band, ax=None, reshape=True):
    center_chip_x, center_chip_y = chip_data.shape[1] / 2, chip_data.shape[0] / 2
    
    ellipse = None
    try:
        # fit a 2-d gaussian to get a sigma for 99% of the object
        x_fit = np.arange(0, chip_data.shape[0])
        y_fit = np.arange(0, chip_data.shape[1])
        x_fit, y_fit = np.meshgrid(x_fit, y_fit)
        initial_guess = (1, center_chip_x, 1, center_chip_y, 1, 0, 0)
        popt, pcov = curve_fit(gaussian_2d, (x_fit, y_fit), chip_data.ravel(), p0=initial_guess)
        A, mu_x, sig_x, mu_y, sig_y, theta, offset = popt
        theta_deg = 180 / np.pi * theta
        
        
        # add a gaussian 
        n_sig = 3
        dy = abs(sig_y * (n_sig + 2))
        dx = abs(sig_x * (n_sig + 2))
        
        if reshape:
            dy = dy if dy >= 5 else 5
            dx = dx if dx >= 5 else 5
            chip_data = chip_data[int(center_chip_y - dy): int(center_chip_y + dy), 
                                int(center_chip_x - dx): int(center_chip_x + dx)]
            
            
            mu_x = mu_x - (center_chip_x - dx)
            mu_y = mu_y - (center_chip_y - dy)
         
        if ax:  
            ellipse = patches.Ellipse((mu_x, mu_y), sig_y * n_sig*2, sig_x * n_sig*2, theta_deg, fill=False, color='red', zorder=2) # n_sig * 2 because this takes diameter
            ax.add_patch(ellipse)

            
        print(f"{chip_data.shape = }")
        
        return ax, chip_data, ellipse
    except Exception as e:
        # we werent able to fit a gaussian. we should still make the chip become 20x20
        if reshape:
            chip_data = chip_data[int(center_chip_y - 10):int(center_chip_y + 10), int(center_chip_x-10):int(center_chip_x+10 )]
        print(f"EXCEPTION {e}. SKIPPING.")
        return ax, chip_data, ellipse
        


from requests.exceptions import ConnectionError

bad_idx_file = Path("./bad_idxs.txt")
if bad_idx_file.exists():
    with open(bad_idx_file, "rb") as f:
        content = f.readlines()
    bad_idxs = [int(idx) for idx in content]
else:
    bad_idxs = []
def get_imgs(df, last_idx, mod=0, n_para=1, max_idx = 100_000, data_dir=Path("./chips2")):
    try:
        # figure out which ones we want to make objects for
        idxs = np.arange(0,len(df),1)
        # idxs_to_generate = idxs
        idxs_to_generate = idxs[idxs%n_para == mod]
        
        # data_dir = Path("G:/My Drive/MIDS/207/SDSS-Classification/chips")
        # data_dir = Path("./chips")
        print(data_dir)
        print(f"{data_dir.exists() = }")
        contents = data_dir.glob("**/*.jpeg")
        contents = [content for content in contents]
            
        
        generated_idxs = list(set([int(content.name.split('-')[0]) for content in contents]))
        idxs_to_generate = idxs_to_generate[~np.isin(idxs_to_generate, generated_idxs)]
        idxs_to_generate = idxs_to_generate[idxs_to_generate < max_idx]

        # # check that images for all bands have been generated
        # print("Checking for missing images...")
        # for gen_idx in generated_idxs:
        #     print(f"Testing {gen_idx = }", end='\r')
        #     for band in bands:
        #         if not (data_dir / f"{gen_idx}-{band}.jpeg").exists():
        #             print(f"MISSING {gen_idx}-{band}.jpeg")
        #             idxs_to_generate = np.append(idxs_to_generate, gen_idx)
        #             break
            
            

        for obj_idx, row in df.iterrows():
            print(f"Testing {obj_idx = }")
            if obj_idx not in idxs_to_generate or obj_idx in bad_idxs:
                print(f"Skipping {obj_idx = }")
                continue
            print(f"Running OBJ_IDX = {obj_idx}")
            print(f"{obj_idx / len(df)*100:.1f}")
            object = row
            run_id = object['run_ID']
            rerun_id = object['rerun_ID']
            cam_col = object['cam_col']
            field_id = object['field_ID']
            spec_obj_id = object['spec_obj_ID']
            obj_class = object['class']

            # get coordinates of object RA/DEC
            ra = Angle(object['alpha'], unit=u.deg)
            dec = Angle(object['delta'], unit=u.deg)
            c = SkyCoord(ra, dec, frame='icrs')

            xids_table = SDSS.query_region(c) # this can return multiple items
            xids = xids_table.to_pandas()
            xids = xids.query(f"run == {run_id} and rerun == {rerun_id} and camcol == {cam_col} and field == {field_id}")
            xids["distance_from_object"] = np.sqrt((xids['ra'] - ra)**2 + (xids['dec'] - dec)**2)
            best_plate = xids[xids['distance_from_object'] == xids['distance_from_object'].min()]

            best_plate = Table.from_pandas(best_plate.drop("distance_from_object", axis=1))

            # LOOP OVER ALL IMAGES HERE
            img_data_buffer = [SDSS.get_images(matches=best_plate, band = band) for band in bands]
            all_bands_img_data = np.ndarray(shape=(img_data_buffer[0][0][0].data.shape[0], img_data_buffer[0][0][0].data.shape[1], len(bands)))
            all_bands_sources = []

            # get all sources in all bands
            try:
                print("Getting All Sources...")
                for idx, (img, band) in enumerate(zip(img_data_buffer,bands)):

                    header = img[0][0].header
                    wcs = WCS(header)
                    obj_pix_x, obj_pix_y = wcs.world_to_pixel(c)
                    
                    # find the bounding box/boxes that contain the object
                    img_data = img[0][0].data.copy()
                    # background subtract image
                    bg_mean, bg_median, bg_std = sigma_clipped_stats(img_data, sigma=3.0)
                    img_data = (img_data - bg_mean) / bg_std

                    # now scale image
                    img_data = scale_img(img_data)
                    # plt.imshow(img_data, cmap='gray')
                    # plt.show()

                    # use subsection for thresholding
                    # subsection_size = 600
                    # min_x = max(0, int(obj_pix_x - subsection_size/2))
                    # max_x = min(img_data.shape[1], int(obj_pix_x + subsection_size/2))
                    # min_y = max(0, int(obj_pix_y - subsection_size/2))
                    # max_y = min(img_data.shape[0], int(obj_pix_y + subsection_size/2))
                    # img_data = img_data[min_y:max_y, min_x:max_x]
                    

                    # now find sources
                    threshold = detect_threshold(img_data, nsigma=3)
                    source_map = detect_sources(img_data, threshold, npixels=10)
                    img_sources = SourceCatalog(data=img_data, segment_img = source_map)

                    all_bands_sources.append(img_sources)


            except Exception as e:
                print(f"Unable to process image... {e}")
                print(f"Skipping {obj_idx = } and")

                
            # find which source contains the object we are interested in
            object_band_sources = []
            try:
                print("Finding Sources with Object...")
                for idx, (img, band, sources) in enumerate(zip(img_data_buffer,bands, all_bands_sources)):
                    # convert the ra dec coords to pixel coords
                    header = img[0][0].header
                    wcs = WCS(header)
                    obj_pix_x, obj_pix_y = wcs.world_to_pixel(c)

                    source_with_object = None
                    potential_sources_with_object = []
                    for source in sources:
                        ixmin, ixmax, iymin, iymax = source.bbox.ixmin, source.bbox.ixmax, source.bbox.iymin, source.bbox.iymax
                        if (obj_pix_x > ixmin) and (obj_pix_x < ixmax) and (obj_pix_y > iymin) and (obj_pix_y < iymax):
                            potential_sources_with_object.append(source)
                    
                    if len(potential_sources_with_object) == 0:
                        print(f"Unable to find source that contains object in band {band}")


                    elif len(potential_sources_with_object) > 1:
                        # find the source with the object closest to center
                        distances_from_center = np.array([])
                        for potential_source in potential_sources_with_object:
                            center_x, center_y = potential_source.centroid
                            distance_from_center = np.sqrt((center_x - obj_pix_x)**2 + (center_y - obj_pix_y)**2)
                            distances_from_center = np.append(distances_from_center, distance_from_center)
                        source_with_object = potential_sources_with_object[np.argmin(distances_from_center)]

                    elif len(potential_sources_with_object) == 1:
                        source_with_object = potential_sources_with_object[0]

                    else:
                        print("Something else... idk man...")

                    object_band_sources.append(source_with_object)
            except Exception as e:
                print(f"Unable to find source that contains objects... {e}")

            # best_object_source = None
            # try:
            #     # find the source bounding box that contains the object closest to the center
            #     print("Finding Source with Object Closest to Center...")
            #     distances_from_center = np.array([])
            #     for (object_band_source, img) in zip(object_band_sources, img_data_buffer):
            #         if object_band_source is None:
            #             distances_from_center = np.append(distances_from_center, np.inf)
            #             continue

            #         header = img[0][0].header
            #         wcs = WCS(header)
            #         obj_pix_x, obj_pix_y = wcs.world_to_pixel(c)

            #         center_x, center_y = object_band_source.centroid
            #         distance_from_center = np.sqrt((center_x - obj_pix_x)**2 + (center_y - obj_pix_y)**2)
            #         distances_from_center = np.append(distances_from_center, distance_from_center)
                
            #     best_band_idx = np.argmin(distances_from_center)
            #     best_object_source = object_band_sources[best_band_idx]
            # except Exception as e:
            #     print(f"Unable to find source that contains objects... {e}")


            # if every item in object_band_sources is None, then we can't make a chip for this object
            if all([object_band_source is None for object_band_source in object_band_sources]):
                print(f"Unable to find source that contains object in any band")
                bad_idxs.append(obj_idx)
                with open(bad_idx_file, 'a') as f:
                    f.write(f"{obj_idx}\n")
                continue
                
            try:
                print("Making chip for all sources.")
                # now to make chips for every image at this source location:
                for idx, (img, band, object_band_source) in enumerate(zip(img_data_buffer,bands, object_band_sources)):
                    if object_band_source is None:
                        continue
                    ixmin, ixmax, iymin, iymax = object_band_source.bbox_xmin, object_band_source.bbox_xmax, object_band_source.bbox_ymin, object_band_source.bbox_ymax
                    
                    img_data = img[0][0].data.copy()
                    chip_data = img_data[iymin:iymax, ixmin:ixmax]
                    chip_data = scale_img(chip_data)

                    # save image chip
                    im = Image.fromarray((chip_data*255).astype('uint8'))
                    im = im.convert('L')
                    im.save(data_dir / f"{obj_idx}-{band}-{obj_class}.jpeg")
            except Exception as e:
                print(f"Unable to make chips... {e}")
                print(object_band_sources)



                    





                    # # slice_y_0 and slice_x_0 cannot be negative
                    # slice_y_0 = slice_y_0 if slice_y_0 > 0 else 0
                    # slice_x_0 = slice_x_0 if slice_x_0 > 0 else 0

                    # # slice_y_1 and slice_x_1 cannot be greater than the size of the respective dimension
                    # slice_y_1 = slice_y_1 if slice_y_1 < img_data.shape[0] else img_data.shape[0]
                    # slice_x_1 = slice_x_1 if slice_x_1 < img_data.shape[1] else img_data.shape[1]


                    # # make initial large chip of the object we are interested in
                    # chip_data = img_data[slice_y_0:slice_y_1, slice_x_0:slice_x_1]
                    # center_chip_x, center_chip_y = chip_data.shape[1] / 2, chip_data.shape[0] / 2

                    
                    # # plotting
                    # _, chip_data, ellipse = fit_2d_gaussian(chip_data, object, band, None, reshape=True)

                    # interval = ZScaleInterval(max_iterations=10)
                    # vmin, vmax = interval.get_limits(chip_data)
                    # norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch(), clip=True)
                    # chip_data = norm(chip_data)    
                    
                    # im = Image.fromarray((chip_data*255).astype('uint8'))
                    # im = im.convert("L")
                    # # im.show()
                    # im.save(f"./chips/{obj_idx}-{band}.jpeg")
            
            
            # except Exception as e:
            #     print(f"Unable to process image... {e}")
            #     print(f"Skipping {obj_idx = } and")






            
                
    except ConnectionAbortedError as e:
        print(e)
        time.sleep(30)
        print("Sleeping...")
        print("Sleeping...")
        print("Sleeping...")
        print("Sleeping...")
        try:
            contents = data_dir.glob("**/*.jpeg")
            idxs = list(set([int(content.name.split('-')[0]) for content in contents]))
            last_idx = idxs[-4]
            get_imgs(df, last_idx, mod=mod, n_para=n_para)
        except IndexError as e:
            print(e)
            last_idx = 0

    except ConnectionError as e:
        print(e)
        print("Sleeping...")
        print("Sleeping...")
        print("Sleeping...")
        print("Sleeping...")
        time.sleep(30)
        try:
            contents = data_dir.glob("**/*.jpeg")
            idxs = list(set([int(content.name.split('-')[0]) for content in contents]))
            last_idx = idxs[-4]
            get_imgs(df, last_idx, mod=mod, n_para=n_para)
        except IndexError as e:
            print(e)
            last_idx = 0
    
    except TimeoutError as e:
        print(e)
        print("Sleeping...")
        print("Sleeping...")
        print("Sleeping...")
        print("Sleeping...")
        time.sleep(30)
        try:
            contents = data_dir.glob("**/*.jpeg")
            idxs = list(set([int(content.name.split('-')[0]) for content in contents]))
            last_idx = idxs[-4]
            get_imgs(df, last_idx, mod=mod, n_para=n_para)
        except IndexError as e:
            print(e)
            last_idx = 0
            
    except AQTimeoutError as e:
        print(e)
        print("Sleeping...")
        print("Sleeping...")
        print("Sleeping...")
        print("Sleeping...")
        time.sleep(30)
        try:
            contents = data_dir.glob("**/*.jpeg")
            idxs = list(set([int(content.name.split('-')[0]) for content in contents]))
            last_idx = idxs[-4]
            get_imgs(df, last_idx, mod=mod, n_para=n_para)
        except IndexError as e:
            print(e)
            last_idx = 0
    
    except URLError as e:
        print(e)
        print("Sleeping...")
        print("Sleeping...")
        print("Sleeping...")
        print("Sleeping...")
        time.sleep(30)
        try:
            contents = data_dir.glob("**/*.jpeg")
            idxs = list(set([int(content.name.split('-')[0]) for content in contents]))
            last_idx = idxs[-4]
            get_imgs(df, last_idx, mod=mod, n_para=n_para)
        except IndexError as e:
            print(e)
            last_idx = 0
    
    # except Exception as e:
    #     print(e)
    #     print("Sleeping...")
    #     print("Sleeping...")
    #     print("Sleeping...")
    #     print("Sleeping...")
    #     time.sleep(30)
    #     try:
    #         contents = data_dir.glob("**/*.jpeg")
    #         idxs = list(set([int(content.name.split('-')[0]) for content in contents]))
    #         last_idx = idxs[-4]
    #         get_imgs(df, last_idx, mod=mod, n_para=n_para)
    #     except IndexError as e:
    #         print(e)
    #         last_idx = 0


try:
    data_dir = Path("./chips2")
    contents = data_dir.glob("**/*.jpeg")
    idxs = list(set([int(content.name.split('-')[0]) for content in contents]))
    last_idx = idxs[-4]
except IndexError:
    last_idx = 0





if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mod", type=int, help="mod", required=True)
    parser.add_argument("-n", "--n_para", type=int, help="n_para", required=True)
    args = parser.parse_args()
    mod = args.mod
    n_para = args.n_para
    get_imgs(df, last_idx, mod=mod, n_para=n_para)

    # profile(get_imgs(df, last_idx, mod=mod, n_para=1))
    # get_imgs(df, last_idx)