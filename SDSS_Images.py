
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
from astropy.stats import sigma_clip
from astropy.visualization import MinMaxInterval, SqrtStretch, ImageNormalize, ZScaleInterval, LogStretch, LinearStretch, SinhStretch, AsinhStretch, make_lupton_rgb

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from scipy.optimize import curve_fit
from PIL import Image
from sklearn.preprocessing import MinMaxScaler

import time

from pathlib import Path




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

def get_imgs(df, last_idx, mod=0, n_para=1, max_idx = 53457):
    try:
        # figure out which ones we want to make objects for
        idxs = np.arange(0,len(df),1)
        # idxs_to_generate = idxs
        idxs_to_generate = idxs[idxs%n_para == mod]
        
        # data_dir = Path("G:/My Drive/MIDS/207/SDSS-Classification/chips")
        data_dir = Path("./chips")
        print(data_dir)
        print(f"{data_dir.exists() = }")
        contents = data_dir.glob("**/*.jpeg")
        contents = [content for content in contents]
            
        
        generated_idxs = list(set([int(content.name.split('-')[0]) for content in contents]))
        idxs_to_generate = idxs_to_generate[~np.isin(idxs_to_generate, generated_idxs)]
        idxs_to_generate = idxs_to_generate[idxs_to_generate < max_idx]


        # check that images for all bands have been generated
        for gen_idx in generated_idxs:
            for band in bands:
                if not (data_dir / f"{gen_idx}-{band}.jpeg").exists():
                    print(f"MISSING {gen_idx}-{band}.jpeg")
                    idxs_to_generate = np.append(idxs_to_generate, gen_idx)
                    break
            
            

        for obj_idx, row in df.iterrows():
            print(f"Testing {obj_idx = }")
            if obj_idx not in idxs_to_generate:
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
            for idx, (img, band) in enumerate(zip(img_data_buffer,bands)):
                
                out_dir = Path("./chips")
                out_dir = out_dir / band
                header = img[0][0].header
                wcs = WCS(header)
                obj_pix_x, obj_pix_y = wcs.world_to_pixel(c)

                chip_size = 50
                slice_y_0, slice_y_1 = int(obj_pix_y-chip_size), int(obj_pix_y+chip_size)
                slice_x_0, slice_x_1 = int(obj_pix_x-chip_size), int(obj_pix_x+chip_size)
                # slice_y_0, slice_y_1 = 10, -10
                # slice_x_0, slice_x_1 = 10, -10
                
                
                img_data = img[0][0].data.copy()

                # slice_y_0 and slice_x_0 cannot be negative
                slice_y_0 = slice_y_0 if slice_y_0 > 0 else 0
                slice_x_0 = slice_x_0 if slice_x_0 > 0 else 0

                # slice_y_1 and slice_x_1 cannot be greater than the size of the respective dimension
                slice_y_1 = slice_y_1 if slice_y_1 < img_data.shape[0] else img_data.shape[0]
                slice_x_1 = slice_x_1 if slice_x_1 < img_data.shape[1] else img_data.shape[1]


                # make initial large chip of the object we are interested in
                chip_data = img_data[slice_y_0:slice_y_1, slice_x_0:slice_x_1]
                center_chip_x, center_chip_y = chip_data.shape[1] / 2, chip_data.shape[0] / 2

                
                # plotting
                _, chip_data, ellipse = fit_2d_gaussian(chip_data, object, band, None, reshape=True)

                interval = ZScaleInterval(max_iterations=10)
                vmin, vmax = interval.get_limits(chip_data)
                norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SinhStretch(), clip=True)
                chip_data = norm(chip_data)    
                
                im = Image.fromarray((chip_data*255).astype('uint8'))
                im = im.convert("L")
                # im.show()
                im.save(f"./chips/{obj_idx}-{band}.jpeg")
                
    except ConnectionAbortedError as e:
        print(e)
        time.sleep(30)
        print("Sleeping...")
        print("Sleeping...")
        print("Sleeping...")
        print("Sleeping...")
        try:
            data_dir = Path("./chips")
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
            data_dir = Path("./chips")
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
            data_dir = Path("./chips")
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
            data_dir = Path("./chips")
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
            data_dir = Path("./chips")
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
    #         data_dir = Path("./chips")
    #         contents = data_dir.glob("**/*.jpeg")
    #         idxs = list(set([int(content.name.split('-')[0]) for content in contents]))
    #         last_idx = idxs[-4]
    #         get_imgs(df, last_idx, mod=mod, n_para=n_para)
    #     except IndexError as e:
    #         print(e)
    #         last_idx = 0


try:
    data_dir = Path("./chips")
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