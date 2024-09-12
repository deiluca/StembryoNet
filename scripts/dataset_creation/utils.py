import numpy as np
import pandas as pd
from os.path import join as opj
import os
import skimage.io as skio
import tifffile as tiff
import sys
import cv2


def generate_sbatch_file(SBATCH_DIR, sbatchfilename, slurm_backbone, cmd):
    """
    Creates a SLURM batch script file in the specified directory.

    Parameters:
    SBATCH_DIR (str): Directory where the SLURM script file will be saved.
    sbatchfilename (str): Name of the SLURM script file.
    slurm_backbone (str): Base content of the SLURM script.
    cmd (str): Command(s) to be appended to the SLURM script.

    Returns:
    None: This function creates a file and prints a confirmation message.

    Side Effects:
    - Creates the specified directory if it does not exist.
    - Writes the SLURM script to the specified file.
    """
    os.makedirs(SBATCH_DIR, exist_ok=True)
    with open(opj(SBATCH_DIR, sbatchfilename), "w") as f:
        f.write(slurm_backbone + cmd)
    print(f"SLURM script written: {opj(SBATCH_DIR, sbatchfilename)}")

def preprocess_img(img_, t, channels):
    """
    Preprocesses an image at a specific time point.

    Parameters:
    img_ (numpy.ndarray): Input image array. Shapes can be [time, height, width, channels] or [time, height, width].
    t (int): Time point index.
    channels (int or list of int): Channel(s) to select (ignored if not applicable).

    Returns:
    numpy.ndarray: Preprocessed image with shape [height, width, channels] or [height, width].

    Raises:
    SystemExit: If img_ does not have the expected shape.
    """
    if len(img_.shape) == 4:
        img = np.squeeze(img_[t, :, :, channels])
    elif len(img_.shape) == 3:
        img = np.squeeze(img_[t, :, :])
    else:
        sys.exit('img.shape != [3, 4]')
    return img

def preprocess_video(img_, channels):
    """
    Preprocesses an image at a specific time point.

    Parameters:
    img_ (numpy.ndarray): Input image array. Shapes can be [time, height, width, channels] or [time, height, width].
    t (int): Time point index.
    channels (int or list of int): Channel(s) to select (ignored if not applicable).

    Returns:
    numpy.ndarray: Preprocessed image with shape [height, width, channels] or [height, width].

    Raises:
    SystemExit: If img_ does not have the expected shape.
    """
    if len(img_.shape)==4:
        img = img_[:, :, :, channels]
    elif len(img_.shape)==3:
        img = img_
    return img


def save_img(img, channels, outfile):
    """
    Saves an image to a file based on the number of channels.

    Parameters:
    img (numpy.ndarray): The image data to be saved. Should be in the format compatible with the specified channels.
    channels (list of int): List of channel indices. If channels contains only [0], the image is saved as a TIFF file.
    outfile (str): Path to the output file where the image will be saved.

    Returns:
    None: This function writes the image data to the specified file.
    """
    if channels == [0]:
        tiff.imwrite(outfile, img, imagej=True)
    else:
        skio.imsave(outfile, img, photometric='rgb')

def save_imgs_selected_general5cvs_(df, outdir, t_col='Uniformly_sampled_timepoint', t_minus=0, all_channels=True, t_fix=None, img_col='img', label_col='Score_new_str', id_col='embryo_id_y', channels=[0], tmin=-1, tmax=-1, save_type='img'):
    """
    Saves images or videos based on a dataframe and specified parameters.

    Parameters:
    df (pandas.DataFrame): DataFrame containing image metadata.
    outdir (str): Directory where images or videos will be saved.
    t_col (str): Column name indicating time points ('Uniformly_sampled_timepoint' or 'plain').
    t_minus (int): Offset to adjust the time point.
    all_channels (bool): Whether to process all channels.
    t_fix (int): Fixed time point for 'plain' time point.
    img_col (str): Column name containing image paths.
    label_col (str): Column name used for subdirectory names.
    id_col (str): Column name for embryo IDs.
    channels (list of int): List of channel indices to process.
    tmin (int): Minimum time point for video or image range.
    tmax (int): Maximum time point for video or image range.
    save_type (str): Type of output to save ('img' for images, 'video' for videos).

    Returns:
    None: This function writes images or videos to the specified directories.

    Side Effects:
    - Creates directories as needed.
    - Saves images as TIFF files or videos as MP4 files.
    """
    assert save_type in ['img', 'video']
    for _, row in df.iterrows():
        outdir2 = opj(outdir, row[label_col])
        os.makedirs(outdir2, exist_ok=True)
        embryo_id = row[id_col]
        img_ = skio.imread(row[img_col].replace('/mnt/lsdf_iai-aida/Daten_Deininger/', '/lsdf/kit/iai/projects/iai-aida/Daten_Deininger/'))
        if save_type=='img':
            if t_col == 'Uniformly_sampled_timepoint':
                t = int(row[t_col]) - 1 - t_minus  # Paolo used 1-indexing
                t = t - 1 if t == 153 else t
                outfile = opj(outdir2, embryo_id + '_' + str(t) + '.tif')
                # Handle only one time point
                img = preprocess_img(img_, t, channels)
                save_img(img, channels, outfile)
            elif t_col == 'plain':
                if tmin == -1 and tmax==-1:
                    tmin = t_fix
                    tmax = t_fix+1
                for t_fix in range(tmin, tmax):
                    # Handle a range of time points
                    t = t_fix
                    outfile = opj(outdir2, embryo_id + '_' + str(t) + '.tif')
                    img = preprocess_img(img_, t, channels)
                    save_img(img, channels, outfile)
        elif save_type=='video':
            outfile = opj(outdir2, embryo_id+'.mp4')
            img = preprocess_video(img_, channels)
            width = 288
            height = 288            
            fps = 22
            video = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(*'mp4v'), float(fps), (width, height), True)
            assert tmin!=-1 and tmax!=-1
            for t in range(tmin, tmax):
                x = img[t, ...]
                video.write(x)
            video.release()
        else:
            sys.exit('Can only save images and videos')

def save_imgs_selected_general_5cvs(cv_nr, train, val, test, i, outdir, t_minus=0, all_channels=False, t_col='Uniformly_sampled_timepoint_new', t=None, img_col='img', label_col = 'Score_new_str', id_col='embryo_id_y', channels = [0], savetrain=True, save_type='img', tmin=-1, tmax=-1):
    """
    Organizes and saves training, validation, and test images or videos based on cross-validation splits.

    Parameters:
    cv_nr (int): Cross-validation number.
    train (pandas.DataFrame): DataFrame containing training image metadata.
    val (pandas.DataFrame): DataFrame containing validation image metadata.
    test (pandas.DataFrame): DataFrame containing test image metadata.
    i (int): Split index.
    outdir (str): Base directory for saving images or videos.
    t_minus (int): Offset to adjust the time point.
    all_channels (bool): Whether to process all channels.
    t_col (str): Column name indicating time points.
    t (int): Fixed time point for 'Uniformly_sampled_timepoint_new' or 'plain' time column.
    img_col (str): Column name containing image paths.
    label_col (str): Column name used for subdirectory names.
    id_col (str): Column name for embryo IDs.
    channels (list of int): List of channel indices to process.
    savetrain (bool): Whether to save training images.
    save_type (str): Type of output to save ('img' for images, 'video' for videos).
    tmin (int): Minimum time point for video or image range.
    tmax (int): Maximum time point for video or image range.

    Returns:
    None: This function organizes and saves images or videos for different data splits.

    Side Effects:
    - Creates directories for training, validation, and test sets.
    - Calls `save_imgs_selected_general5cvs_` to save images or videos.
    """
    outdir_imgs = outdir
    if savetrain:
        outdir_imgs_train = os.path.join(outdir_imgs, 'cv'+str(cv_nr) +'_split'+str(i)+'_training')
    outdir_imgs_val = os.path.join(outdir_imgs, 'cv'+str(cv_nr) +'_split'+str(i)+'_validation')
    outdir_imgs_test = os.path.join(outdir_imgs, 'cv'+str(cv_nr) +'_split'+str(i)+'_test')

    os.makedirs(outdir, exist_ok=True)
    if savetrain:
        os.makedirs(outdir_imgs_train, exist_ok=True)
    os.makedirs(outdir_imgs_val, exist_ok=True)
    os.makedirs(outdir_imgs_test, exist_ok=True)

    if savetrain:
        save_imgs_selected_general5cvs_(train, outdir_imgs_train, t_minus=t_minus, t_col=t_col, t_fix=t, img_col=img_col, label_col = label_col, id_col=id_col, channels = channels, save_type=save_type, tmin=tmin, tmax=tmax)
    save_imgs_selected_general5cvs_(val, outdir_imgs_val, t_minus=t_minus, t_col=t_col, t_fix=t, img_col=img_col, label_col = label_col, id_col=id_col, channels = channels, save_type=save_type, tmin=tmin, tmax=tmax)
    save_imgs_selected_general5cvs_(test, outdir_imgs_test, t_minus=t_minus, t_col=t_col, t_fix=t, img_col=img_col, label_col = label_col, id_col=id_col, channels = channels, save_type=save_type, tmin=tmin, tmax=tmax)