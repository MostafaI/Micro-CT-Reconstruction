import pandas as pd 
import numpy as np 
import itk
from tqdm import tqdm

def get_geometrical_data(geopar_path):
    # ---- Parse geopar 
    geo = {}
    with open(geopar_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if "=" in line:
                k, v = line.split("=", 1)
                k = k.strip().replace("#", '')
                v = v.strip()
                # try numeric
                try:
                    if "." in v or "e" in v.lower() or "E" in v:
                        geo[k] = float(v)
                    else:
                        geo[k] = int(v)
                except Exception:
                    geo[k] = v
    return geo

def get_angles(proj_log_csv, first_angle=0):
    # ---- Read angular positions from proj_log.csv (3rd column)
    df = pd.read_csv(proj_log_csv, header=None)
    if df.shape[1] < 3: raise ValueError("proj_log.csv should have at least 3 columns; the 3rd column holds angles.")
    angles_raw = df.iloc[:, 2].to_numpy().astype(float)
    return first_angle + angles_raw

def get_projections(corr_dir):
    # assume corr_dir contains the corrected (dynamic range) tiff projections (16-bit images)
    # convert the intensity to attenuation (projections) 
    # ---- List projection images in ct-data/corr (tif/tiff)
    tiff_paths = sorted([*corr_dir.glob("*.tif"), *corr_dir.glob("*.tiff")])
    if len(tiff_paths) == 0: raise FileNotFoundError(f"No TIFF projections found in {corr_dir}")
    # ---- Read projections into a 3D stack (x,y,projection)
    # ITK reads pixel spacing from TIFF tags (74.8 µm) automatically.
    ImageType3D = itk.Image[itk.F, 3]
    series_reader = itk.ImageSeriesReader[ImageType3D].New()
    series_reader.SetFileNames([str(p) for p in tiff_paths])
    series_reader.Update()
    projs = series_reader.GetOutput()
    # --- Convert 16-bit corrected TIFFs to line integrals: -log(I / I0)
    arr = itk.array_from_image(projs).astype(np.float32)   # shape: (Nproj, Y, X) or (Z,Y,X) depending on ITK build
    arr_norm = arr / 65535.0
    np.clip(arr_norm, 1e-6, 1.0, out=arr_norm)
    arr_line = -np.log(arr_norm)
    # Back to ITK image, preserving geometry of the stack
    line_int = itk.image_from_array(arr_line)
    line_int.CopyInformation(projs)
    return line_int


def recon(
    geo: dict, # Geometry dict with sid, sdd, ... etc.. 
    line_int: np.ndarray, # (N-projections, detector_rows, detector_cols)
    gantry_angles: np.ndarray, # Ganty angles 
    vol_size    = [400, 400, 457],    # desired voxels (x, y, z) in reconned image
    vol_spacing = [0.16, 0.16, 0.16], # desired mm per voxel
    hann_freq   = 0.5, # Hann apodization cutoff frequency
    fwhm_mm = 0.320,    # mm gaussian filter FWHM (like milabs)
    tqdm_bar_title = "Recon", 
    save_name = "", 
):
    def tqdm_callback():
        progress = fdk.GetProgress() * 100.0
        pbar.n = int(progress)
        pbar.refresh()
        
    # ---- Build RTK ThreeDCircularProjectionGeometry
    GeometryType = itk.ThreeDCircularProjectionGeometry
    geometry = GeometryType.New()
    # If your proj_iso_x/y vary per projection, replace proj_iso_x/y below with arrays (same length as angles).
    # The vendor said proj_iso_* are defined on the in-plane rotated detector; pass as-is.
    for th in gantry_angles:
        geometry.AddProjection(
            geo["sid"],
            geo["sdd"],
            float(th),                 # gantryAngle in degrees
            geo["proj_iso_x"],         # projOffsetX (mm)
            geo["proj_iso_y"],         # projOffsetY (mm)
            0,                         # outOfPlaneAngle (deg) - set if you have it
            geo["in_angle"],           # inPlaneAngle (deg)
            geo["source_x"],           # sourceOffsetX (mm)
            geo["source_y"]            # sourceOffsetY (mm)
        )
    
    # Desired world origin (of voxel index [0,0,0])
    # Center the volume around isocenter by setting origin symmetrically:
    vol_origin = [-(vol_size[0]*vol_spacing[0])/2.0,
                  -(vol_size[1]*vol_spacing[1])/2.0,
                  -(vol_size[2]*vol_spacing[2])/2.0]
    
    ImageType3D = itk.Image[itk.F, 3]
    ConstantSourceType = itk.RTK.ConstantImageSource[ImageType3D]
    constant_source = ConstantSourceType.New()
    constant_source.SetSize(vol_size)
    constant_source.SetSpacing(vol_spacing)
    constant_source.SetOrigin(vol_origin)
    constant_source.SetConstant(0.0)           # background
    
    # ---- Set up FDK reconstruction
    FDKType = itk.RTK.FDKConeBeamReconstructionFilter[ImageType3D]
    fdk = FDKType.New()
    fdk.SetInput(0, constant_source.GetOutput())
    fdk.SetInput(1, line_int)
    fdk.SetGeometry(geometry)
    
    # Optional: tweak ramp filter (Hann window)
    ramp = fdk.GetRampFilter()
    ramp.SetHannCutFrequency(hann_freq)           # Example if you want a Hann apodization
    # By default, RTK uses pure Ramp when cut frequency = 0.0 (no apodization).
    # You can also try:
    # ramp.SetTruncationCorrection(0.0)
    
    cmd = itk.PyCommand.New()
    cmd.SetCommandCallable(tqdm_callback)
    fdk.AddObserver(itk.ProgressEvent(), cmd)

    # ---- Run the pipeline
    pbar = tqdm(total=100, desc=tqdm_bar_title, unit="%");
    fdk.Update()
    reconned = fdk.GetOutput()
    pbar.close()

    # ----- Smooth the output 
    sigma_vox = fwhm_mm / 2.355
    # --- Make a fully-buffered, standalone copy (break the pipeline)
    DupType = itk.ImageDuplicator[ImageType3D]
    dup = DupType.New()
    dup.SetInputImage(reconned)
    dup.Update()
    recon_buf = dup.GetOutput()
    Smooth = itk.SmoothingRecursiveGaussianImageFilter[ImageType3D, ImageType3D].New()
    Smooth.SetInput(recon_buf)
    Smooth.SetSigma(sigma_vox)           # sigma in *voxel* units
    Smooth.Update()
    recon_smoothed = Smooth.GetOutput()
    if save_name != '':
        itk.imwrite(recon_smoothed, str(save_name))
    return recon_smoothed