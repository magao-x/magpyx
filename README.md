# magpyx
Library of python tools for MagAO-X

Examples of interactive python usage are provided [here](https://github.com/magao-x/magpyx/blob/master/notebooks/dm_interaction.ipynb). Shell scripts are described below.

## Shell Scripts

### DM Interaction

* Generate a set of Zernike modes projected onto an elliptical beam footprint on a DM and save them to a FITS cube:

```
dm_project_zernikes [nterms] [nact] [angle in degrees] [ horizontal fill fraction] [outfile]
```

* Generate woofer-to-tweeter offload matrices from `cacao` zrespM.fits files:

```
dm_offload_matrix [zrespM_woofer] [zrespM_tweeter] [outname] [--n_threshold threshold] [--display] [--crosscheck]
```

* Run the eye doctor on a single DM mode:

```
dm_eye_doctor_mode [portINDI] [device] [shared memory image] [core radius] [mode #] [range to sample in microns]
```

* Run a comprehensive eye exam (global optimization):

```
dm_eye_doctor_comprehensive [portINDI] [device] [shared memory image] [core radius] [modes ex: 1,2,3...10,11,13] [range to sample in microns]
```

* Write optimized flats to file

```
dm_eye_doctor_to_fits [dm name] [--filename FILENAME] [--symlink]
```

### Shared Memory Interaction

* Send test pokes to DMs:

```
dm_send_poke [shmim name] [x index] [y index] [poke in microns]
```

* Send a FITS file to shared memory

```
send_fits_to_shmim [shmim name] [fitsfile]
```

* Write a shared memory image to a FITS file

```
send_shmim_to_fits [shmim name] [fitsfile]
```

* Write zeros to a shared memory image

```
send_zeros_to_shmim [shmim name]
```

### INDI Interaction

* Set an INDI element value with status reporting

```
pyindi_send_triplet [device].[property].[element]=[value] [--host HOST] [--port PORT]
```

* Send MagAO-X to a predefined state with status reporting

```
pyindi_send_preset [preset name] [--host HOST] [--port PORT]
```

* List the avaiable instrument presets:

```
pyindi_send_preset --ls
```
