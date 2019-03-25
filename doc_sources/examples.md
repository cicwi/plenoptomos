# Examples analysis

Here we analyse the code and results of the example files
Examples 02 and 03 use data from the [Stanford light-field archive](http://lightfields.stanford.edu/).

## example_01_logos.py

This is a preliminary eaxample that serves only to demonstrate the concepts discussed in:
> N. Viganò, et al., “Tomographic approach for the quantitative scene reconstruction from light field images,” Opt. Express, vol. 26, no. 18, p. 22574, Sep. 2018.

In particular it displays the scaling differents in refocusing alpha parameter, and the size of the refocused objects in the scene.

## example_02_flower_refocusing.py

This example shows a selection of all the possible refocusing options and tools available in plenoptomos.
It uses the [flowers & plants](http://lightfields.stanford.edu/flowers_plants.html) example number 30, and so it requires importing.

First we import the ESLF image of choice:
```
(lf_r ,lf_g, lf_b) = pleno.import_lf.from_lytro(dpath, jpath, source='eslf', mode='rgb')
```
By choosing the `mode='rgb'` we obtain three light-fields (one per RGB channel).

We then create a (v, u) PSF for each color channel, using the following two lines:
```
psf_ml_r = pleno.psf.PSF.create_theo_psf(lf_r.camera, coordinates='vu', airy_rings=2)
psf_ml_r = pleno.psf.PSFApply2D(psf_d=psf_ml_r, use_otf=False)
```
which first create the theoretical PSF for an incoherent light source (for the wavelengths indicated in `lf_r.camera`), including only the first two orders of the Airy function.
The second line creates an object able to manipulate the light-field data, and apply the computed PSF.

The following line computes the acquisition focal plane in the object space:
```
z0 = lf_r.camera.get_focused_distance()
```
We then define the alpha parameters:
```
alphas_con = np.linspace(0.5, 3.0, 46)
```

We then convert the true distances in parallel beam distances because we will produce that type of refocused images:
```
alphas_par = lf_r.camera.get_alphas(alphas_con, beam_geometry_in='cone', beam_geometry_out='parallel')
z0s = z0 * alphas_par
```

For convenience we create a function that handles the RGB channels:
```
def refocus_rgb(refocus_func, renorm=False):
    imgs_r = refocus_func(lf_r, psf_ml_r)
    imgs_g = refocus_func(lf_g, psf_ml_g)
    imgs_b = refocus_func(lf_b, psf_ml_b)
    if renorm:
        lf_ones = lf_r.clone()
        lf_ones.data = np.ones_like(lf_ones.data)
        imgs_ones = refocus_func(lf_ones, psf_ml_r)
        imgs_r /= imgs_ones
        imgs_g /= imgs_ones
        imgs_b /= imgs_ones
    return pleno.colors.merge_rgb_images(imgs_r, imgs_g, imgs_b)
```
and lambda functions that handle the refocusing for each method.
The Integration, Back-projection and SIRT functions are straight-forward:
```
refocus_int = lambda x, _ : pleno.refocus.compute_refocus_integration(x, z0s_sel, beam_geometry='parallel')
refocus_bpj = lambda x, _ : pleno.tomo.compute_refocus_iterative(x, z0s_sel, beam_geometry='parallel', algorithm='bpj')
refocus_sirt = lambda x, _ : pleno.tomo.compute_refocus_iterative(x, z0s_sel, beam_geometry='parallel', iterations=3, algorithm='sirt')
```
while the CP-LS-TV version needs a Solver object defined with custom parameters chosen by the user:
```
algo = pleno.solvers.CP_tv(data_term='l2', lambda_tv=1e-1, axes=(-2, -1))
refocus_cplstv_p = lambda x, p : pleno.tomo.compute_refocus_iterative(x, z0s_sel, beam_geometry='parallel', iterations=50, algorithm=algo, psf=p)
```
This syntax allows the users to create their own refocusing algorithms, and pass them to the function `pleno.tomo.compute_refocus_iterative`.

The expected refocusing for one distance will be:
![](Images/example_02_dist10.png "results_rose")

Which zoomed on the rose will be:
![](Images/example_02_zoom_rose_dist10.png "results_rose_zoom")


## example_03_flower_depth.py

This example displays the use of the depth-estimation routines available in plenoptomos.

In this case the light-field is loaded in grayscale mode:
```
lf = pleno.import_lf.from_lytro(dpath, jpath, source='eslf')
```

The depth cues described in Tao's paper are computed with the following function call:
```
dc = pleno.depth.compute_depth_cues(lf, z0s)
```

The computed depth cues are then assembled in a depth-map with the following optimization routine:
```
dm = pleno.depth.compute_depth_map(dc, lambda_tv=1.0, lambda_smooth=None)
```
The expected output is:
![](Images/example_03_results.png "results_rose_depth")

The function `compute_depth_cues` is highly tunable and allows the user to use advanced refocusing methods in the computation of the said depth cues.
For instance:
```
dc = pleno.depth.compute_depth_cues(lf, z0s, algorithm='sirt')
```
will use the sirt algorithm to compute the the focal stack.
Moreover, it allows more advanced filtering options than what was initially proposed in Tao's article.
Aside from the proposed rectangular filter (`window_shape='rect'`), it also accepts triangular (`'tri'`), circular (`'circ'`), gaussian (`'gauss'`) filters.
The size of the filters is adjusted using the option `window_size` (default: `window_size=(9, 9)`).
