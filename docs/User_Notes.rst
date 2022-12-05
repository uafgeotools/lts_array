User Notes
================
Thanks to helpful feedback from users, here we list a few notes for infrasound array processing with least trimmed squares (LTS).


A note on :math:`{\alpha}`
------------------------------------
To completely remove one element during LTS processing, set :math:`{\alpha}` = 1 - 2/n.

3 Element Arrays
--------------------------
For 3 element arrays, least trimmed squares cannot be used. This is because we are trying to fit a 2D plane with 3 choose 2 = 3 elements. Ordinary least squares (:math:`{\alpha}` = 1.0) should be used in this case.

4 Element Arrays
---------------------------
For 4 element arrays, least trimmed squares can be used, but its effectiveness is limited. This is because there are 4 choose 2 = 6 data points used in the regression, but each element is involved in 3 cross correlations (no autocorrelations). Maximum trimming , :math:`{\alpha}` = 0.5, here actually chooses 4 elements, so the data will still be contaminated. For this reason, we have added an option to remove an element prior to processing. If a four element array is suspected to have an issue with an element, we recommend the user remove the element and process the array as a 3 element array.

5+ Element Arrays
------------------------------
LTS is the most effective for processing arrays with at least five elements.

Uncertainty Quantification
---------------------------------------
The code now automatically calculates uncertainty estimates using the slowness ellipse method of Szuberla and Olson (2004). The default is currently set at 90% confidence, but this value can be changed in the `LsBeam` class in `lts_classes.py`. Note, the values here are 1/2 the extremal values described in Szuberla and Olson (2004) and are meant to approximate confidence intervals, i.e. value +/- uncertainty estimate, not the area of the coverage ellipse. The :math:`{\sigma_\tau}` value is now calculated automatically by both the ordinary least squares and the least trimmed squares routines.