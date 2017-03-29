# STparametric
parametric modeling of spatiotemporal data

This repo is largely a sandbox where I've experimented with different approaches for modeling Cuebiq data with parametric shapes.

The scripts that are necessary for the analysis are the following:

functions.py:  Contains all functions applied (curve fitting to stories, data prep, bezier curve building, etc)
fittingStream.py: takes some data, fit it to a curve and get the bezier curves.
CompareBezOri.py:  Shows a comparison between the bezier curves stored in a PG database and the original points

In addition, a file with one day of data for one deviceID is provided.
