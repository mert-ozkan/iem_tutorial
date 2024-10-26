# iem_tutorial

Tutorial for building inverted encoding models (IEM) with EEG data.

The data and data description can be found on this [link](https://osf.io/qx94k/?view_only=e256bebd6e7c4f888a696fb35919913d).

The current repository illustrates several **data preprocessing methods to improve IEMs** including principal and independent component analyses (**PCA**, **ICA**) as well as **ridge regularization by shrinkage** which is an effective method to reduce multi-collinearities that are present in EEG.

It also includes a novel methodology making use of **Procrustes transformation**. This method is of specific use for the current dataset which used a frequency–tagging method called steady-state evoked potentials (**SSVEPs**) where a target and a distractor stimuli were presented simultaneously, each flickering at a distinct frequency. The Procrustes transformation allows combining the data from two frequencies when building the IEM model, which were conventionally analyzed separately [REF]. Therefore, it increases the statistical power of the IEM procedure.
