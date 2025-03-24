
Put anyone's face on anything with Byte Dance's [InfiniteYou](https://github.com/bytedance/InfiniteYou).

## Hardware

Uses 40GB of VRAM.  Takes about a minute on an RTX A6000, 48GB.

Note: I only had temporary access to the hardware to run this.  Might not be able to help much.


## Installation

To get FLUX, you need to run `huggingface-cli login`.
And put in your huggingface access token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

At the moment it needs `diffusers==0.31` `numpy<2`.  If something goes wrong make sure these versions are installed `pip show numpy diffusers`


## runpod

Link the ~/.cache folder to permanent storage or else it'll get wiped every time and it'll redownload the models every time.

