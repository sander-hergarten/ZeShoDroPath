# Environment Invariant Auto-Framing for Autonomous Drones
Automating visual tasks with drones—whether for inspection, delivery, or media—requires GPS coordinates and Orientations. This project aims to generate these coordinates, with no human intervention, from just a single recording of the site (i.e. a flyover) and an image of the desired feature (packaging labels, QR-Codes, ...). 
<br><\br>
## How it works
The small amount of inputs provided by the user requires a high use of synthetic data for visual encoder training. Traditionaly introduces the challenge of producing images that are acurate to the deployment environment. To overcome this we treat the task of encoding as a denoising problem and leverage diffusion models. This way we just need to reproduce common noise variants in our data which are significantly more general.

### Data Generation
Images are generated containing high levels of visual noise in different environment with randomised orientations. The images need to be seperable in noise and clean pixel data and as such we also generate a mask of the image. All images are generated using the Bevy Game engine (crates/data_generation). As such we generate our noisy image which is the base normally rendered image, and the clean image: $image * mask$
![image](https://storage.googleapis.com/readme-drone-pathing-images/image.png)
![mask](https://storage.googleapis.com/readme-drone-pathing-images/mask.png)

### Data Processing
Since Diffusion trains a stochastic process the images need to be interpolated into multiple timesteps. This is done by applying a Noisy Distance Field Algorithm to the original mask and multiplying it with the image (crates/data_processing). A noise level of 1 is an all white mask while noise level 0 is the original mask. The code uses WGPU-Compute shaders to apply these transformations efficiently
![obfuscation](https://storage.googleapis.com/readme-drone-pathing-images/obfuscation.png)

### Data storage
The applied masks are save to a Parquet file 

### Diffusion Training
The Diffusion net is trained on the Data generated synthetically. It outputs the next mask which can then be applied to the image and placed back as an input for inference.

### Scene Generation
With the user provided recording we calculate a 3d environment using Gausian Splats.

### RL Algorithm.
Now we train a RL agent traversing the environment, the agent recieves the diffusion nets inputs as an observation and finds positions maximizing a feature in frame. 
