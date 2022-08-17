# STC Live Preprocessing

### Install

1. Install node v. 14.0.0 and npm
2. Clone this repo (https://github.com/felixboettger/stc-live-preprocessing/)
3. Change into the cloned directory and execute `npm i` to install dependencies

### Usage

1. Put a folder of images into the cloned directory and change the folder's name to "dataset"
2. Execute `node app.js` with the cloned directory as the working directory

### Explanation of output

NI_dataset contains the greyscaled, cropped and masked faces 
NL_dataset contains the landmarks
NH_dataset contains the hog values 

The folder structure of these folders resembles the original (dataset) folder structure
