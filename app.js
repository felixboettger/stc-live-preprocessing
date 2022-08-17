const fs = require('fs');
const path = require('path');
// const sharp = require('sharp');
const faceapi = require('@vladmandic/face-api');
const {Image} = require('image-js');
const hog = require('hog-features');
// const output = require('sharp/lib/output');
const { create } = require('domain');
const canvas = require('canvas');
const { image, StringToHashBucketFast } = require('@tensorflow/tfjs-core');
const { count } = require('console');
require('console-png').attachTo(console);

const imageDims = 256; // change this to change resolution of preprocessed image
const shiftDown = 0; 

function *walkSync(dir) {
    const files = fs.readdirSync(dir, { withFileTypes: true });
    for (const file of files) {
      if (file.isDirectory()) {
        yield* walkSync(path.join(dir, file.name));
      } else {
        yield path.join(dir, file.name);
      }
    }
}

function formatTime(milliseconds){
  let totalSeconds = milliseconds / 1000;
  let hours = Math.floor(totalSeconds / 3600);
  let minutes = Math.floor((totalSeconds % 3600) / 60);
  let seconds = Math.floor((totalSeconds % 60))
  return String(hours).padStart(2, '0') + ":" + String(minutes).padStart(2, '0') + ":" + String(seconds).padStart(2, '0');
}

function getRemainingTimeString(startTime, currentNumber, totalNumber){
  currentTime = new Date().getTime();
  let timePassed = currentTime - startTime;
  let timePerElement = timePassed / currentNumber;
  let remainingElements = totalNumber - currentNumber;
  let remainingTime = remainingElements * timePerElement;
  return `elapsed: ${formatTime(timePassed)} - remaining: ${formatTime(remainingTime)}`;
}

function landmarkListToString(landmarkList){
    // console.log(landmarkList)
    let str = "[\n";
    landmarkList.forEach(landmark => {
        str += "[" + landmark[0] + ", " + landmark[1] + "]," + "\n";
    });
    str = str.substring(0, str.length - 1) + "]";
    return str;
}  

function resizeLandmarks(landmarks,  use, useSide) {
    const landmarkList = [];
    for (let i = 0; i < landmarks.length; i++) {
      landmarkList.push(getNewCoords(landmarks[i]._x, landmarks[i]._y, use._x, use._y, use._width, use._height, useSide));
    }
    return landmarkList;
  }

  function rotateLandmarks(landmarks, angle) {
    const rotatedLandmarks = []
    cosAlpha = Math.cos(angle);
    sinAlpha = Math.sin(angle);
  
    for (let i = 0; i < landmarks.length; i++) {
      currentX = landmarks[i][0] - 56;
      currentY = landmarks[i][1] - 56;
  
      newX = currentX * cosAlpha - currentY * sinAlpha;
      newY = currentX * sinAlpha + currentY * cosAlpha;
  
      rotatedLandmarks.push([newX + 56, newY + 56]);
    }
    return rotatedLandmarks;
  }

function getNewCoords(x, y, boundingBoxUpperLeftX, boundingBoxUpperLeftY, width, height, useSide){
    x = x - boundingBoxUpperLeftX;
    y = y - boundingBoxUpperLeftY;
    const smallSide = Math.min(width, height);
    const bigSide = Math.max(width, height);
    const scaleSide = (useSide === "long") ? bigSide : smallSide;
    const ratio = (imageDims/scaleSide);
    const newX = x * ratio;
    const newY = y * ratio;
    return [newX.toFixed(3), newY.toFixed(3)];
}

function createDirs(filePath){
    let directoryPath = path.dirname(filePath);
    let newImagePrefix = "NI_";
    let newLandmarkPrefix = "NL_";
    let newHogsPrefix = "NH_";
    let subDirs = [newImagePrefix, newLandmarkPrefix, newHogsPrefix];
    subDirs.forEach(subDir => {
        newDir = subDir + directoryPath;
        if (!fs.existsSync(newDir)){
            fs.mkdirSync(newDir, { recursive: true });
        }
    });
}

function cropRotateFace(x, y, width, height, angle, useSide, canvasInput, canvasCropped) {  // x,y = topleft x,y
    const ctx2 = canvasCropped.getContext("2d");
    const tempCanvas1 = canvas.createCanvas()
    const tctx1 = tempCanvas1.getContext("2d");
    tempCanvas1.height = tempCanvas1.width = imageDims;
    tctx1.fillRect(0, 0, tempCanvas1.width, tempCanvas1.height);
    tctx1.translate(tempCanvas1.width / 2, tempCanvas1.height / 2);
    tctx1.strokeStyle = "orange";
    tctx1.rotate(angle);
    tctx1.translate(-tempCanvas1.width / 2, -tempCanvas1.height / 2);
    const longSideScale = Math.min(tempCanvas1.width / width, tempCanvas1.height / height);
    const shortSideScale = Math.max(tempCanvas1.width / width, tempCanvas1.height / height);
    let scale = (useSide === "long") ? longSideScale : shortSideScale;
    tctx1.drawImage(canvasInput, x, y, width, height, 0,0, width*scale, height*scale);
  
    ctx2.clearRect(0, 0, imageDims, imageDims);
    ctx2.drawImage(tempCanvas1, 0, shiftDown);
    let imgData = ctx2.getImageData(0, 0, ctx2.canvas.width, ctx2.canvas.height);
    let pixels = imgData.data;
    for (var i = 0; i < pixels.length; i += 4) {

      let lightness = parseInt((pixels[i] + pixels[i + 1] + pixels[i + 2]) / 3);

      pixels[i] = lightness;
      pixels[i + 1] = lightness;
      pixels[i + 2] = lightness;
    }
    ctx2.putImageData(imgData, 0, 0);
  }

  function maskFaceNew(faceLandmarks, canvasCropped) {
    const ctx2 = canvasCropped.getContext("2d");

    const marginX = 0;
    // c = sqrt((x_a - x_b)^2 + (y_a - y_b)^2)
    const eyeEyeBrowDistance = Math.sqrt(Math.pow((faceLandmarks[37][0] - faceLandmarks[19][0]), 2) + Math.pow((faceLandmarks[37][1] - faceLandmarks[19][1]), 2));
    // In pyfeat paper, 1.5 * eyeEyeBrowDistance was used, change rotation to not have white parts in special cases
    const marginY = eyeEyeBrowDistance;
    ctx2.beginPath();
    const fistCoordinateX = faceLandmarks[0][0] += marginX;
    const fistCoordinateY = faceLandmarks[0][1];
    ctx2.moveTo(fistCoordinateX, fistCoordinateY);
    for (let i = 1; i < 17; i++) {
      currentCoordinate = faceLandmarks[i];
      currentX = currentCoordinate[0];
      if (i < 8){
        currentX += marginX;
      } else if (i > 19){
        currentX -= marginX;
      }
      
      currentY = currentCoordinate[1];
      ctx2.lineTo(currentX, currentY);
    }
    // Brows Right
    for (let i = 26; i > 24; i--){
      currentCoordinate = faceLandmarks[i];
      currentX = currentCoordinate[0];
      currentY = currentCoordinate[1] - marginY;
      ctx2.lineTo(currentX, currentY);
    }
    // Brows Left
    for (let i = 18; i > 16; i--){
      currentCoordinate = faceLandmarks[i];
      currentX = currentCoordinate[0];
      currentY = currentCoordinate[1] - marginY;
      ctx2.lineTo(currentX, currentY);
    }
  
    ctx2.lineTo(fistCoordinateX, fistCoordinateY);
    ctx2.lineTo(0, fistCoordinateY)
    ctx2.lineTo(0, 0);
    ctx2.lineTo(imageDims, 0);
    ctx2.lineTo(imageDims, imageDims);
    ctx2.lineTo(0, imageDims);
    ctx2.lineTo(0, fistCoordinateY);
  
    ctx2.closePath();
    ctx2.fill();
  }


function saveLandmarks(path, detections){
    // console.log(detections);
    const resizedLandmarks = resizeLandmarks(detections.landmarks._positions, detections.detection._box, "long");
    const rollAngle = detections.angle.roll;
    const rotatedLandmarks = rotateLandmarks(resizedLandmarks, rollAngle);
    fs.writeFile(path + "txt", landmarkListToString(rotatedLandmarks), function(err){
         if (err) return console.log(err);
    });
    return rotatedLandmarks;
}

function saveCroppedMaskedImage(path, use, useSide, detections, canvasInput, canvasCropped, rotatedLandmarks){
    const rollAngle = detections.angle.roll;
    cropRotateFace(use._x, use._y, use._width, use._height, rollAngle, useSide, canvasInput, canvasCropped);
    maskFaceNew(rotatedLandmarks, canvasCropped);
    const buffer = canvasCropped.toBuffer('image/png')
    fs.writeFileSync(path, buffer)

}

function saveHogs(filePath){

        let originalImage = filePath;
    
        if ((originalImage.includes("png") || (originalImage.includes("jpg")))){
    
            
        let directoryPath = path.dirname(filePath);
    
        let newHogsDirectory = "NH_" + directoryPath;
        if (!fs.existsSync(newHogsDirectory)){
            fs.mkdirSync(newHogsDirectory, { recursive: true });
        }
        let outputHogs = 'NH_' + originalImage.substring(0, originalImage.length - 3);
    
        Image.load(originalImage).then(function (originalImage) {
            var descriptor = hog.extractHOG(originalImage);
            fs.writeFile(outputHogs + "txt", descriptor.toString(), function(err){
                   if (err) return console.log(err);
            })
        });
    
    }
}

async function main() {

    faceapi.env.monkeyPatch({ Canvas: canvas.Canvas, Image: canvas.Image, ImageData: canvas.ImageData });

    await faceapi.nets.ssdMobilenetv1.loadFromDisk('model'); // load models from a specific patch
    await faceapi.nets.faceLandmark68Net.loadFromDisk('model');
    await faceapi.nets.ageGenderNet.loadFromDisk('model');
    await faceapi.nets.faceRecognitionNet.loadFromDisk('model');
    await faceapi.nets.faceExpressionNet.loadFromDisk('model');
    const options = new faceapi.SsdMobilenetv1Options({ minConfidence: 0.1, maxResults: 10 }); // set model options

    let imageDirectory = 'dataset';
    let i = 0;
    for (const filePath of walkSync(imageDirectory)) {
      i ++
    }
    console.log("Total: " + i + " images");

    let startTime = new Date().getTime();

    let j = 0;
    for (const filePath of walkSync(imageDirectory)) {
        j ++;
        
        let originalImage = filePath;
        

        if ((originalImage.includes("png") || originalImage.includes("jpg"))){ // current file is an image file
            createDirs(filePath); // create directories to store results
            var previewImage = fs.readFileSync(filePath);
            const image = await canvas.loadImage(originalImage);

            const canvasInput = canvas.createCanvas(480, 360);
            const ctx1 = canvasInput.getContext("2d");
            const canvasCropped = canvas.createCanvas(imageDims, imageDims);
            const ctx2 = canvasCropped.getContext("2d");
            const displaySize = { width: canvasInput.width, height: canvasInput.height }

            var hRatio = canvasInput.width / image.width;
            var vRatio = canvasInput.height / image.height;
            var ratio  = Math.min ( hRatio, vRatio );

	        ctx1.drawImage(image, 0,0, image.width, image.height, 0,0,image.width*ratio, image.height*ratio); // draw image to canvas without changing aspect ratio



            const detections = await faceapi.detectSingleFace(canvasInput)
            .withFaceLandmarks()
            .withFaceExpressions()
            .withFaceDescriptor()
            .withAgeAndGender();

            // const buffer = fs.readFileSync(originalImage); // load image as binary
            // const decodeT = faceapi.tf.node.decodeImage(buffer, 3); // decode binary buffer to rgb tensor
            // const expandT = faceapi.tf.expandDims(decodeT, 0); // add batch dimension to tensor
            // const result = await faceapi.detectAllFaces(expandT, options) // run detection
            // .withFaceLandmarks()
            // .withFaceExpressions()
            // .withFaceDescriptors()
            // .withAgeAndGender();
            // faceapi.tf.dispose([decodeT, expandT]);
            // let f = result[0]["detection"]["_box"];

            if (typeof detections != "undefined") {
                const resizedDetections = faceapi.resizeResults(detections, displaySize);
                const alr = resizedDetections.alignedRect._box;
                const det = resizedDetections.detection._box;
                const use = det;   // define here the bounding box that we are using
                const useSide = "long"; // long: resize to fit bigger side in canvasCropped, short: resize to let smaller side fill canvasCropped

                let landmarkPath = 'NL_' + originalImage.substring(0, originalImage.length - 3)
                let imagePath = 'NI_' + originalImage.substring(0, originalImage.length - 3)
                let rotatedLandmarks = saveLandmarks(landmarkPath, detections);
                saveCroppedMaskedImage(imagePath, use, useSide, detections, canvasInput, canvasCropped, rotatedLandmarks);
            }

        console.clear();
        console.log("Step 1/2 - " + j + "/" + i + " - " + filePath + " - " + getRemainingTimeString(startTime, j, i));
        console.log("\n")
        // console.png(previewImage);

    
    }
}
j = 0;

for (const filePath of walkSync(imageDirectory)) {
  j++
  console.clear();
  console.log("Step 2/2 - " + j + "/" + i);
  saveHogs(filePath);

}
}
    
  
main();
