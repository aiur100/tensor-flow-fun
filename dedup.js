
//console.log("Hello dedup",Compressor);

/**
 * Upload original image
 * - resize image
 * - crop image
 * - gray scale image
 * - Add various quailities to the classifier
 * - Add a few random photos
 * 
 * Upload challenge image
 * - resize image
 * - crop image
 * - gray scale image
 * - check if it matches 
 * 
 */
let previewContainer;

const reader = new FileReader();
reader.onloadend = function() {
	previewContainer.src = reader.result;
}


const original  = document.getElementById("original");
const originalPreview = document.getElementById("originalPreview");

original.onchange = event => {
	previewContainer = originalPreview;
	reader.readAsDataURL(event.target.files[0]);
}


const challenge  = document.getElementById("challenge");
const challengePreview = document.getElementById("challengePreview");

challenge.onchange = event => {
	previewContainer = challengePreview;
	reader.readAsDataURL(event.target.files[0]);
}

const classifier = knnClassifier.create();
let net;

(async function(){

	// Load the model.
	net = await mobilenet.load();
	console.log('Successfully loaded model');
	const classes = ["correct","incorrect"];

	const correct = document.getElementById("correct");
	const incorrect = document.getElementById("incorrect");
	const predictButton = document.getElementById("predict");
	const predictOutput = document.getElementById("prediction");

	correct.onclick = event => {
		// Get the intermediate activation of MobileNet 'conv_preds' and pass that
		// to the KNN classifier.
		const activation = net.infer(originalPreview, true);
		console.log("Trained correct...",originalPreview);
		// Pass the intermediate activation to the classifier.
		classifier.addExample(activation, 0);
	}

	incorrect.onclick = event => {
		// Get the intermediate activation of MobileNet 'conv_preds' and pass that
		// to the KNN classifier.
		const activation = net.infer(originalPreview, true);
		console.log("Trained incorrect...");
		// Pass the intermediate activation to the classifier.
		classifier.addExample(activation, 1);
	}

	predictButton.onclick = async event => {
		// Get the activation from mobilenet from the webcam.
		const activation = net.infer(challengePreview, 'conv_preds');
		// Get the most likely class and confidence from the classifier module.
		const result = await classifier.predictClass(activation);
		const confidence = parseInt((result.confidences[result.label]*100).toFixed(2));
		console.log(confidence,result.label,typeof confidence);
		const message = confidence > 95 && result.label == 0   ? 
						"This is the correct image (duplicate)" :
						"This is the incorrect image";
		predictOutput.innerHTML = message;
	}

})().catch(error => {
	console.error("Whoops...",error);
});