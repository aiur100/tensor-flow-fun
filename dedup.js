
//console.log("Hello dedup",Compressor);
console.log(Jimp);

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

async function getJimpImage(url){
	return await Jimp.read({ url });
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function process(file,ratio=0.3,quality=100,base64=true){
	const jimpImg = await (typeof file === "string" ? 
					getJimpImage(file) : 
					Jimp.read(file));
	const resized = await jimpImg.resize(640,480);
	const x = parseInt(640 * ratio);
	const y = parseInt(480 * ratio);
	const qcrop = await resized.quality(quality);
	const cropped = await qcrop.crop(x, y, 200, 200).rgba(false).greyscale();
	//console.log("JIMP IMG",cropped);
	return await (base64 ? cropped.getBase64Async("image/jpeg") : cropped.getBufferAsync("image/jpeg"));
}

reader.onloadend = async function() {
	previewContainer.src = reader.result;//await process(reader.result);
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

	const train = document.getElementById("correct");
	const predictButton = document.getElementById("predict");
	const predictOutput = document.getElementById("prediction");

	const qualities = [100,50,94,93,92,91,90,80,81,82,70];
	const ratios = [0.2,0.3];

	/**
	 * Train the model 
	 * 
	 * @param {*} event 
	 */
	train.onclick = async event => {

		train.disabled = true;
		const loadingSpinner = document.getElementById("training-loading");
		const trainingText = document.getElementById("train-text");
		trainingText.innerText = "Training correct samples...";
		loadingSpinner.classList.remove("d-none");
		const rawPhoto = originalPreview.src;

		// Get the intermediate activation of MobileNet 'conv_preds' and pass that
		// to the KNN classifier.
		await Promise.all(qualities.map(async qualityPercentage => {

			const ratio = ratios[ Math.floor(Math.random() * ratios.length) ];
			const processed = await process(rawPhoto, ratio, qualityPercentage);

			previewContainer.src = processed;
			const img = tf.browser.fromPixels(previewContainer);
			const activation = net.infer(img, true);

			console.log(`Trained correct...${qualityPercentage}`);

			classifier.addExample(activation, 0);
		}));

		const incorrect = [
			"inc_1",
			"inc_2",
			"inc_3",
			"inc_4",
			"inc_5",
			"inc_6",
			"inc_7",
			"inc_8",
			"inc_9",
			"inc_5",
			"inc_6",
			"inc_7",
			"inc_8",
			"inc_9",
			"inc_5",
			"inc_6",
			"inc_7",
			"inc_8",
			"inc_9"
		];
		trainingText.innerText = "Training incorrect samples...";

		await Promise.all(incorrect.map(async inc => {
			const ratio = ratios[Math.floor(Math.random() * ratios.length)];
			const incorrect = document.getElementById(inc);
			const processed = await process(incorrect.src,ratio);
			previewContainer.src = processed;
			const img = tf.browser.fromPixels(previewContainer);
			const activation = net.infer(img, true);
			console.log(`Trained incorrect...${inc}`);
			// Pass the intermediate activation to the classifier.
			classifier.addExample(activation, 1);
		}));

		trainingText.innerText = "Train";
		loadingSpinner.classList.add("d-none");
	}

	predictButton.onclick = async event => {
		const preview = await process(challengePreview.src,0.2);
		challengePreview.src = preview;
		// Get the activation from mobilenet from the webcam.
		const activation = net.infer(challengePreview, 'conv_preds');

		// Get the most likely class and confidence from the classifier module.
		const result = await classifier.predictClass(activation);
		const confidence = parseInt((result.confidences[result.label]*100).toFixed(2));
		console.log({ result });

		const message = confidence > 95 && result.label == 0   ? 
						"This is the correct image (duplicate)" :
						"This is the incorrect image";

		predictOutput.innerHTML = `${message}<br>
			<b>Confidence:</b>${confidence}%<br>
			<b>Label:</b>${classes[result.label]}
		`;
	}

})().catch(error => {
	console.error("Whoops...",error);
});