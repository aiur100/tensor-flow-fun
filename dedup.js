
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
	const cropped = await qcrop.crop(x, y, 250, 200).rgba(false).greyscale().posterize(4);
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

function getByteNumbers(base64string){
	const b = atob(base64string)
	let byteNumbers = new Array(b.length);
	for (let i = 0; i < b.length; i++) {
		byteNumbers[i] = b.charCodeAt(i);
	}
	return byteNumbers;
}

(async function(){

	// Load the model.
	net = await mobilenet.load();
	console.log('Successfully loaded model');

	const classes = ["correct","incorrect"];

	const train = document.getElementById("correct");
	const predictButton = document.getElementById("predict");
	const predictOutput = document.getElementById("prediction");

	const qualities = [100,50,40,2];
	const ratios = [0.3,0.4];

	function createImage(base64str){
		return new Promise((resolve, reject) => {
			const im = new Image();
			im.crossOrigin = 'anonymous';
			im.src = base64str;
			im.onload = () => {
				resolve(im)
			};
		})
	}

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

		for(let i = 0; i < qualities.length; i++){
			const qualityPercentage = qualities[i];
			const ratio = ratios[ Math.floor(Math.random() * ratios.length) ];

			const processed = await process(rawPhoto, ratio, qualityPercentage);

			const imgEl = await createImage(processed);

			document.getElementById("leftContainer").appendChild(imgEl);
			
			const img = tf.browser.fromPixels(imgEl);
			const activation = net.infer(img, true);
			activation.print();

			console.log(`Trained correct...${qualityPercentage}`);

			classifier.addExample(activation, 0);
			await sleep(1000);
		}

		
		trainingText.innerText = "Training incorrect samples...";

		const incorrect = [
			"inc_1",
			"inc_2",
			"inc_3",
			"inc_4",
			"inc_5",
			"inc_6",
			"inc_7",
			"inc_8"
		];

		for(let x = 0; x < incorrect.length; x++){
			const inc = incorrect[x];
			const ratio = ratios[ Math.floor(Math.random() * ratios.length) ];
			const incElm = document.getElementById(inc);
			const processed = await process(incElm.src,ratio);

			const imgEl = await createImage(processed);
			//previewContainer.src = processed;
			console.log(previewContainer.src);
			document.getElementById("leftContainer").appendChild(imgEl);

			const img = tf.browser.fromPixels(imgEl);
			const activation = net.infer(img, true);
			activation.print();			
			console.log(`Trained incorrect...${inc}`);
			// Pass the intermediate activation to the classifier.
			classifier.addExample(activation, 1);
			await sleep(1000);
		}

		trainingText.innerText = "Train";
		loadingSpinner.classList.add("d-none");
		const classiferData = classifier.getClassifierDataset();
		console.log(classiferData);
	}

	predictButton.onclick = async event => {
		const preview = await process(challengePreview.src,0.3);
		challengePreview.src = preview;
		const imgEl = await createImage(preview);

		const img = tf.browser.fromPixels(imgEl);
		const activation = net.infer(img, 'conv_preds');

		// Get the most likely class and confidence from the classifier module.
		const result = await classifier.predictClass(activation);
		const confidence = parseInt((result.confidences[result.label]*100).toFixed(2));
		console.log({ result });

		/*
		const message = confidence > 95 && result.label == 0   ? 
						"This is the correct image (duplicate)" :
						"This is the incorrect image";
						*/
		predictOutput.innerHTML = JSON.stringify(result,null,2);
						/*
		predictOutput.innerHTML = `${message}<br>
			<b>Confidence:</b>${confidence}%<br>
			<b>Label:</b>${classes[result.label]}
		`;
		*/
	}

})().catch(error => {
	console.error("Whoops...",error);
});