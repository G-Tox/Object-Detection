let video = document.querySelector("#videoElement");
let canvas = document.querySelector("#canvasOutput");
let ctx = canvas.getContext('2d');

var localMediaStream = null;

var socket = io.connect('http://localhost:5000');
socket.on('connect', function(){
        console.log("Connected...!", socket.connected)});

var constraints = {
    video: {
      width: { min: 640 },
      height: { min: 480 }
    }
  };


function sendSnapshot() {
    if (!localMediaStream) {
      return;
    }

    ctx.drawImage(video, 0, 0, video.videoWidth, video.videoHeight, 0, 0, 300, 150);
    var type = "image/jpeg"
    let dataURL = canvas.toDataURL('image/jpeg');
    dataURL = dataURL.replace('data:' + type + ';base64,', ''); 
    socket.emit('image', dataURL);
  }



if (navigator.mediaDevices.getUserMedia) {
  navigator.mediaDevices.getUserMedia(constraints)
  .then(function (stream) {
    video.srcObject = stream;
    localMediaStream = stream;
  })
  .catch(function (err0r) {
    console.log(err0r)
    console.log("Something went wrong!");
  });
}

setInterval(function () {
      sendSnapshot();
    }, 50);

socket.on('response_back', function(image){
  const image_id = document.getElementById('image');
  image_id.src = image;
});

