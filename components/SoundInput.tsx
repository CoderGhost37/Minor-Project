import { useState } from 'react';

const SoundInput = () => {
  const [recording, setRecording] = useState(false);
  const [audioURL, setAudioURL] = useState('');
  const [mediaRecorder, setMediaRecorder] = useState(null);

  const startRecording = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const newMediaRecorder = new MediaRecorder(stream);

    const chunks: BlobPart[] | undefined = [];
    newMediaRecorder.addEventListener('dataavailable', event => {
      chunks.push(event.data);
    });

    newMediaRecorder.addEventListener('stop', () => {
      const audioBlob = new Blob(chunks);
      const audioURL = URL.createObjectURL(audioBlob);
      setAudioURL(audioURL);
      setRecording(false);
    });

    newMediaRecorder.start();
    setRecording(true);
    setMediaRecorder(newMediaRecorder);
  };

  const stopRecording = () => {
    mediaRecorder.stop();
  };

  return (
    <div className="flex flex-col items-center justify-center">
      {!recording && (
        <button
          className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
          onClick={startRecording}
        >
          Start Recording
        </button>
      )}

      {recording && (
        <button
          className="bg-red-500 hover:bg-red-700 text-white font-bold py-2 px-4 rounded"
          onClick={stopRecording}
        >
          Stop Recording
        </button>
      )}

      {audioURL && (
        <audio
          className="mt-4"
          src={audioURL}
          controls
        />
      )}
    </div>
  );
};

export default SoundInput;
