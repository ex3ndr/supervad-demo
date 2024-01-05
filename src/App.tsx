import * as React from 'react';
import './App.css'
import { SuperVADEngine, SuperVADRealtime, optimalParameters } from 'supervad';
import { AsyncLock } from './utils/lock';
import WhisperWorker from './WhisperWorker?worker';

const worker = new WhisperWorker();
let workerId = 0;
async function workerRequest(type: string, data?: Float32Array) {
  return new Promise((resolve) => {
    const id = workerId++;
    const callback = (e: MessageEvent) => {
      if (e.data.id === id) {
        worker.removeEventListener('message', callback);
        resolve(e.data);
      }
    }
    worker.addEventListener('message', callback);
    worker.postMessage({ id, type, data });
  });
}

const AudioPreview = React.memo((props: { url: string, raw: Float32Array }) => {

  const [text, setText] = React.useState<string | null>(null);
  React.useEffect(() => {
    workerRequest('transcribe', props.raw).then((v) => {
      setText((v as any).text);
    });
  }, [props.url]);

  return (
    <>
      <audio controls src={props.url} />
      {text !== null && (<span>{text}</span>)}
      {text === null && (<span>Transcribing...</span>)}
    </>
  )
});

function App() {
  // const [loading, setLoading] = React.useState(false);
  const [state, setState] = React.useState<{
    state: 'empty'
  } | {
    state: 'loading'
  } | {
    state: 'loaded', session: SuperVADRealtime
  } | {
    state: 'online', session: SuperVADRealtime, stream: MediaStream, segments: { url: string, raw: Float32Array }[]
  }>({ state: 'empty' });

  // Load model
  React.useEffect(() => {
    let exited = false;
    if (state.state === 'loading') {
      (async () => {

        // const transformers = await import('@xenova/transformers');
        // transformers.env.localModelPath = 'https://shared.korshakov.com/models/hugginface/';
        // transformers.env.remoteHost = 'https://shared.korshakov.com/models/hugginface/';

        // Download whisper
        await workerRequest('init');

        // Load model
        const engine = await SuperVADEngine.create('./supervad.onnx');

        // Create session
        const params = optimalParameters();
        const session = SuperVADRealtime.create(engine, params);
        if (exited) {
          return;
        }
        setState({ state: 'loaded', session });
      })()
    }
    return () => {
      exited = true;
    }
  }, [state]);

  // Load microphone
  const doLoadMic = React.useCallback(() => {

    // Check state
    if (state.state !== 'loaded') {
      return;
    }
    const session = state.session;

    // Start microphone
    navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {

      const audioContext = new AudioContext({ sampleRate: 16000 });
      const source = audioContext.createMediaStreamSource(stream);
      const processor = audioContext.createScriptProcessor(2048, 1, 1);
      const lock = new AsyncLock();
      let pending: Float32Array = new Float32Array(0);
      let segments: { url: string, raw: Float32Array }[] = [];

      processor.onaudioprocess = function (e) {
        const input = e.inputBuffer.getChannelData(0);

        lock.inLock(async () => {
          pending = concat(pending, input);
          while (pending.length > SuperVADEngine.TOKEN_SIZE) {
            const output = await session.process(pending.subarray(0, SuperVADEngine.TOKEN_SIZE));
            pending = pending.subarray(SuperVADEngine.TOKEN_SIZE);
            if (output) {
              console.log(output);
              if (output.kind === 'complete') {

                const wavData = float32ArrayToWav(output.buffer, 16000); // Convert to WAV format
                const blob = new Blob([wavData], { type: 'audio/wav' }); // Create a blob
                const audioUrl = URL.createObjectURL(blob); // Create an object URL

                segments = [...segments, { url: audioUrl, raw: output.buffer }];
                setState({ state: 'online', session: state.session, stream, segments });
              }
            }
          }

        });
      };

      source.connect(processor);
      processor.connect(audioContext.destination);


      setState({ state: 'online', session: state.session, stream, segments });
    });
  }, [state]);

  return (
    <>
      <h1>SuperVAD</h1>
      <div className="card">
        <p>
          Press button below to download models<br /> (SuperVAD + Whisper) first (~60mb)
        </p>

        {state.state === 'empty' && (
          <button onClick={() => setState({ state: 'loading' })}>
            Download model
          </button>
        )}
        {state.state === 'loading' && (
          <button disabled>
            Model is downloading...
          </button>
        )}
        {state.state === 'loaded' && (
          <button onClick={doLoadMic}>
            Allow microphone access
          </button>
        )}
        {state.state === 'online' && (
          <button>
            Speak!
          </button>
        )}
        {state.state === 'online' && (
          <div className='samples'>
            {state.segments.map((segment, index) => (
              <AudioPreview key={index} url={segment.url} raw={segment.raw} />
            ))}
          </div>
        )}
      </div>
      {state.state === 'online' && state.segments.length === 0 && (<p className="read-the-docs">
        Voice segments will appear here
      </p>
      )}
      {state.state !== 'online' && (<p className="read-the-docs">
        Voice activity detection is all you need
      </p>
      )}
    </>
  )
}

export default App

function concat(a: Float32Array, b: Float32Array) {
  const c = new Float32Array(a.length + b.length);
  c.set(a);
  c.set(b, a.length);
  return c;
}

function float32ArrayToWav(float32Array: Float32Array, sampleRate: number): ArrayBuffer {
  const numChannels: number = 1;
  const numSamples: number = float32Array.length;
  const format: number = 1; // PCM - integer samples
  const bitDepth: number = 16;

  const blockAlign: number = numChannels * bitDepth / 8;
  const byteRate: number = sampleRate * blockAlign;
  const dataSize: number = numSamples * blockAlign;

  const buffer: ArrayBuffer = new ArrayBuffer(44 + dataSize);
  const view: DataView = new DataView(buffer);

  function writeString(view: DataView, offset: number, string: string): void {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  function floatTo16BitPCM(output: DataView, offset: number, input: Float32Array): void {
    for (let i = 0; i < input.length; i++, offset += 2) {
      const s: number = Math.max(-1, Math.min(1, input[i]));
      output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
  }

  // RIFF header
  writeString(view, 0, 'RIFF');  // ChunkID
  view.setUint32(4, 36 + dataSize, true); // ChunkSize
  writeString(view, 8, 'WAVE'); // Format

  // fmt subchunk
  writeString(view, 12, 'fmt '); // Subchunk1ID
  view.setUint32(16, 16, true); // Subchunk1Size
  view.setUint16(20, format, true); // AudioFormat
  view.setUint16(22, numChannels, true); // NumChannels
  view.setUint32(24, sampleRate, true); // SampleRate
  view.setUint32(28, byteRate, true); // ByteRate
  view.setUint16(32, blockAlign, true); // BlockAlign
  view.setUint16(34, bitDepth, true); // BitsPerSample

  // data subchunk
  writeString(view, 36, 'data'); // Subchunk2ID
  view.setUint32(40, dataSize, true); // Subchunk2Size

  floatTo16BitPCM(view, 44, float32Array);

  return buffer;
}