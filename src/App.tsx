import * as React from 'react';
import './App.css'
import { SuperVADStreamEngine } from './supervad/engine';
import { AsyncLock } from './utils/lock';

function App() {
  // const [loading, setLoading] = React.useState(false);
  const [state, setState] = React.useState<{
    state: 'empty'
  } | {
    state: 'loading'
  } | {
    state: 'loaded', session: SuperVADStreamEngine
  } | {
    state: 'online', session: SuperVADStreamEngine, stream: MediaStream, segments: string[]
  }>({ state: 'empty' });

  // Load model
  React.useEffect(() => {
    let exited = false;
    if (state.state === 'loading') {
      (async () => {
        const session = await SuperVADStreamEngine.create({
          deactivation_threshold: 0.6,
          deactivation_tokens: 20,
          activation_threshold: 0.8,
          activation_tokens: 1,
          prebuffer_tokens: 20,
          min_active_tokens: 10,
        });
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
      let segments: string[] = [];

      processor.onaudioprocess = function (e) {
        const input = e.inputBuffer.getChannelData(0);

        lock.inLock(async () => {
          pending = concat(pending, input);
          while (pending.length > SuperVADStreamEngine.TOKEN_SIZE) {
            const output = await session.process(pending.subarray(0, SuperVADStreamEngine.TOKEN_SIZE));
            pending = pending.subarray(SuperVADStreamEngine.TOKEN_SIZE);
            if (output.state !== 'unchanged') {
              console.log(output);
            }
            if (output.state === 'complete') {

              const wavData = float32ArrayToWav(output.buffer, 16000); // Convert to WAV format
              const blob = new Blob([wavData], { type: 'audio/wav' }); // Create a blob
              const audioUrl = URL.createObjectURL(blob); // Create an object URL

              segments = [...segments, audioUrl];
              setState({ state: 'online', session: state.session, stream, segments });
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
          Press button below to download model first (~20mb)
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
            Online
          </button>
        )}
        {state.state === 'online' && (
          <div className='samples'>
            {state.segments.map((segment, index) => (
              <audio key={index} controls src={segment} />
            ))}
          </div>
        )}
      </div>
      <p className="read-the-docs">
        Voice activity detection is all you need
      </p>
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