import { env, AutomaticSpeechRecognitionPipeline, pipeline, AutomaticSpeechRecognitionOutput } from '@xenova/transformers';
import { AsyncLock } from './utils/lock';

env.localModelPath = 'https://shared.korshakov.com/models/hugginface/';
env.remoteHost = 'https://shared.korshakov.com/models/hugginface/';

const lock = new AsyncLock();
let whisper: AutomaticSpeechRecognitionPipeline | null = null;

self.onmessage = (event) => {
    lock.inLock(async () => {

        // Handle init
        if (event.data.type === 'init') {
            if (whisper) {
                self.postMessage({ type: 'ready', id: event.data.id });
                return;
            } else {
                whisper = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny.en');
                self.postMessage({ type: 'ready', id: event.data.id });
                return;
            }
        }

        if (event.data.type === 'transcribe') {
            let data = event.data.data;
            let res = await whisper!(data) as AutomaticSpeechRecognitionOutput;
            self.postMessage({ type: 'ready', id: event.data.id, text: res.text });
        }
    });
}